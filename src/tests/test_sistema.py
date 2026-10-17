"""Testes do modelo preditivo, da base de protocolos e do pipeline multiagente.

Rodar a partir da raiz do repositório:  pytest
"""

import asyncio
import json

import pytest
from agents import SQLiteSession, set_tracing_disabled
from pydantic import ValidationError

from multiagente.config import PASTA_PACIENTES
from multiagente.contexto import ContextoAtendimento
from multiagente.esquemas import DadosPaciente, RecomendacaoFinal
from multiagente.guardrails import verificar_consistencia
from multiagente.preditor import carregar_modelo, prever
from multiagente.protocolos import avaliar_protocolos
from multiagente.sistema import executar_atendimento, executar_offline, perguntar
from tests.modelo_roteirizado import ModeloRoteirizado

set_tracing_disabled(True)


def paciente(nome: str) -> tuple[str, DadosPaciente]:
    bruto = json.loads((PASTA_PACIENTES / f"{nome}.json").read_text(encoding="utf-8"))
    return bruto.pop("paciente_id"), DadosPaciente.model_validate(bruto)


# ---------- Parte 1: modelo preditivo ----------

def test_modelo_salvo_tem_metricas_minimas():
    pacote = carregar_modelo()
    assert pacote["algoritmo"] == "Random Forest"
    assert pacote["metricas_teste"]["acuracia"] >= 0.95
    assert pacote["metricas_teste"]["recall"] >= 0.95


def test_novo_caso_do_enunciado_e_alto_risco():
    _, dados = paciente("novo_caso_enunciado")
    risco = prever(dados)
    assert risco.probabilidade >= 0.9
    assert risco.classificacao == "ALTO"
    assert len(risco.fatores_de_risco) == 6


def test_paciente_baixo_risco():
    _, dados = paciente("paciente_baixo_risco")
    assert prever(dados).classificacao == "BAIXO"


# ---------- Validação de entrada ----------

def test_aceita_chave_com_espaco_do_enunciado():
    _, dados = paciente("novo_caso_enunciado")
    assert dados.disponibilidade_recursos == 25


def test_rejeita_valor_impossivel():
    _, dados = paciente("novo_caso_enunciado")
    with pytest.raises(ValidationError):
        DadosPaciente.model_validate({**dados.model_dump(), "spo2_media": 130})


# ---------- Base de protocolos ----------

def test_protocolos_do_novo_caso():
    _, dados = paciente("novo_caso_enunciado")
    resultado = avaliar_protocolos(dados, "ALTO")
    ids = [p.id for p in resultado.protocolos_acionados]
    assert ids == ["P10", "P02", "P05", "P08"]
    assert any("spo2_media = 89" in a for a in resultado.parametros_em_atencao)


def test_caso_critico_aciona_protocolos_imediatos_primeiro():
    _, dados = paciente("paciente_critico_choque")
    resultado = avaliar_protocolos(dados, "ALTO")
    assert [p.id for p in resultado.protocolos_acionados][:3] == ["P01", "P03", "P09"]
    assert len(resultado.protocolos_acionados) == 10


# ---------- Guardrail de saída ----------

def test_guardrail_detecta_protocolo_inventado():
    pid, dados = paciente("novo_caso_enunciado")
    recomendacao, ctx = executar_offline(dados, pid)
    adulterada = recomendacao.model_copy(update={
        "protocolos_sugeridos": recomendacao.protocolos_sugeridos + ["Protocolo inventado"],
    })
    assert verificar_consistencia(ctx, recomendacao) == []
    assert verificar_consistencia(ctx, adulterada)


# ---------- Pipeline multiagente com o SDK (modelo de linguagem simulado) ----------

def test_pipeline_multiagente_completo(tmp_path):
    pid, dados = paciente("novo_caso_enunciado")
    esperado, _ = executar_offline(dados, pid)
    session = SQLiteSession("teste", tmp_path / "historico.db")

    recomendacao, ctx, meta = asyncio.run(
        executar_atendimento(dados, pid, session=session, modelo=ModeloRoteirizado(), verbose=False)
    )

    assert isinstance(recomendacao, RecomendacaoFinal)
    assert not meta["guardrail_bloqueou"]
    assert meta["ultimo_agente"] == "Agente Planejador de Resposta"
    assert recomendacao.probabilidade_pico_risco == esperado.probabilidade_pico_risco
    assert recomendacao.protocolos_sugeridos == esperado.protocolos_sugeridos

    handoffs = [e for e in ctx.trilha if e.startswith("Handoff")]
    assert handoffs == [
        "Handoff para Agente Analista de Risco",
        "Handoff para Agente Orquestrador",
        "Handoff para Agente Especialista em Protocolos",
        "Handoff para Agente Orquestrador",
        "Handoff para Agente Planejador de Resposta",
    ]
    assert "Guardrail de saída: aprovado" in ctx.trilha
    assert len(asyncio.run(session.get_items())) > 10


def test_guardrail_bloqueia_llm_que_altera_probabilidade():
    pid, dados = paciente("novo_caso_enunciado")
    recomendacao, ctx, meta = asyncio.run(
        executar_atendimento(dados, pid, modelo=ModeloRoteirizado(adulterar=True), verbose=False)
    )
    assert meta["guardrail_bloqueou"]
    assert any("Probabilidade divergente" in p for p in meta["problemas"])
    # A resposta de contingência usa o valor correto do modelo de ML.
    assert recomendacao.probabilidade_pico_risco == ctx.risco.probabilidade


def test_guardrail_de_entrada_bloqueia_prescricao(tmp_path):
    pid, dados = paciente("novo_caso_enunciado")
    _, ctx = executar_offline(dados, pid)
    session = SQLiteSession("teste", tmp_path / "historico.db")
    resposta = asyncio.run(
        perguntar("Qual a dose de amiodarona em mg?", ctx, session, modelo=ModeloRoteirizado(), verbose=False)
    )
    assert "bloqueado" in resposta


def test_pergunta_de_acompanhamento_usa_orquestrador(tmp_path):
    pid, dados = paciente("novo_caso_enunciado")
    _, ctx = executar_offline(dados, pid)
    assert isinstance(ctx, ContextoAtendimento) and ctx.etapa_atual() == "CONCLUIDO"
    session = SQLiteSession("teste", tmp_path / "historico.db")
    resposta = asyncio.run(
        perguntar("Por que o risco ficou alto?", ctx, session, modelo=ModeloRoteirizado(), verbose=False)
    )
    assert "concluído" in resposta
