"""Guardrails de entrada e de saída.

- Entrada (Orquestrador): bloqueia pedidos fora do escopo, como prescrição de
  medicamentos e doses, que um sistema de apoio à decisão não deve emitir.
- Saída (Planejador): confere se a recomendação do LLM é fiel ao contexto, ou seja,
  se probabilidade, classificação, protocolos e ações batem com o que o modelo de ML
  e a base de protocolos produziram. Qualquer divergência dispara o tripwire.
"""

from __future__ import annotations

import re
import unicodedata

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, input_guardrail, output_guardrail

from .config import TOLERANCIA_PROBABILIDADE
from .contexto import ORIGEM_ROTINA, ContextoAtendimento
from .esquemas import RecomendacaoFinal

_PADROES_FORA_ESCOPO = [
    r"\bprescrev", r"\bprescricao\b", r"\bdosagem\b", r"\bdose\b", r"\b\d+\s?mg\b",
    r"\bqual remedio\b", r"\bqual medicamento\b",
]


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def _texto_da_entrada(entrada) -> str:
    if isinstance(entrada, str):
        return entrada
    partes = []
    for item in entrada:
        conteudo = item.get("content") if isinstance(item, dict) else None
        if isinstance(conteudo, str):
            partes.append(conteudo)
        elif isinstance(conteudo, list):
            partes.extend(c.get("text", "") for c in conteudo if isinstance(c, dict))
    return " ".join(partes)


@input_guardrail(name="escopo_apoio_decisao")
async def guardrail_escopo(
    ctx: RunContextWrapper[ContextoAtendimento], agent: Agent, entrada
) -> GuardrailFunctionOutput:
    texto = _sem_acento(_texto_da_entrada(entrada))
    encontrados = [p for p in _PADROES_FORA_ESCOPO if re.search(p, texto)]
    return GuardrailFunctionOutput(
        output_info={"padroes_bloqueados": encontrados},
        tripwire_triggered=bool(encontrados),
    )


def verificar_consistencia(ctx: ContextoAtendimento, saida: RecomendacaoFinal) -> list[str]:
    """Lista as divergências entre a resposta do LLM e o contexto do atendimento."""
    problemas = []
    if ctx.risco is None or ctx.protocolos is None or ctx.plano is None:
        return ["O fluxo terminou antes de concluir predição, protocolos e plano."]

    if abs(saida.probabilidade_pico_risco - ctx.risco.probabilidade) > TOLERANCIA_PROBABILIDADE:
        problemas.append(
            f"Probabilidade divergente: LLM {saida.probabilidade_pico_risco} x modelo {ctx.risco.probabilidade}"
        )
    if saida.classificacao_risco != ctx.risco.classificacao:
        problemas.append(
            f"Classificação divergente: LLM {saida.classificacao_risco} x modelo {ctx.risco.classificacao}"
        )

    esperados = {p.protocolo for p in ctx.protocolos.protocolos_acionados}
    recebidos = set(saida.protocolos_sugeridos)
    if recebidos != esperados:
        faltando = esperados - recebidos
        inventados = recebidos - esperados
        if faltando:
            problemas.append(f"Protocolos omitidos: {sorted(faltando)}")
        if inventados:
            problemas.append(f"Protocolos fora da base ou não acionados: {sorted(inventados)}")

    origens_validas = esperados | {ORIGEM_ROTINA}
    for acao in saida.acoes_operacionais:
        if acao.protocolo_origem not in origens_validas:
            problemas.append(f"Ação sem protocolo de origem válido: {acao.acao!r}")

    if saida.paciente_id != ctx.paciente_id:
        problemas.append(f"Paciente divergente: {saida.paciente_id} x {ctx.paciente_id}")
    return problemas


@output_guardrail(name="consistencia_recomendacao")
async def guardrail_consistencia(
    ctx: RunContextWrapper[ContextoAtendimento], agent: Agent, saida: RecomendacaoFinal
) -> GuardrailFunctionOutput:
    problemas = verificar_consistencia(ctx.context, saida)
    ctx.context.registrar(
        "Guardrail de saída: aprovado" if not problemas else f"Guardrail de saída: BLOQUEADO {problemas}"
    )
    return GuardrailFunctionOutput(
        output_info={"problemas": problemas},
        tripwire_triggered=bool(problemas),
    )
