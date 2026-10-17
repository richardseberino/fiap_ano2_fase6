"""Tools (function tools) disponibilizadas aos agentes.

As tools leem os dados do paciente do contexto compartilhado e gravam nele os seus
resultados. O LLM decide quando chamar cada tool, mas não digita nenhum valor clínico,
o que evita erros de transcrição em números sensíveis.
"""

from __future__ import annotations

import json

from agents import RunContextWrapper, function_tool

from . import preditor
from .contexto import AVISO_PADRAO, ContextoAtendimento, montar_plano
from .protocolos import avaliar_protocolos, listar_base


def _json(dados) -> str:
    return json.dumps(dados, ensure_ascii=False, indent=2)


@function_tool
def consultar_estado_atendimento(ctx: RunContextWrapper[ContextoAtendimento]) -> str:
    """Consulta a etapa atual do atendimento e os resultados já produzidos pelos agentes."""
    c = ctx.context
    return _json({
        "paciente_id": c.paciente_id,
        "etapa_atual": c.etapa_atual(),
        "dados_paciente": c.paciente.model_dump(),
        "risco": c.risco.model_dump() if c.risco else None,
        "protocolos": c.protocolos.model_dump() if c.protocolos else None,
        "plano": [a.model_dump() for a in c.plano] if c.plano else None,
    })


@function_tool
def prever_risco_cardiaco(ctx: RunContextWrapper[ContextoAtendimento]) -> str:
    """Executa o modelo de Machine Learning treinado (Random Forest) sobre os dados do
    paciente atual e retorna a probabilidade de pico de risco, a classificação e os
    fatores de risco identificados."""
    c = ctx.context
    c.risco = preditor.prever(c.paciente)
    c.registrar(f"Modelo de ML: probabilidade {c.risco.probabilidade:.2f} ({c.risco.classificacao})")
    return _json(c.risco.model_dump())


@function_tool
def consultar_protocolos(ctx: RunContextWrapper[ContextoAtendimento]) -> str:
    """Cruza os dados do paciente e o risco previsto com a base de protocolos simulada.
    Retorna os protocolos acionados, ordenados por prioridade, e os parâmetros em zona
    de atenção. Exige que a predição de risco já tenha sido feita."""
    c = ctx.context
    if c.risco is None:
        return "ERRO: a predição de risco ainda não foi feita. Devolva ao orquestrador."
    c.protocolos = avaliar_protocolos(c.paciente, c.risco.classificacao)
    c.registrar(f"Base de protocolos: {len(c.protocolos.protocolos_acionados)} protocolo(s) acionado(s)")
    return _json({
        **c.protocolos.model_dump(),
        "total_protocolos_na_base": len(listar_base()),
    })


@function_tool
def planejar_resposta_operacional(ctx: RunContextWrapper[ContextoAtendimento]) -> str:
    """Consolida risco previsto e protocolos acionados em um plano de ações operacionais
    ordenado por prioridade, com responsável e prazo. Retorna todos os dados que devem
    compor a recomendação final."""
    c = ctx.context
    if c.risco is None or c.protocolos is None:
        return "ERRO: predição ou protocolos pendentes. Devolva ao orquestrador."
    c.plano = montar_plano(c.protocolos)
    c.registrar(f"Plano operacional: {len(c.plano)} ação(ões)")
    return _json({
        "paciente_id": c.paciente_id,
        "probabilidade_pico_risco": c.risco.probabilidade,
        "classificacao_risco": c.risco.classificacao,
        "fatores_de_risco": c.risco.fatores_de_risco,
        "protocolos_sugeridos": [p.protocolo for p in c.protocolos.protocolos_acionados],
        "acoes_operacionais": [a.model_dump() for a in c.plano],
        "parametros_em_atencao": c.protocolos.parametros_em_atencao,
        "aviso": AVISO_PADRAO,
    })
