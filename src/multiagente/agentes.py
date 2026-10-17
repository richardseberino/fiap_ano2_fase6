"""Definição dos agentes e da topologia de handoffs.

Topologia hub-and-spoke: o Orquestrador é o ponto de entrada e decide, a cada
etapa, para qual especialista transferir o atendimento. O Analista de Risco e o
Especialista em Protocolos devolvem o controle ao Orquestrador. O Planejador de
Resposta encerra o fluxo com a saída estruturada ``RecomendacaoFinal``.

    Orquestrador ──► Analista de Risco ──► (volta) Orquestrador
    Orquestrador ──► Especialista em Protocolos ──► (volta) Orquestrador
    Orquestrador ──► Planejador de Resposta ──► RecomendacaoFinal

Cada handoff do Orquestrador só fica habilitado (``is_enabled``) quando a etapa
anterior já foi concluída, o que impede pular etapas ou repetir a predição.
"""

from __future__ import annotations

from agents import Agent, Model, ModelSettings, RunContextWrapper, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from .config import MODELO_LLM
from .contexto import ContextoAtendimento
from .esquemas import RecomendacaoFinal
from .ferramentas import (
    consultar_estado_atendimento,
    consultar_protocolos,
    planejar_resposta_operacional,
    prever_risco_cardiaco,
)
from .guardrails import guardrail_consistencia, guardrail_escopo

INSTRUCOES_ORQUESTRADOR = f"""{RECOMMENDED_PROMPT_PREFIX}
Você é o Agente Orquestrador de um sistema de apoio à decisão para eventos cardíacos.
Você coordena o fluxo entre agentes especialistas e nunca calcula risco nem escolhe
protocolos por conta própria.

Fluxo obrigatório:
1. Chame a tool consultar_estado_atendimento para saber a etapa_atual.
2. Conforme a etapa_atual:
   - PREDICAO_PENDENTE: transfira para o Agente Analista de Risco.
   - PROTOCOLOS_PENDENTES: transfira para o Agente Especialista em Protocolos.
   - PLANO_PENDENTE: transfira para o Agente Planejador de Resposta.
   - CONCLUIDO: responda à pergunta do usuário em português, de forma breve, usando
     apenas o histórico da conversa e os dados retornados pela tool. Não invente valores.
"""

INSTRUCOES_ANALISTA = f"""{RECOMMENDED_PROMPT_PREFIX}
Você é o Agente Analista de Risco. Sua única função é consultar o modelo preditivo.
1. Chame a tool prever_risco_cardiaco exatamente uma vez.
2. Em seguida, transfira imediatamente de volta para o Agente Orquestrador.
Não reescreva nem arredonde a probabilidade. Não escolha protocolos.
"""

INSTRUCOES_ESPECIALISTA = f"""{RECOMMENDED_PROMPT_PREFIX}
Você é o Agente Especialista em Protocolos. Sua função é cruzar o risco previsto com a
base de protocolos simulada.
1. Chame a tool consultar_protocolos exatamente uma vez.
2. Em seguida, transfira imediatamente de volta para o Agente Orquestrador.
Nunca sugira protocolos que não tenham sido retornados pela tool.
"""

INSTRUCOES_PLANEJADOR = f"""{RECOMMENDED_PROMPT_PREFIX}
Você é o Agente Planejador de Resposta. Você consolida as informações dos outros agentes
e produz a recomendação final estruturada.
1. Chame a tool planejar_resposta_operacional exatamente uma vez.
2. Preencha a RecomendacaoFinal copiando EXATAMENTE os valores retornados pela tool:
   paciente_id, probabilidade_pico_risco, classificacao_risco, fatores_de_risco,
   protocolos_sugeridos (texto idêntico), acoes_operacionais, parametros_em_atencao e aviso.
3. Escreva a justificativa em português, com 2 a 4 frases, relacionando os fatores de
   risco, a probabilidade prevista e os protocolos acionados.
Não acrescente, remova nem reescreva protocolos. Uma resposta divergente será bloqueada
pelo guardrail de saída.
"""


def _configuracao_modelo(modelo: str | Model) -> ModelSettings:
    # Modelos de raciocínio (família o* e gpt-5) não aceitam temperature.
    if isinstance(modelo, str) and modelo.startswith(("o1", "o3", "o4", "gpt-5")):
        return ModelSettings()
    return ModelSettings(temperature=0)


def _registrar_handoff(destino: str):
    def _on_handoff(ctx: RunContextWrapper[ContextoAtendimento]) -> None:
        ctx.context.registrar(f"Handoff para {destino}")
    return _on_handoff


def _etapa_igual(etapa: str):
    def _habilitado(ctx: RunContextWrapper[ContextoAtendimento], agent: Agent) -> bool:
        return ctx.context.etapa_atual() == etapa
    return _habilitado


def criar_agentes(modelo: str | Model = MODELO_LLM) -> dict[str, Agent[ContextoAtendimento]]:
    """Cria os quatro agentes e conecta os handoffs. Retorna um dicionário por nome."""
    config = _configuracao_modelo(modelo)

    analista = Agent[ContextoAtendimento](
        name="Agente Analista de Risco",
        handoff_description="Consulta o modelo de ML e calcula a probabilidade de pico de risco.",
        instructions=INSTRUCOES_ANALISTA,
        tools=[prever_risco_cardiaco],
        model=modelo,
        model_settings=config,
    )

    especialista = Agent[ContextoAtendimento](
        name="Agente Especialista em Protocolos",
        handoff_description="Cruza o risco previsto com a base de protocolos simulada.",
        instructions=INSTRUCOES_ESPECIALISTA,
        tools=[consultar_protocolos],
        model=modelo,
        model_settings=config,
    )

    planejador = Agent[ContextoAtendimento](
        name="Agente Planejador de Resposta",
        handoff_description="Consolida risco e protocolos e gera a recomendação final estruturada.",
        instructions=INSTRUCOES_PLANEJADOR,
        tools=[planejar_resposta_operacional],
        output_type=RecomendacaoFinal,
        output_guardrails=[guardrail_consistencia],
        model=modelo,
        model_settings=config,
    )

    orquestrador = Agent[ContextoAtendimento](
        name="Agente Orquestrador",
        instructions=INSTRUCOES_ORQUESTRADOR,
        tools=[consultar_estado_atendimento],
        input_guardrails=[guardrail_escopo],
        model=modelo,
        model_settings=config,
    )

    orquestrador.handoffs = [
        handoff(analista, on_handoff=_registrar_handoff(analista.name),
                is_enabled=_etapa_igual("PREDICAO_PENDENTE")),
        handoff(especialista, on_handoff=_registrar_handoff(especialista.name),
                is_enabled=_etapa_igual("PROTOCOLOS_PENDENTES")),
        handoff(planejador, on_handoff=_registrar_handoff(planejador.name),
                is_enabled=_etapa_igual("PLANO_PENDENTE")),
    ]
    retorno = handoff(orquestrador, on_handoff=_registrar_handoff(orquestrador.name))
    analista.handoffs = [retorno]
    especialista.handoffs = [retorno]

    return {
        "orquestrador": orquestrador,
        "analista": analista,
        "especialista": especialista,
        "planejador": planejador,
    }
