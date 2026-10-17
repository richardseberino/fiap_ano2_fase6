"""Execução do pipeline multiagente: atendimento completo, perguntas de acompanhamento,
modo offline e formatação da saída."""

from __future__ import annotations

from typing import Any

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    Model,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    RunHooks,
    Runner,
    Session,
    Tool,
)

from . import preditor
from .agentes import criar_agentes
from .config import MAX_TURNOS, MODELO_LLM
from .contexto import ContextoAtendimento, montar_plano, recomendacao_deterministica
from .esquemas import DadosPaciente, RecomendacaoFinal
from .protocolos import avaliar_protocolos

MAX_RETOMADAS = 2


class HooksTrilha(RunHooks[ContextoAtendimento]):
    """Imprime no terminal cada agente, handoff e tool acionados durante a execução."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, texto: str) -> None:
        if self.verbose:
            print(f"  · {texto}")

    async def on_agent_start(self, context, agent: Agent) -> None:
        self._log(f"[agente] {agent.name} assumiu o atendimento")

    async def on_handoff(self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent) -> None:
        self._log(f"[handoff] {from_agent.name} → {to_agent.name}")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self._log(f"[tool] {agent.name} chamou {tool.name}")


def _mensagem_inicial(ctx: ContextoAtendimento) -> str:
    return (
        f"Novo atendimento do paciente {ctx.paciente_id}. Os dados clínicos simulados já foram "
        f"validados e estão no contexto do sistema: {ctx.paciente.model_dump_json()}. "
        "Execute o fluxo completo e gere a recomendação final estruturada."
    )


async def executar_atendimento(
    dados: DadosPaciente,
    paciente_id: str,
    session: Session | None = None,
    modelo: str | Model = MODELO_LLM,
    verbose: bool = True,
) -> tuple[RecomendacaoFinal, ContextoAtendimento, dict[str, Any]]:
    """Executa o fluxo multiagente completo para um novo paciente.

    Retorna a recomendação final, o contexto (com a trilha) e metadados da execução.
    """
    agentes = criar_agentes(modelo)
    ctx = ContextoAtendimento(paciente_id=paciente_id, paciente=dados)
    hooks = HooksTrilha(verbose)
    meta: dict[str, Any] = {"retomadas": 0, "guardrail_bloqueou": False, "problemas": []}

    entrada = _mensagem_inicial(ctx)
    for tentativa in range(MAX_RETOMADAS + 1):
        try:
            resultado = await Runner.run(
                agentes["orquestrador"], entrada, context=ctx, session=session,
                hooks=hooks, max_turns=MAX_TURNOS,
            )
        except OutputGuardrailTripwireTriggered as erro:
            # O LLM devolveu algo divergente do modelo/protocolos: usamos a versão do contexto.
            problemas = erro.guardrail_result.output.output_info["problemas"]
            meta.update(guardrail_bloqueou=True, problemas=problemas)
            if ctx.plano is None:
                ctx.plano = montar_plano(ctx.protocolos)
            return recomendacao_deterministica(ctx), ctx, meta

        meta["ultimo_agente"] = resultado.last_agent.name
        if isinstance(resultado.final_output, RecomendacaoFinal):
            return resultado.final_output, ctx, meta

        # Algum agente respondeu em texto em vez de transferir. Retomamos pelo Orquestrador,
        # que consulta a etapa atual no contexto e continua de onde parou.
        meta["retomadas"] = tentativa + 1
        ctx.registrar(f"Retomada {tentativa + 1}: fluxo parou em {ctx.etapa_atual()}")
        entrada = "Continue o fluxo do atendimento a partir da etapa atual até a recomendação final."
        if session is None:
            entrada = resultado.to_input_list() + [{"role": "user", "content": entrada}]

    raise RuntimeError(f"O fluxo não foi concluído após {MAX_RETOMADAS} retomadas (etapa {ctx.etapa_atual()}).")


async def perguntar(
    pergunta: str,
    ctx: ContextoAtendimento,
    session: Session,
    modelo: str | Model = MODELO_LLM,
    verbose: bool = True,
) -> str:
    """Faz uma pergunta de acompanhamento ao Orquestrador, que usa o histórico da sessão."""
    agentes = criar_agentes(modelo)
    try:
        resultado = await Runner.run(
            agentes["orquestrador"], pergunta, context=ctx, session=session,
            hooks=HooksTrilha(verbose), max_turns=MAX_TURNOS,
        )
    except InputGuardrailTripwireTriggered:
        return ("Pedido bloqueado pelo guardrail de entrada: o sistema apoia a decisão com "
                "probabilidades e protocolos, mas não prescreve medicamentos nem doses.")
    return str(resultado.final_output)


def executar_offline(dados: DadosPaciente, paciente_id: str) -> tuple[RecomendacaoFinal, ContextoAtendimento]:
    """Executa as mesmas etapas sem LLM, na ordem que o Orquestrador seguiria.

    Útil para testes automatizados e para conferir a resposta esperada sem chave da OpenAI.
    """
    ctx = ContextoAtendimento(paciente_id=paciente_id, paciente=dados)
    ctx.risco = preditor.prever(dados)
    ctx.registrar(f"Modelo de ML: probabilidade {ctx.risco.probabilidade:.2f} ({ctx.risco.classificacao})")
    ctx.protocolos = avaliar_protocolos(dados, ctx.risco.classificacao)
    ctx.registrar(f"Base de protocolos: {len(ctx.protocolos.protocolos_acionados)} protocolo(s) acionado(s)")
    ctx.plano = montar_plano(ctx.protocolos)
    ctx.registrar(f"Plano operacional: {len(ctx.plano)} ação(ões)")
    return recomendacao_deterministica(ctx), ctx


def formatar_recomendacao(r: RecomendacaoFinal) -> str:
    linha = "=" * 72
    partes = [
        linha,
        f"RECOMENDAÇÃO FINAL · paciente {r.paciente_id}",
        linha,
        f"RISCO PREVISTO: {r.probabilidade_pico_risco:.2f}",
        f"CLASSIFICAÇÃO DE RISCO: {r.classificacao_risco}",
        "",
        "FATORES DE RISCO IDENTIFICADOS:",
        *[f"- {f}" for f in r.fatores_de_risco or ["Nenhum critério de risco acima do limite"]],
        "",
        "PROTOCOLOS SUGERIDOS:",
        *[f"- {p}" for p in r.protocolos_sugeridos or ["Nenhum protocolo acionado"]],
        "",
        "AÇÕES OPERACIONAIS:",
        *[f"[P{a.prioridade}] {a.acao}\n     Responsável: {a.responsavel} | Prazo: {a.prazo}"
          for a in sorted(r.acoes_operacionais, key=lambda a: a.prioridade)],
    ]
    if r.parametros_em_atencao:
        partes += ["", "PARÂMETROS EM ZONA DE ATENÇÃO:", *[f"- {p}" for p in r.parametros_em_atencao]]
    partes += ["", f"JUSTIFICATIVA: {r.justificativa}", "", f"AVISO: {r.aviso}", linha]
    return "\n".join(partes)
