"""Contexto compartilhado entre os agentes durante um atendimento.

O SDK repassa o mesmo objeto de contexto para todas as tools, guardrails e hooks
de uma execução. Assim, o resultado da predição gravado pelo Agente Analista de
Risco fica disponível para os demais agentes sem que o LLM precise copiar números.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .esquemas import (
    AcaoOperacional,
    DadosPaciente,
    RecomendacaoFinal,
    ResultadoProtocolos,
    ResultadoRisco,
)

ORIGEM_ROTINA = "Sem protocolo acionado (acompanhamento de rotina)"
AVISO_PADRAO = (
    "Apoio à decisão gerado com dados sintéticos para fins acadêmicos. "
    "Não substitui avaliação médica."
)


@dataclass
class ContextoAtendimento:
    paciente_id: str
    paciente: DadosPaciente
    risco: ResultadoRisco | None = None
    protocolos: ResultadoProtocolos | None = None
    plano: list[AcaoOperacional] | None = None
    trilha: list[str] = field(default_factory=list)

    def etapa_atual(self) -> str:
        if self.risco is None:
            return "PREDICAO_PENDENTE"
        if self.protocolos is None:
            return "PROTOCOLOS_PENDENTES"
        if self.plano is None:
            return "PLANO_PENDENTE"
        return "CONCLUIDO"

    def registrar(self, evento: str) -> None:
        self.trilha.append(evento)


def montar_plano(protocolos: ResultadoProtocolos) -> list[AcaoOperacional]:
    """Converte os protocolos acionados em ações operacionais ordenadas por prioridade."""
    from .protocolos import REGRAS_POR_TEXTO

    acoes = []
    for p in protocolos.protocolos_acionados:
        regra = REGRAS_POR_TEXTO[p.protocolo]
        acoes.append(AcaoOperacional(
            prioridade=regra.prioridade, acao=regra.acao, responsavel=regra.responsavel,
            prazo=regra.prazo, protocolo_origem=regra.texto,
        ))
    if not acoes:
        acoes.append(AcaoOperacional(
            prioridade=3,
            acao="Manter acompanhamento de rotina e reavaliar sinais vitais a cada 4 h",
            responsavel="Equipe de enfermagem", prazo="Rotina", protocolo_origem=ORIGEM_ROTINA,
        ))
    return acoes


def recomendacao_deterministica(ctx: ContextoAtendimento, justificativa: str | None = None) -> RecomendacaoFinal:
    """Monta a recomendação final apenas com os dados do contexto, sem LLM.

    Usada no modo offline e como resposta de contingência quando o guardrail de
    saída bloqueia uma resposta inconsistente do LLM.
    """
    assert ctx.risco and ctx.protocolos and ctx.plano is not None
    if justificativa is None:
        n = len(ctx.protocolos.protocolos_acionados)
        justificativa = (
            f"O modelo {ctx.risco.algoritmo} estimou probabilidade de {ctx.risco.probabilidade:.2f} "
            f"de pico de risco, classificada como {ctx.risco.classificacao}. "
            f"Foram identificados {len(ctx.risco.fatores_de_risco)} fatores de risco e "
            f"{n} protocolo(s) acionado(s) pelos limites da base de protocolos."
        )
    return RecomendacaoFinal(
        paciente_id=ctx.paciente_id,
        probabilidade_pico_risco=ctx.risco.probabilidade,
        classificacao_risco=ctx.risco.classificacao,
        fatores_de_risco=ctx.risco.fatores_de_risco,
        protocolos_sugeridos=[p.protocolo for p in ctx.protocolos.protocolos_acionados],
        acoes_operacionais=ctx.plano,
        parametros_em_atencao=ctx.protocolos.parametros_em_atencao,
        justificativa=justificativa,
        aviso=AVISO_PADRAO,
    )
