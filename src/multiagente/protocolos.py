"""Base de protocolos simulada, estruturada para consulta pelos agentes.

Os textos dos protocolos são exatamente os do enunciado (importados de ``ml.base_sintetica``).
Cada texto recebe uma regra legível por máquina (campo, operador e limite), além da
ação operacional que o Agente Planejador usa para montar o plano de resposta.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

from ml.base_sintetica import PROTOCOLOS

from .esquemas import ClassificacaoRisco, DadosPaciente, ProtocoloAcionado, ResultadoProtocolos

_OPERADORES = {">": operator.gt, "<": operator.lt}


@dataclass(frozen=True)
class RegraProtocolo:
    id: str
    texto: str
    campo: str
    operador: str
    limite: float | str
    unidade: str
    margem_atencao: float  # distância do limite que coloca o parâmetro "em atenção"
    prioridade: int  # 1 = imediata, 2 = urgente, 3 = operacional
    acao: str
    responsavel: str
    prazo: str


BASE_PROTOCOLOS: list[RegraProtocolo] = [
    RegraProtocolo("P01", PROTOCOLOS[0], "spo2_media", "<", 88, "%", 2, 1,
                   "Iniciar oxigenoterapia ou ventilação e reavaliar SpO2 a cada 15 min",
                   "Equipe médica e fisioterapia respiratória", "Imediato"),
    RegraProtocolo("P02", PROTOCOLOS[1], "carga_sistema", ">", 85, "%", 5, 3,
                   "Abrir leitos e unidades temporárias para absorver a demanda",
                   "Gestão de leitos", "Até 2 h"),
    RegraProtocolo("P03", PROTOCOLOS[2], "fc_media", ">", 120, " bpm", 10, 1,
                   "Transferir para monitoramento cardíaco contínuo (ECG e sinais vitais)",
                   "Equipe de cardiologia e enfermagem", "Imediato"),
    RegraProtocolo("P04", PROTOCOLOS[3], "eventos_24h", ">", 5, " eventos", 1, 2,
                   "Acionar a equipe de resposta rápida para avaliação à beira-leito",
                   "Time de resposta rápida", "Até 15 min"),
    RegraProtocolo("P05", PROTOCOLOS[4], "disponibilidade_recursos", "<", 30, "%", 5, 3,
                   "Redistribuir insumos e equipamentos entre as unidades",
                   "Farmácia e logística hospitalar", "Até 1 h"),
    RegraProtocolo("P06", PROTOCOLOS[5], "tempo_espera", ">", 90, " min", 15, 2,
                   "Reclassificar o paciente na fila e priorizar a triagem emergencial",
                   "Enfermagem da triagem", "Até 10 min"),
    RegraProtocolo("P07", PROTOCOLOS[6], "temperatura", ">", 38, " °C", 0.5, 2,
                   "Isolar o paciente, colher culturas e iniciar o protocolo infeccioso",
                   "CCIH e equipe médica", "Até 1 h"),
    RegraProtocolo("P08", PROTOCOLOS[7], "densidade_eventos", ">", 6, " eventos", 1, 3,
                   "Ativar o plano de contingência e reforçar escalas",
                   "Coordenação de plantão", "Até 1 h"),
    RegraProtocolo("P09", PROTOCOLOS[8], "pa_sistolica", "<", 90, " mmHg", 10, 1,
                   "Iniciar reposição volêmica e avaliar vasopressores (protocolo de choque)",
                   "Equipe médica", "Imediato"),
    RegraProtocolo("P10", PROTOCOLOS[9], "classificacao_risco", "==", "ALTO", "", 0, 2,
                   "Declarar estado de emergência e comunicar a direção clínica",
                   "Direção clínica", "Até 30 min"),
]

REGRAS_POR_TEXTO = {r.texto: r for r in BASE_PROTOCOLOS}


def _formatar(valor: float, unidade: str) -> str:
    numero = f"{valor:g}" if isinstance(valor, (int, float)) else str(valor)
    return f"{numero}{unidade}"


def avaliar_protocolos(dados: DadosPaciente, classificacao: ClassificacaoRisco) -> ResultadoProtocolos:
    """Cruza os dados do paciente e o risco previsto com a base de protocolos.

    Retorna os protocolos acionados, ordenados por prioridade, e os parâmetros que
    estão próximos do limite mas não acionam protocolo ("zona de atenção").
    """
    acionados: list[ProtocoloAcionado] = []
    atencao: list[str] = []

    for regra in BASE_PROTOCOLOS:
        if regra.campo == "classificacao_risco":
            if classificacao == regra.limite:
                acionados.append(ProtocoloAcionado(
                    id=regra.id, protocolo=regra.texto,
                    valor_observado=f"classificação {classificacao}", prioridade=regra.prioridade,
                ))
            continue

        valor = getattr(dados, regra.campo)
        if _OPERADORES[regra.operador](valor, regra.limite):
            acionados.append(ProtocoloAcionado(
                id=regra.id, protocolo=regra.texto,
                valor_observado=f"{regra.campo} = {_formatar(valor, regra.unidade)}",
                prioridade=regra.prioridade,
            ))
        elif abs(valor - regra.limite) <= regra.margem_atencao:
            atencao.append(
                f"{regra.campo} = {_formatar(valor, regra.unidade)} "
                f"(limite {regra.operador} {_formatar(regra.limite, regra.unidade)}, "
                f"protocolo {regra.id} ainda não acionado)"
            )

    acionados.sort(key=lambda p: (p.prioridade, p.id))
    return ResultadoProtocolos(protocolos_acionados=acionados, parametros_em_atencao=atencao)


def listar_base() -> list[dict]:
    """Base completa em formato serializável, usada pela tool de consulta."""
    return [
        {"id": r.id, "protocolo": r.texto, "prioridade": r.prioridade}
        for r in BASE_PROTOCOLOS
    ]
