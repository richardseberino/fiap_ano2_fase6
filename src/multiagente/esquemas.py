"""Esquemas Pydantic de entrada, de resultados intermediários e da saída final.

A validação acontece em dois pontos:
- na entrada, ``DadosPaciente`` rejeita valores fisiologicamente impossíveis;
- na saída, ``RecomendacaoFinal`` é o ``output_type`` do Agente Planejador, e o SDK
  obriga o LLM a responder exatamente nesse formato.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ClassificacaoRisco = Literal["BAIXO", "MODERADO", "ALTO"]


def _normalizar_chave(chave: str) -> str:
    """Converte 'disponibilidade recursos' ou 'FC-Media' em 'disponibilidade_recursos'/'fc_media'."""
    return re.sub(r"[\s\-]+", "_", chave.strip().lower())


class DadosPaciente(BaseModel):
    """Dados clínicos e operacionais simulados de um novo paciente."""

    model_config = ConfigDict(extra="forbid")

    idade_media: float = Field(ge=0, le=120, description="Idade em anos")
    carga_sistema: float = Field(ge=0, le=100, description="Ocupação do sistema (%)")
    tempo_espera: float = Field(ge=0, le=600, description="Tempo de espera (min)")
    fc_media: float = Field(ge=20, le=250, description="Frequência cardíaca média (bpm)")
    spo2_media: float = Field(ge=50, le=100, description="Saturação de oxigênio (%)")
    pa_sistolica: float = Field(ge=40, le=260, description="Pressão arterial sistólica (mmHg)")
    temperatura: float = Field(ge=0, le=45, description="Temperatura média (°C)")
    densidade_eventos: int = Field(ge=0, description="Densidade de eventos")
    eventos_24h: int = Field(ge=0, description="Eventos nas últimas 24 h")
    disponibilidade_recursos: float = Field(ge=0, le=100, description="Recursos disponíveis (%)")

    @model_validator(mode="before")
    @classmethod
    def normalizar_chaves(cls, dados):
        # O exemplo do enunciado traz "disponibilidade recursos" com espaço; aceitamos as duas formas.
        if isinstance(dados, dict):
            return {_normalizar_chave(str(k)): v for k, v in dados.items()}
        return dados


class ResultadoRisco(BaseModel):
    """Saída do Agente Analista de Risco, compartilhada no contexto."""

    probabilidade: float
    classificacao: ClassificacaoRisco
    fatores_de_risco: list[str]
    algoritmo: str


class ProtocoloAcionado(BaseModel):
    id: str
    protocolo: str
    valor_observado: str
    prioridade: int


class ResultadoProtocolos(BaseModel):
    """Saída do Agente Especialista em Protocolos, compartilhada no contexto."""

    protocolos_acionados: list[ProtocoloAcionado]
    parametros_em_atencao: list[str]


class AcaoOperacional(BaseModel):
    prioridade: int = Field(description="1 = mais urgente")
    acao: str
    responsavel: str
    prazo: str
    protocolo_origem: str = Field(description="Texto exato do protocolo que originou a ação")


class RecomendacaoFinal(BaseModel):
    """Resposta final estruturada produzida pelo Agente Planejador de Resposta."""

    paciente_id: str
    probabilidade_pico_risco: float = Field(description="Probabilidade prevista pelo modelo de ML (0 a 1)")
    classificacao_risco: ClassificacaoRisco
    fatores_de_risco: list[str]
    protocolos_sugeridos: list[str] = Field(description="Textos exatos dos protocolos acionados")
    acoes_operacionais: list[AcaoOperacional]
    parametros_em_atencao: list[str]
    justificativa: str = Field(description="Explicação curta da recomendação, em português")
    aviso: str = Field(description="Aviso de que se trata de apoio à decisão com dados simulados")
