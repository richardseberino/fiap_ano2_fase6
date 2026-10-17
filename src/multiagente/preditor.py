"""Integração com o modelo de Machine Learning salvo na Parte 1."""

from __future__ import annotations

import warnings
from functools import lru_cache

import joblib
import pandas as pd
import sklearn

from .config import CAMINHO_MODELO, LIMIAR_ALTO, LIMIAR_MODERADO
from .esquemas import ClassificacaoRisco, DadosPaciente, ResultadoRisco

# Critérios clínico-operacionais que compõem o risk_score da base sintética.
# São usados apenas para explicar a previsão ao usuário, nunca para substituí-la.
CRITERIOS_RISCO = [
    ("carga_sistema", ">", 80, "Carga do sistema acima de 80%"),
    ("tempo_espera", ">", 60, "Tempo de espera acima de 60 min"),
    ("spo2_media", "<", 92, "SpO2 abaixo de 92%"),
    ("fc_media", ">", 110, "Frequência cardíaca acima de 110 bpm"),
    ("eventos_24h", ">", 4, "Mais de 4 eventos nas últimas 24 h"),
    ("disponibilidade_recursos", "<", 40, "Disponibilidade de recursos abaixo de 40%"),
]


@lru_cache(maxsize=1)
def carregar_modelo() -> dict:
    """Carrega o pacote do modelo. Se ele não existir, treina um novo automaticamente."""
    if not CAMINHO_MODELO.exists():
        from ml.treinar_modelo import treinar

        print("Modelo não encontrado. Treinando um novo modelo...")
        treinar(salvar_figuras=False, verbose=False)

    pacote = joblib.load(CAMINHO_MODELO)
    if pacote.get("versao_sklearn") != sklearn.__version__:
        warnings.warn(
            f"Modelo salvo com scikit-learn {pacote.get('versao_sklearn')} e carregado com "
            f"{sklearn.__version__}. Se houver erro, rode: python src/ml/treinar_modelo.py",
            stacklevel=2,
        )
    return pacote


def classificar(probabilidade: float) -> ClassificacaoRisco:
    if probabilidade >= LIMIAR_ALTO:
        return "ALTO"
    if probabilidade >= LIMIAR_MODERADO:
        return "MODERADO"
    return "BAIXO"


def fatores_de_risco(dados: DadosPaciente) -> list[str]:
    fatores = []
    for campo, op, limite, descricao in CRITERIOS_RISCO:
        valor = getattr(dados, campo)
        if (op == ">" and valor > limite) or (op == "<" and valor < limite):
            fatores.append(f"{descricao} (valor: {valor:g})")
    return fatores


def prever(dados: DadosPaciente) -> ResultadoRisco:
    """Calcula a probabilidade de pico de risco com o modelo treinado."""
    pacote = carregar_modelo()
    entrada = pd.DataFrame([dados.model_dump()])[pacote["features"]]
    probabilidade = round(float(pacote["modelo"].predict_proba(entrada)[0, 1]), 2)
    return ResultadoRisco(
        probabilidade=probabilidade,
        classificacao=classificar(probabilidade),
        fatores_de_risco=fatores_de_risco(dados),
        algoritmo=pacote["algoritmo"],
    )
