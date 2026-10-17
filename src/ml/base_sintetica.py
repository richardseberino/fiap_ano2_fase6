"""Geração da base sintética de pico de risco cardíaco e da base de protocolos.

O código de geração reproduz exatamente a base fornecida no enunciado da Fase 6,
para garantir que o modelo treinado aqui seja o mesmo do notebook do Colab.
"""

import numpy as np
import pandas as pd

# Ordem oficial das variáveis de entrada do modelo.
FEATURES = [
    "idade_media",
    "carga_sistema",
    "tempo_espera",
    "fc_media",
    "spo2_media",
    "pa_sistolica",
    "temperatura",
    "densidade_eventos",
    "eventos_24h",
    "disponibilidade_recursos",
]

TARGET = "pico_risco"

# Base de conhecimento de protocolos, exatamente como fornecida no enunciado.
PROTOCOLOS = [
    "Se SPO2 < 88% → iniciar suporte respiratório imediato",
    "Se carga do sistema > 85% → abrir unidades temporárias",
    "Se FC > 120 → iniciar monitoramento intensivo",
    "Se eventos_24h > 5 → ativar equipe de resposta rápida",
    "Se disponibilidade_recursos < 30% → redistribuir insumos",
    "Se tempo_espera > 90 min → priorizar triagem emergencial",
    "Se temperatura > 38°C média → iniciar protocolo infeccioso",
    "Se densidade_eventos > 6 → ativar plano de contingência",
    "Se PA sistólica < 90 → iniciar protocolo de choque",
    "Se risco sistêmico alto → ativar estado de emergência",
]


def gerar_base(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Gera a base sintética de treinamento com a variável alvo ``pico_risco``."""
    np.random.seed(seed)

    df = pd.DataFrame({
        "idade_media": np.random.normal(55, 12, n).clip(18, 90),
        "carga_sistema": np.random.uniform(30, 100, n),
        "tempo_espera": np.random.normal(45, 20, n).clip(5, 180),
        "fc_media": np.random.normal(85, 15, n).clip(50, 150),
        "spo2_media": np.random.normal(95, 3, n).clip(70, 100),
        "pa_sistolica": np.random.normal(125, 20, n).clip(80, 200),
        "temperatura": np.random.normal(24, 5, n).clip(10, 40),
        "densidade_eventos": np.random.poisson(3, n),
        "eventos_24h": np.random.poisson(2, n),
        "disponibilidade_recursos": np.random.uniform(20, 100, n),
    })

    risk_score = (
        (df["carga_sistema"] > 80).astype(int) +
        (df["tempo_espera"] > 60).astype(int) +
        (df["spo2_media"] < 92).astype(int) +
        (df["fc_media"] > 110).astype(int) +
        (df["eventos_24h"] > 4).astype(int) +
        (df["disponibilidade_recursos"] < 40).astype(int)
    )

    df[TARGET] = (risk_score >= 2).astype(int)
    return df
