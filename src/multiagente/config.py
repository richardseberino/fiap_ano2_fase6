"""Configurações centrais do sistema multiagente."""

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ / ".env")

CAMINHO_MODELO = RAIZ / "src" / "models" / "modelo_pico_risco.joblib"
CAMINHO_HISTORICO = RAIZ / "data" / "historico_atendimentos.db"
PASTA_PACIENTES = RAIZ / "data" / "pacientes"

# Modelo de linguagem usado pelos agentes. Pode ser trocado pela variável OPENAI_MODEL.
MODELO_LLM = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Faixas de classificação aplicadas sobre a probabilidade prevista pelo modelo de ML.
LIMIAR_MODERADO = 0.30
LIMIAR_ALTO = 0.70

# Tolerância usada pelo guardrail de saída ao comparar a probabilidade relatada pelo LLM.
TOLERANCIA_PROBABILIDADE = 0.005

# Limite de turnos do Runner (orquestrador + 3 especialistas + retornos de handoff).
MAX_TURNOS = 25
