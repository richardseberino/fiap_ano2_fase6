"""Ponto de entrada do Sistema Preditivo Multiagente para Eventos Cardíacos.

Exemplos (a partir da raiz do repositório):
    python src/main.py                                    # paciente do enunciado
    python src/main.py --paciente data/pacientes/paciente_baixo_risco.json
    python src/main.py --interativo                       # perguntas de acompanhamento
    python src/main.py --offline                          # sem LLM, só para conferência
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import ValidationError  # noqa: E402

from multiagente.config import CAMINHO_HISTORICO, MODELO_LLM, PASTA_PACIENTES, RAIZ  # noqa: E402
from multiagente.esquemas import DadosPaciente  # noqa: E402
from multiagente.sistema import (  # noqa: E402
    executar_atendimento,
    executar_offline,
    formatar_recomendacao,
    perguntar,
)

PASTA_SAIDAS = RAIZ / "data" / "saidas"


def carregar_paciente(caminho: Path) -> tuple[str, DadosPaciente]:
    bruto = json.loads(caminho.read_text(encoding="utf-8"))
    paciente_id = bruto.pop("paciente_id", caminho.stem)
    try:
        return paciente_id, DadosPaciente.model_validate(bruto)
    except ValidationError as erro:
        sys.exit(f"Dados do paciente inválidos em {caminho}:\n{erro}")


def salvar_saida(paciente_id: str, recomendacao, trilha: list[str]) -> Path:
    PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_SAIDAS / f"{paciente_id}.json"
    destino.write_text(
        json.dumps({"recomendacao": recomendacao.model_dump(), "trilha": trilha}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destino


async def principal(args: argparse.Namespace) -> None:
    paciente_id, dados = carregar_paciente(Path(args.paciente))
    print(f"\nPaciente {paciente_id}: dados validados")
    print(json.dumps(dados.model_dump(), indent=2, ensure_ascii=False))

    if args.offline:
        print("\nModo OFFLINE: etapas executadas sem LLM, na mesma ordem do orquestrador.\n")
        recomendacao, ctx = executar_offline(dados, paciente_id)
        print(formatar_recomendacao(recomendacao))
        print(f"\nSaída salva em {salvar_saida(paciente_id, recomendacao, ctx.trilha).relative_to(RAIZ)}")
        return

    chave = os.getenv("OPENAI_API_KEY", "").strip()
    if not chave or chave == "sk-...":
        sys.exit(
            "Chave da OpenAI não configurada. Edite o arquivo .env na raiz do projeto e troque "
            "'sk-...' pela sua chave real (https://platform.openai.com/api-keys), ou use --offline."
        )

    from agents import SQLiteSession

    sessao_id = f"{paciente_id}-{datetime.now():%Y%m%d-%H%M%S}"
    session = SQLiteSession(sessao_id, CAMINHO_HISTORICO)

    print(f"\nIniciando pipeline multiagente (modelo {MODELO_LLM}, sessão {sessao_id})")
    recomendacao, ctx, meta = await executar_atendimento(dados, paciente_id, session=session)

    print("\nTrilha do atendimento (contexto compartilhado):")
    for evento in ctx.trilha:
        print(f"  - {evento}")
    if meta["guardrail_bloqueou"]:
        print("\nATENÇÃO: o guardrail de saída bloqueou a resposta do LLM por divergência:")
        for problema in meta["problemas"]:
            print(f"  - {problema}")
        print("A recomendação abaixo foi montada diretamente do contexto validado.")

    print()
    print(formatar_recomendacao(recomendacao))
    historico = await session.get_items()
    print(f"\nHistórico da sessão: {len(historico)} itens salvos em {CAMINHO_HISTORICO.relative_to(RAIZ)}")
    print(f"Saída salva em {salvar_saida(paciente_id, recomendacao, ctx.trilha).relative_to(RAIZ)}")

    if args.interativo:
        print("\nModo interativo: faça perguntas sobre este atendimento (Enter vazio para sair).")
        while True:
            pergunta = input("\nVocê: ").strip()
            if not pergunta:
                break
            resposta = await perguntar(pergunta, ctx, session)
            print(f"\nOrquestrador: {resposta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sistema Preditivo Multiagente para Eventos Cardíacos")
    parser.add_argument("--paciente", default=str(PASTA_PACIENTES / "novo_caso_enunciado.json"),
                        help="Arquivo JSON com os dados do paciente")
    parser.add_argument("--offline", action="store_true",
                        help="Executa as etapas sem LLM, apenas para conferência")
    parser.add_argument("--interativo", action="store_true",
                        help="Após a recomendação, permite perguntas usando o histórico da sessão")
    asyncio.run(principal(parser.parse_args()))


if __name__ == "__main__":
    main()
