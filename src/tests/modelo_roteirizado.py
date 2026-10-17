"""Modelo de linguagem simulado para testar o pipeline multiagente sem chamar a OpenAI.

Ele segue as mesmas instruções dadas aos agentes reais: chama a tool do agente uma vez,
depois faz o handoff disponível, e o Planejador devolve o JSON da RecomendacaoFinal.
Assim, o teste exercita os handoffs, tools, guardrails e a sessão reais do SDK.
"""

from __future__ import annotations

import json
from itertools import count

from agents import Handoff, Model, ModelResponse, Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

_ids = count(1)


def _item(obj, campo):
    return obj.get(campo) if isinstance(obj, dict) else getattr(obj, campo, None)


def _chamada(nome: str) -> ResponseFunctionToolCall:
    n = next(_ids)
    return ResponseFunctionToolCall(
        type="function_call", id=f"fc_{n}", call_id=f"call_{n}", name=nome, arguments="{}",
    )


def _mensagem(texto: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        type="message", id=f"msg_{next(_ids)}", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=texto, annotations=[])],
    )


class ModeloRoteirizado(Model):
    """``adulterar`` permite simular um LLM que altera a probabilidade na resposta final."""

    def __init__(self, adulterar: bool = False):
        self.adulterar = adulterar
        self.chamadas = 0

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs: list[Handoff], tracing, *, previous_response_id=None,
                           conversation_id=None, prompt=None) -> ModelResponse:
        self.chamadas += 1
        itens = [] if isinstance(input, str) else list(input)
        nomes_handoff = {h.tool_name for h in handoffs}

        # Itens produzidos desde o último handoff, ou seja, no turno do agente atual.
        inicio = 0
        for i, item in enumerate(itens):
            if _item(item, "type") == "function_call" and str(_item(item, "name")).startswith("transfer_to"):
                inicio = i + 1
        recentes = itens[inicio:]
        ja_chamadas = {_item(i, "name") for i in recentes if _item(i, "type") == "function_call"}

        tool_propria = next((t.name for t in tools if t.name not in nomes_handoff), None)
        if tool_propria and tool_propria not in ja_chamadas:
            saida = [_chamada(tool_propria)]
        elif handoffs:
            saida = [_chamada(handoffs[0].tool_name)]
        elif output_schema is not None:
            resultado = next(
                json.loads(_item(i, "output")) for i in reversed(recentes)
                if _item(i, "type") == "function_call_output"
            )
            resultado["justificativa"] = "Resposta gerada pelo modelo roteirizado de teste."
            if self.adulterar:
                resultado["probabilidade_pico_risco"] = 0.42
            saida = [_mensagem(json.dumps(resultado, ensure_ascii=False))]
        else:
            saida = [_mensagem("Atendimento concluído. Consulte a recomendação final.")]

        return ModelResponse(output=saida, usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError
