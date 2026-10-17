# Roteiro do vídeo demonstrativo (até 4 minutos)

Publicar no YouTube como **não listado** e colar o link no README e no `entrega_portal.txt`.

| Tempo | O que mostrar na tela | Fala sugerida |
|---|---|---|
| 0:00–0:20 | README do repositório | Apresentação do grupo e do objetivo: prever picos de risco cardíaco e orquestrar agentes que transformam a previsão em recomendação. |
| 0:20–1:20 | Notebook no Colab, rolando pelas seções | Geração da base sintética, divisão 80/20 estratificada, comparação de três modelos, escolha do Random Forest, métricas e matriz de confusão (1 falso negativo em 1.000), salvamento do modelo. |
| 1:20–1:50 | `docs/arquitetura_multiagente.pdf`, diagrama | Orquestrador no centro, três especialistas, handoffs numerados, contexto compartilhado e guardrails. |
| 1:50–3:00 | Terminal: `python src/main.py` | Mostrar os dados validados, a trilha de agentes, handoffs e tools, e a recomendação final com risco 0.99, classificação ALTO e protocolos. Comentar os parâmetros em zona de atenção, como a SpO2 de 89%. |
| 3:00–3:35 | Terminal: `python src/main.py --interativo` | Perguntar "Por que o risco ficou alto?" para mostrar o histórico. Perguntar "Qual a dose de amiodarona?" para mostrar o guardrail de entrada bloqueando. |
| 3:35–4:00 | Terminal: `pytest` | Os testes cobrem o modelo, os protocolos, os handoffs e os guardrails. Encerramento com limitações e próximos passos. |

Dica: rode `python src/main.py --paciente data/pacientes/paciente_baixo_risco.json` antes de gravar
se quiserem mostrar também um caso de risco baixo.
