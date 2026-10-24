# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/">
  <img src="assets/logo-fiap.png"
       alt="FIAP - Faculdade de Informática e Administração Paulista"
       width="40%">
</a>
</p>

<br>

# Sistema Preditivo Multiagente para Eventos Cardíacos

## Nome do grupo DRELL

## 👨‍🎓 Integrantes:
- <a href="https://www.linkedin.com/in/richard-marques-26b3a14/">Richard</a>
- <a href="https://www.linkedin.com/in/luis-fernando-dos-santos-costa-b69894365/">Luis</a>

## 👩‍🏫 Professores:
### Tutor(a) 
- <a href="https://github.com/leoruiz197">Leo Ruiz</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/in/andregodoichiovato/">Andre Godoi</a>


## 🎬 Vídeo demonstrativo

**Link (YouTube, não listado):** https://youtu.be/1GLxRKDapgg


## 📜 Descrição

Nesta fase, o sistema deixa de apenas reagir e passa a analisar dados clínicos simulados para apoiar a decisão.
O projeto tem duas partes.

**Parte 1: modelo preditivo de pico de risco.** A partir da base sintética fornecida no enunciado, com 5.000
registros e 10 variáveis, treinamos um classificador supervisionado para prever a variável `pico_risco`.
Comparamos Regressão Logística, Árvore de Decisão e Random Forest com validação cruzada estratificada.
O **Random Forest** foi o escolhido, com os resultados abaixo no conjunto de teste:

| Métrica | Valor |
|---|---|
| Acurácia | 99,9% |
| Precisão | 100% |
| Recall | 99,6% |
| ROC AUC | 1,00 |

O modelo foi salvo em `src/models/modelo_pico_risco.joblib` para uso na Parte 2.

**Parte 2: sistema multiagente com o OpenAI Agents SDK.** Quatro agentes colaboram para transformar a previsão
em uma recomendação estruturada:

- **Agente Orquestrador.** É o ponto de entrada. Consulta a etapa do atendimento e coordena o fluxo com handoffs.
- **Agente Analista de Risco.** Chama o modelo de ML por meio de uma tool e gera a probabilidade, a classificação e os fatores de risco.
- **Agente Especialista em Protocolos.** Cruza o risco previsto e os dados do paciente com a base de protocolos simulada.
- **Agente Planejador de Resposta.** Consolida tudo em ações operacionais e entrega a `RecomendacaoFinal` estruturada.

O resultado da predição fica em um **contexto compartilhado**, lido por todos os agentes. Assim, nenhum agente
precisa copiar números clínicos. O sistema usa **handoffs** com habilitação por etapa e **tools** com
`@function_tool`. O **histórico de mensagens** fica em uma `SQLiteSession`. A **validação de saída** combina o
`output_type` em Pydantic com um guardrail que confere a resposta do LLM contra o contexto. Um guardrail de
entrada bloqueia pedidos de prescrição de medicamentos, que estão fora do escopo de um sistema de apoio à decisão.

A resposta final traz a probabilidade calculada, a classificação de risco, os protocolos sugeridos, as ações com
responsável e prazo, e os parâmetros que estão perto de acionar um protocolo.

> Todos os dados são sintéticos. O sistema é um exercício acadêmico e não substitui avaliação médica.


## 📁 Estrutura de pastas

```
.
├── assets/                     # logo da FIAP usado no README
├── data/
│   ├── base_sintetica_pico_risco.csv   # base gerada (seed 42)
│   └── pacientes/              # pacientes de exemplo em JSON (entrada do sistema)
├── docs/
│   ├── relatorio_tecnico.pdf   # relatório técnico do modelo (2 páginas)
│   ├── arquitetura_multiagente.pdf     # diagrama, papéis, handoffs, tools e exemplo (3 páginas)
│   ├── roteiro_video.md        # roteiro sugerido para o vídeo de 4 minutos
│   ├── img/                    # matriz de confusão e importância das variáveis
│   └── fontes/                 # HTML/CSS dos PDFs e script gerar_pdfs.sh
├── src/
│   ├── notebooks/
│   │   └── modelo_preditivo_pico_risco.ipynb  # Parte 1 (Google Colab)
│   ├── ml/
│   │   ├── base_sintetica.py   # geração da base e textos dos protocolos
│   │   └── treinar_modelo.py   # treino, avaliação e salvamento (mesmo fluxo do notebook)
│   ├── models/
│   │   ├── modelo_pico_risco.joblib   # modelo treinado
│   │   └── metricas_modelo.json       # métricas e comparação de modelos
│   ├── multiagente/
│   │   ├── agentes.py          # os 4 agentes e a topologia de handoffs
│   │   ├── ferramentas.py      # tools chamadas pelos agentes
│   │   ├── guardrails.py       # guardrails de entrada e de saída
│   │   ├── contexto.py         # contexto compartilhado entre agentes
│   │   ├── esquemas.py         # modelos Pydantic de entrada e saída
│   │   ├── preditor.py         # integração com o modelo de ML
│   │   ├── protocolos.py       # base de protocolos estruturada
│   │   ├── sistema.py          # execução do pipeline e formatação da saída
│   │   └── config.py
│   ├── tests/                  # testes automatizados (pytest)
│   └── main.py                 # ponto de entrada do sistema
├── entrega_portal.txt          # modelo do .txt para o Portal do Aluno
├── requirements.txt
└── .env.example
```


## 🔧 Como executar o código

### Pré-requisitos

- Python 3.11 ou superior
- Uma chave da API da OpenAI para rodar os agentes
- Google Colab, ou Jupyter local, para o notebook

### Instalação

```bash
git clone https://github.com/richardseberino/fiap_ano2_fase6.git
cd fiap_ano2_fase6
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # depois edite o .env com a sua OPENAI_API_KEY
```

### Parte 1: notebook do modelo preditivo

1. No Google Colab, abra `src/notebooks/modelo_preditivo_pico_risco.ipynb` pelo menu *Arquivo > Abrir notebook > GitHub*. Para repositório privado, marque a opção de incluir repositórios privados.
2. Execute todas as células. O notebook gera a base, compara os modelos, avalia o Random Forest, salva o modelo e simula o novo paciente.
3. Para treinar pelo terminal, com o mesmo resultado do notebook:

```bash
python src/ml/treinar_modelo.py
```

### Parte 2: sistema multiagente

```bash
# Paciente do enunciado
python src/main.py

# Outro paciente
python src/main.py --paciente data/pacientes/paciente_baixo_risco.json

# Depois da recomendação, faça perguntas que usam o histórico da sessão
python src/main.py --interativo
```

Pacientes de exemplo em `data/pacientes/`:

| Arquivo | Cenário |
|---|---|
| `novo_caso_enunciado.json` | caso do enunciado, risco alto |
| `paciente_risco_moderado.json` | caso de fronteira, risco moderado |
| `paciente_baixo_risco.json` | sinais estáveis, risco baixo |
| `paciente_critico_choque.json` | todos os protocolos acionados |

A opção `--offline` executa as mesmas etapas sem LLM. Ela serve só para conferir a resposta esperada sem
gastar créditos da API.

### Testes

```bash
pytest
```

Os testes cobrem o modelo salvo, a validação de entrada, a base de protocolos e o pipeline multiagente completo.
O pipeline é testado com um modelo de linguagem simulado, o que exercita os handoffs, as tools, os guardrails e a
sessão reais do SDK sem chamar a API.

### Exemplo de saída

```
RISCO PREVISTO: 0.99
CLASSIFICAÇÃO DE RISCO: ALTO

PROTOCOLOS SUGERIDOS:
- Se risco sistêmico alto → ativar estado de emergência
- Se carga do sistema > 85% → abrir unidades temporárias
- Se disponibilidade_recursos < 30% → redistribuir insumos
- Se densidade_eventos > 6 → ativar plano de contingência

PARÂMETROS EM ZONA DE ATENÇÃO:
- spo2_media = 89% (limite < 88%, protocolo P01 ainda não acionado)
- ...
```


## 📎 Links e Observações

- **Vídeo demonstrativo:** https://youtu.be/1GLxRKDapgg
- **Relatório técnico:** [docs/relatorio_tecnico.pdf](docs/relatorio_tecnico.pdf)
- **Arquitetura multiagente:** [docs/arquitetura_multiagente.pdf](docs/arquitetura_multiagente.pdf)

**Decisões técnicas**

- **Topologia hub-and-spoke.** O Orquestrador recebe o controle de volta após cada especialista. Cada handoff só fica habilitado na etapa certa, com `is_enabled`, o que impede pular ou repetir etapas.
- **Tools sem parâmetros clínicos.** As tools leem os dados do paciente do contexto compartilhado. O LLM decide quando chamá-las, mas não digita valores clínicos.
- **Guardrail de saída.** Se o LLM alterar a probabilidade ou inventar um protocolo, a resposta é bloqueada e o sistema usa a recomendação montada direto do contexto validado.
- **Protocolos aplicados à risca.** O exemplo do enunciado lista o protocolo de SpO2, mas o paciente tem SpO2 de 89%, acima do limite de 88%. O sistema não aciona esse protocolo e mostra a SpO2 como parâmetro em zona de atenção.

**Observações gerais**

- Todos os dados são sintéticos e o sistema não substitui avaliação médica.


## 🗃 Histórico de lançamentos

* 1.0.0 - XX/XX/2026
    * Entrega da Fase 6: modelo preditivo, sistema multiagente, relatório técnico e documento de arquitetura.


## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/SabrinaOtoni/TEMPLATE-FIAP-GRAD-ON-IA">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">FIAP</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
