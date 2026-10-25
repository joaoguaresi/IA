# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre o projeto

Este é o repositório do Projeto Integrador de Inteligência Artificial e Sistemas Inteligentes de João F. Guaresi e Laura P. Ricci. O objetivo é construir, do zero, um Large Language Model (LLM), seguindo como referência o livro *"Build a Large Language Model (From Scratch)"*, de Sebastian Raschka (`Livro - Build a Large Language Model.pdf`).

Em vez de usar modelos prontos, cada conceito do livro é implementado manualmente: das leituras iniciais até a arquitetura Transformer completa, o treinamento do modelo, a geração de texto e o fine-tuning final. O projeto começou em 30/07/2026 e a entrega final é em 03/12/2026.

O projeto está na **Sprint 4** (Capítulo 4: Transformer Block, arquitetura GPT completa e geração de texto — ver `Sprints.txt`), com o código já implementado. As sprints anteriores (0-1: ambiente + Capítulo 1; 2: tokenização, vocabulário, embeddings e positional embeddings; 3: mecanismos de atenção) já estão implementadas em `src/`, com código, glossário e relatório entregues. O código é **incremental**: cada sprint soma componentes aos já implementados em `src/`, formando o pipeline completo do LLM ao final do semestre.

A Sprint 4 é feita em dupla, dividida verticalmente: João (`src/transformer.py` — Transformer Block e `GPTModel`) e Laura (`src/geracao.py` — inferência e geração autorregressiva). Os dois módulos são desacoplados por design: `geracao.py` aceita qualquer chamável `(batch, tokens) -> (batch, tokens, vocab_size)` como "modelo", então as funções de geração nunca dependeram de `GPTModel` existir para serem desenvolvidas e testadas (ver `experiments/_modelos_para_geracao.py`, que troca automaticamente o modelo provisório pelo `GPTModel` real assim que ele está disponível em `src`).

## Fluxo de trabalho (ciclo por capítulo)

Cada capítulo do livro segue o mesmo ciclo, descrito em `Sprints.txt`:

1. **Leitura orientada** do capítulo correspondente
2. **Glossário técnico** — para cada termo: original em inglês, tradução, definição, função no modelo, relação com outros conceitos e exemplo
3. **Quiz** individual
4. **Implementação** dos conceitos do capítulo (incremental — cada sprint se soma às anteriores)
5. **Experimentação** com diferentes configurações
6. **Análise dos resultados**, relacionando-os aos conceitos estudados (não apenas código/gráficos)

Consulte `Sprints.txt` para os requisitos detalhados de cada sprint antes de trabalhar em qualquer etapa — ele define exatamente o que deve ser lido, implementado e entregue. Os entregáveis não são só código: cada sprint exige glossário atualizado (`glossaries/`), relatório com a análise técnica (`documents/`) e quiz individual.

## Comandos comuns

O ambiente virtual já existe em `.venv/`. Ative-o antes de rodar qualquer comando (PowerShell: `.venv\Scripts\Activate.ps1`; bash: `source .venv/Scripts/activate`).

```
# instalar/atualizar dependências
pip install -r requirements.txt

# validar o ambiente (Python, PyTorch, CUDA, tiktoken, etc.)
python check_environment.py

# rodar todos os testes
python -m unittest discover -s tests

# rodar um único arquivo/teste
python -m unittest tests.test_attention
python -m unittest tests.test_attention.TestCausalAttention.test_futuro_tem_peso_zero

# rodar todos os experimentos de uma sprint em sequência
python experiments/run_all.py            # Sprint 2 (exp01-exp06)
python experiments/run_all_sprint3.py    # Sprint 3 (exp07-exp12)
python experiments/run_all_sprint4.py    # Sprint 4 (exp13-exp17)

# rodar um experimento isolado (cada exp*.py também tem um main() executável)
python experiments/exp07_self_attention.py
```

Os scripts em `experiments/` importam `src` via `sys.path` manipulado em `experiments/_comum.py` (que também expõe helpers de impressão de tabela e `salvar_figura`), então funcionam quando chamados diretamente com `python experiments/exp0N_*.py` a partir de qualquer diretório — não é necessário instalar o pacote. Figuras geradas vão para `experiments/figuras/`.

## Arquitetura

O código em `src/` implementa, de ponta a ponta, a entrada de um LLM estilo GPT — do texto bruto aos context vectors do self-attention — como uma sequência de módulos que se encaixam. `src/__init__.py` reexporta a API pública e documenta o fluxo completo:

```
Texto
  -> tokenizer.py / bpe.py          Token IDs
  -> dataset.py                     pares (entrada, alvo) via janela deslizante + DataLoader
  -> embeddings.py                  token embeddings + positional embeddings  = input embeddings
  -> attention.py                   self / causal / multi-head attention      = context vectors
  -> transformer.py                 N x TransformerBlock + LayerNorm final    = logits
  -> geracao.py                     logits -> probabilidades -> próximo token = texto gerado
```

### Sprint 2 — preparação de dados e embeddings

- **`src/tokenizer.py`** — tokenização por regex e vocabulário simples (`SimpleTokenizerV1`/`V2`, com tratamento de tokens desconhecidos via `<|unk|>`).
- **`src/bpe.py`** — tokenização por subpalavras (Byte Pair Encoding): `TiktokenTokenizer` (adaptador do tokenizador oficial do GPT-2 via `tiktoken`) e `BPETokenizerSimples` (implementação didática treinável, sem dependências). `obter_tokenizador_bpe()` devolve o adaptador `tiktoken`.
- **`src/dataset.py`** — transforma Token IDs em pares (entrada, alvo) via janela deslizante (`criar_pares_janela_deslizante`) e os organiza em lotes com `GPTDatasetV1`/`create_dataloader_v1` (Dataset/DataLoader do PyTorch).
- **`src/embeddings.py`** — `GPTEmbedding` soma token embeddings e positional embeddings absolutos, produzindo a entrada esperada pelo self-attention.

### Sprint 3 — mecanismos de atenção

- **`src/attention.py`** — as quatro variantes do Capítulo 3, na ordem em que o livro as constrói: (1) self-attention simplificado sem pesos treináveis (`self_attention_simples`, `attention_scores_simples`, `context_vector_passo_a_passo`); (2) scaled dot-product com `Wq/Wk/Wv` treináveis (`scaled_dot_product_attention`, `SelfAttention_v1`, `SelfAttention_v2`); (3) causal attention com máscara + dropout (`CausalAttention`); (4) multi-head (`MultiHeadAttentionWrapper` empilha cabeças; `MultiHeadAttention` faz o split em uma única projeção). Todas recebem `(batch, num_tokens, d_in)` e devolvem `(batch, num_tokens, d_out)`, e **guardam os pesos de atenção da última chamada em `ultimo_attn_weights`** para permitir visualização sem recalcular. `entropia_dos_pesos` e `numero_de_parametros` são utilitários de análise usados nos experimentos.
- **`src/visualizacao.py`** — renderização das matrizes de atenção com matplotlib (`mapa_de_atencao`, `salvar_mapa`, `grade_de_mapas`); usado pelos experimentos de visualização da Sprint 3.

### Sprint 4 — Transformer Block, modelo GPT e geração de texto

- **`src/transformer.py`** (Entregáveis 2 e 3) — `LayerNorm` e `GELU` implementados do zero (seção 4.2–4.3, não `nn.LayerNorm`/`nn.GELU` do PyTorch); `FeedForward` (expansão de 4x); `TransformerBlock`, que integra `MultiHeadAttention` + LayerNorm + FeedForward + duas conexões residuais (seção 4.5) e repassa `ultimo_attn_weights` como propriedade, no mesmo contrato da Sprint 3; `GPTModel`, que empilha `n_layers` blocos sobre a `GPTEmbedding` e termina em LayerNorm final + `out_head` (Linear sem bias) produzindo logits `(batch, tokens, vocab_size)`. Os defaults de `GPTModel` reproduzem o `GPT_CONFIG_124M` do livro (768/12/12). `memoria_em_mb` e `parametros_por_componente` são utilitários de análise (este último reusa `numero_de_parametros` de `attention.py`).
- **`src/geracao.py`** (Entregável 4) — ciclo de geração autorregressiva (`gerar`, `gerar_texto`, `gerar_texto_greedy`, `gerar_passo_a_passo`), independente de arquitetura: aceita qualquer chamável `(batch, tokens) -> (batch, tokens, vocab_size)` como modelo. Implementa temperatura (`aplicar_temperatura`) e top-k (`aplicar_top_k`) sobre os logits, conversão texto↔Token IDs (`texto_para_ids`/`ids_para_texto`) e métricas de análise (`entropia_da_distribuicao`, `tokens_para_cobrir`, `taxa_de_repeticao`).
- **`experiments/_modelos_para_geracao.py`** — ponte entre os dois módulos: `obter_modelo_gpt()` tenta importar `GPTModel` de `src` e, se ainda não existir, cai para um `ModeloProvisorio` (embeddings + uma camada de atenção) ou um `ModeloBigrama` (contagem de bigramas do corpus, sem rede neural) — assim os experimentos de geração nunca ficam bloqueados pela implementação da arquitetura.

### Integração e dados

- **`src/pipeline.py`** — integra as etapas da Sprint 2 (`executar_pipeline`) e resume as dimensões produzidas em cada uma (`ResultadoPipeline`), usado pelos experimentos para inspecionar o pipeline de ponta a ponta.
- **`src/dados.py`** — carrega o corpus de treinamento, priorizando `data/the-verdict.txt` (se baixado) sobre `data/amostra.txt`.
- Todos os tokenizadores (`SimpleTokenizerV1/V2`, `TiktokenTokenizer`, `BPETokenizerSimples`) compartilham a mesma interface (`encode`, `decode`, `n_vocab`, `nome`), o que permite trocá-los livremente em `dataset.py` e `pipeline.py`.

`experiments/exp0N_*.py` espelham as seções do capítulo da sprint (um script por subtópico) e reutilizam utilidades de `experiments/_comum.py`. Sprint 2: `exp01`-`exp06` (tokenização, vocabulário/Token IDs, janela deslizante, DataLoader, embeddings, pipeline completo). Sprint 3: `exp07`-`exp12` (self-attention, scaled dot-product, causal, multi-head, visualizações das matrizes, integração com a saída da Sprint 2). Sprint 4: `exp13`-`exp15` (Transformer Block, modelo GPT completo e parâmetros, experimentos de arquitetura — n_layers/num_heads/emb_dim/dropout/contexto) e `exp16`-`exp17` (geração autorregressiva, temperatura e top-k). `run_all_sprint4.py` detecta `exp13`-`exp15` dinamicamente (por `glob`), então roda mesmo que só parte dos experimentos exista.

## Estrutura do repositório

- `src/` — biblioteca do pipeline do LLM (ver Arquitetura acima), reexportada via `src/__init__.py`.
- `experiments/` — scripts de experimentação, um por subtópico, com um `run_all*.py` por sprint. Figuras em `experiments/figuras/`.
- `tests/` — testes `unittest` (compatíveis com `pytest`) para os módulos de `src/`.
- `data/` — corpus de treinamento (`amostra.txt`, atualmente o único presente; `src/dados.py` usa `data/the-verdict.txt` em seu lugar automaticamente se ele for baixado).
- `glossaries/` — glossários técnicos cumulativos, um por integrante (`Glossary - João.md`, `Glossary - Laura`). Cada nova sprint acrescenta uma seção `# Glossário — Capítulo N`. Cada termo segue o formato: termo original, tradução, definição, função no modelo, relação com outros conceitos, exemplo.
- `documents/` — relatórios por capítulo (ex.: `Capítulo 2 - Relatório da Sprint 2.docx`, `Capítulo 3 - Relatório da Sprint 3.docx`), com a análise dos resultados exigida por cada Sprint.
- `Livro - Build a Large Language Model.pdf` — referência principal do projeto.
- `Projeto Final.pdf` — enunciado/critérios de avaliação do projeto integrador.
- `Sprints.txt` — especificação de cada sprint (objetivos, atividades e entregáveis).
- `README.md` — descrição do projeto e critérios de avaliação.

## Convenções

- Arquivos de glossário por pessoa seguem o padrão `Glossary - <Nome>` (sem extensão consistente — alguns são `.md`, outros texto puro). São cumulativos: cada sprint adiciona a seção do novo capítulo sem remover as anteriores.
- Todo o conteúdo (glossários, relatórios, commits, docstrings, nomes de funções/variáveis) é escrito em português do Brasil — incluindo o código em `src/` e `experiments/`, que usa nomes em português (`tokenizar`, `construir_vocabulario`, `criar_pares_janela_deslizante`, `entropia_dos_pesos`) mesmo ao envolver termos técnicos do livro em inglês (Token IDs, embeddings, self-attention). Nomes consagrados de classes/APIs do livro são mantidos como no original (`SelfAttention_v2`, `MultiHeadAttention`, `create_dataloader_v1`).
- Cada módulo/função em `src/` referencia no docstring a seção do capítulo do livro que implementa — mantenha esse vínculo ao adicionar código novo.
- Código novo é organizado de forma incremental (cada sprint constrói sobre os componentes das anteriores) e sua API pública é adicionada aos imports e ao `__all__` de `src/__init__.py`, agrupada por sprint. O pipeline final integra tokenização → embeddings → atenção → Transformer → treinamento → geração → fine-tuning.
- Módulos de atenção expõem os pesos da última passada em `ultimo_attn_weights`; experimentos de visualização dependem disso — preserve esse contrato ao mexer em `src/attention.py` ou em `TransformerBlock`/`GPTModel` (`src/transformer.py`), que o repassam.
- `src/geracao.py` não depende de `GPTModel`: qualquer chamável `(batch, tokens) -> (batch, tokens, vocab_size)` serve como "modelo" nas funções de geração. Ao alterar a assinatura de `GPTModel.forward`, verifique `experiments/_modelos_para_geracao.py` (que importa `GPTModel` de `src` por reflexão) e `tests/test_geracao.py`.
