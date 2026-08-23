# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre o projeto

Este é o repositório do Projeto Integrador de Inteligência Artificial e Sistemas Inteligentes de João F. Guaresi e Laura P. Ricci. O objetivo é construir, do zero, um Large Language Model (LLM), seguindo como referência o livro *"Build a Large Language Model (From Scratch)"*, de Sebastian Raschka (`Livro - Build a Large Language Model.pdf`).

Em vez de usar modelos prontos, cada conceito do livro é implementado manualmente: das leituras iniciais até a arquitetura Transformer completa, o treinamento do modelo, a geração de texto e o fine-tuning final. O projeto começou em 30/07/2026 e a entrega final é em 03/12/2026.

O projeto está na Sprint 2 (tokenização, vocabulário, embeddings e positional embeddings — ver `Sprints.txt`). O código é **incremental**: cada sprint soma componentes aos já implementados em `src/`, formando o pipeline completo do LLM ao final do semestre.

## Fluxo de trabalho (ciclo por capítulo)

Cada capítulo do livro segue o mesmo ciclo, descrito em `Sprints.txt`:

1. **Leitura orientada** do capítulo correspondente
2. **Glossário técnico** — para cada termo: original em inglês, tradução, definição, função no modelo, relação com outros conceitos e exemplo
3. **Quiz** individual
4. **Implementação** dos conceitos do capítulo (incremental — cada sprint se soma às anteriores)
5. **Experimentação** com diferentes configurações
6. **Análise dos resultados**, relacionando-os aos conceitos estudados (não apenas código/gráficos)

Consulte `Sprints.txt` para os requisitos detalhados de cada sprint antes de trabalhar em qualquer etapa — ele define exatamente o que deve ser lido, implementado e entregue.

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
python -m unittest tests.test_tokenizer
python -m unittest tests.test_tokenizer.TestTokenizacao.test_separa_pontuacao

# rodar todos os experimentos da sprint atual em sequência
python experiments/run_all.py

# rodar um experimento isolado (cada exp*.py também tem um main() executável)
python experiments/exp01_tokenizacao.py
```

Os scripts em `experiments/` importam `src` via `sys.path` manipulado em `experiments/_comum.py`, então funcionam quando chamados diretamente com `python experiments/exp0N_*.py` a partir de qualquer diretório — não é necessário instalar o pacote.

## Arquitetura

O código em `src/` implementa o pipeline de preparação de dados de um LLM estilo GPT, como uma sequência de módulos que se encaixam (ver docstring de `src/__init__.py` para o fluxo completo):

```
Texto -> tokenizer.py/bpe.py -> Token IDs -> dataset.py (janela deslizante + DataLoader)
      -> embeddings.py (token + positional embeddings) -> entrada do modelo (Sprint 3)
```

- **`src/tokenizer.py`** — tokenização por regex e vocabulário simples (`SimpleTokenizerV1`/`V2`, com tratamento de tokens desconhecidos via `<|unk|>`).
- **`src/bpe.py`** — tokenização por subpalavras (Byte Pair Encoding): `TiktokenTokenizer` (adaptador do tokenizador oficial do GPT-2 via `tiktoken`) e `BPETokenizerSimples` (implementação didática treinável, sem dependências).
- **`src/dataset.py`** — transforma Token IDs em pares (entrada, alvo) via janela deslizante (`criar_pares_janela_deslizante`) e os organiza em lotes com `GPTDatasetV1`/`create_dataloader_v1` (Dataset/DataLoader do PyTorch).
- **`src/embeddings.py`** — `GPTEmbedding` soma token embeddings e positional embeddings absolutos, produzindo a entrada esperada pelo self-attention da Sprint 3.
- **`src/pipeline.py`** — integra todas as etapas acima (`executar_pipeline`) e resume as dimensões produzidas em cada uma (`ResultadoPipeline`), usado pelos experimentos para inspecionar o pipeline de ponta a ponta.
- **`src/dados.py`** — carrega o corpus de treinamento, priorizando `data/the-verdict.txt` (se baixado) sobre `data/amostra.txt`.
- Todos os tokenizadores (`SimpleTokenizerV1/V2`, `TiktokenTokenizer`, `BPETokenizerSimples`) compartilham a mesma interface (`encode`, `decode`, `n_vocab`, `nome`), o que permite trocá-los livremente em `dataset.py` e `pipeline.py`.

`experiments/exp0N_*.py` espelham as seções do capítulo (um script por subtópico: tokenização, vocabulário/Token IDs, janela deslizante, DataLoader, embeddings, pipeline completo) e reutilizam utilidades de impressão/plot de `experiments/_comum.py`. Figuras geradas vão para `experiments/figuras/`.

## Estrutura do repositório

- `src/` — biblioteca do pipeline de dados do LLM (ver Arquitetura acima), reexportada via `src/__init__.py`.
- `experiments/` — scripts de experimentação da sprint atual, um por subtópico, mais `run_all.py` para rodar todos.
- `tests/` — testes `unittest` (compatíveis com `pytest`) para os módulos de `src/`.
- `data/` — corpus de treinamento (`amostra.txt`, atualmente o único presente; `src/dados.py` usa `data/the-verdict.txt` em seu lugar automaticamente se ele for baixado).
- `glossaries/` — glossários técnicos por capítulo, um por integrante (`Glossary - João.md`, `Glossary - Laura`). Cada termo segue o formato: tradução, definição, funcionalidade, relação com outros conceitos, exemplo conceitual.
- `documents/` — relatórios por capítulo (ex.: `Capítulo 1 - Relatório de leitura.docx`, `Capítulo 2 - Relatório da Sprint 2.docx` com a análise dos resultados exigida pela Sprint).
- `Livro - Build a Large Language Model.pdf` — referência principal do projeto.
- `Sprints.txt` — especificação de cada sprint (objetivos, atividades e entregáveis).
- `README.md` — descrição do projeto e critérios de avaliação.

## Convenções

- Arquivos de glossário por pessoa seguem o padrão `Glossary - <Nome>` (sem extensão consistente — alguns são `.md`, outros texto puro).
- Todo o conteúdo (glossários, relatórios, commits, docstrings, nomes de funções/variáveis) é escrito em português do Brasil — incluindo o código em `src/` e `experiments/`, que usa nomes em português (`tokenizar`, `construir_vocabulario`, `criar_pares_janela_deslizante`) mesmo ao envolver termos técnicos do livro em inglês (Token IDs, embeddings).
- Cada módulo/função em `src/` referencia no docstring a seção do capítulo do livro que implementa — mantenha esse vínculo ao adicionar código novo.
- Quando código for adicionado nas próximas sprints, ele deve ser organizado de forma incremental (cada sprint constrói sobre os componentes das anteriores), já que o pipeline final integra tokenização → embeddings → atenção → Transformer → treinamento → geração → fine-tuning.
