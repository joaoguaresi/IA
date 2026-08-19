# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre o projeto

Este é o repositório do Projeto Integrador de Inteligência Artificial e Sistemas Inteligentes de João F. Guaresi e Laura P. Ricci. O objetivo é construir, do zero, um Large Language Model (LLM), seguindo como referência o livro *"Build a Large Language Model (From Scratch)"*, de Sebastian Raschka (`Livro - Build a Large Language Model.pdf`).

Em vez de usar modelos prontos, cada conceito do livro é implementado manualmente: das leituras iniciais até a arquitetura Transformer completa, o treinamento do modelo, a geração de texto e o fine-tuning final. O projeto começou em 30/07/2026 e a entrega final é em 03/12/2026.

O repositório está em estágio inicial (Sprint 0-1): ainda **não há código-fonte**, apenas material de leitura, glossários e documentação. À medida que as sprints avançam, código de tokenização, embeddings, atenção, treinamento etc. será adicionado (ver `Sprints.txt`).

## Fluxo de trabalho (ciclo por capítulo)

Cada capítulo do livro segue o mesmo ciclo, descrito em `Sprints.txt`:

1. **Leitura orientada** do capítulo correspondente
2. **Glossário técnico** — para cada termo: original em inglês, tradução, definição, função no modelo, relação com outros conceitos e exemplo
3. **Quiz** individual
4. **Implementação** dos conceitos do capítulo (incremental — cada sprint se soma às anteriores)
5. **Experimentação** com diferentes configurações
6. **Análise dos resultados**, relacionando-os aos conceitos estudados (não apenas código/gráficos)

Consulte `Sprints.txt` para os requisitos detalhados de cada sprint antes de trabalhar em qualquer etapa — ele define exatamente o que deve ser lido, implementado e entregue.

## Estrutura do repositório

- `glossaries/` — glossários técnicos por capítulo, um por integrante (`Glossary - João.md`, `Glossary - Laura`). Cada termo segue o formato: tradução, definição, funcionalidade, relação com outros conceitos, exemplo conceitual.
- `documents/` — relatórios de leitura (ex.: `Relatório de leitura.docx`, respostas de apêndices do livro).
- `Livro - Build a Large Language Model.pdf` — referência principal do projeto.
- `Sprints.txt` — especificação de cada sprint (objetivos, atividades e entregáveis).
- `README.md` — descrição do projeto e critérios de avaliação.

## Convenções

- Arquivos de glossário por pessoa seguem o padrão `Glossary - <Nome>` (sem extensão consistente — alguns são `.md`, outros texto puro).
- Todo o conteúdo (glossários, relatórios, commits) é escrito em português do Brasil.
- Quando código for adicionado nas próximas sprints, ele deve ser organizado de forma incremental (cada sprint constrói sobre os componentes das anteriores), já que o pipeline final integra tokenização → embeddings → atenção → Transformer → treinamento → geração → fine-tuning.
