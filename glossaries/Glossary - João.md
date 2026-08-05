# Glossário — Capítulo 1: Understanding Large Language Models

---

### 1. Large Language Model (LLM)
- **Tradução:** Modelo de Linguagem de Grande Escala
- **Definição:** Rede neural profunda treinada em enormes volumes de texto para compreender, gerar e responder a linguagem humana.
- **Funcionalidade:** Prevê a próxima palavra de uma sequência com base no contexto anterior, aprendendo estrutura, gramática e conhecimento geral no processo.
- **Relação com outros conceitos:** É uma aplicação específica de Deep Learning; depende da arquitetura Transformer e de treino em larga escala (pretraining + fine-tuning).
- **Exemplo conceitual:** O ChatGPT é uma interface para um LLM que recebe uma instrução do usuário e gera texto coerente como resposta.

---

### 2. Natural Language Processing (NLP)
- **Tradução:** Processamento de Linguagem Natural
- **Definição:** Área da IA voltada para permitir que computadores compreendam, interpretem e produzam linguagem humana.
- **Funcionalidade:** Engloba tarefas como tradução, classificação de texto, análise de sentimento e sumarização.
- **Relação com outros conceitos:** LLMs representam o estado da arte atual do NLP, superando métodos tradicionais baseados em regras.
- **Exemplo conceitual:** Um filtro de spam de e-mail é uma aplicação clássica de NLP anterior aos LLMs.

---

### 3. Deep Learning
- **Tradução:** Aprendizado Profundo
- **Definição:** Subcampo do Machine Learning que usa redes neurais com três ou mais camadas ("redes neurais profundas") para modelar padrões complexos.
- **Funcionalidade:** Extrai automaticamente as características (features) relevantes dos dados, sem necessidade de definição manual.
- **Relação com outros conceitos:** Está contido dentro do Machine Learning, que por sua vez está contido na IA; é a base tecnológica dos LLMs.
- **Exemplo conceitual:** Uma rede neural com dezenas de camadas que aprende sozinha a distinguir spam de e-mails legítimos, sem que um humano defina as regras.

---

### 4. Machine Learning (ML)
- **Tradução:** Aprendizado de Máquina
- **Definição:** Campo da IA que desenvolve algoritmos capazes de aprender padrões e fazer previsões a partir de dados, sem programação explícita de regras.
- **Funcionalidade:** Minimiza o erro de previsão sobre um conjunto de dados de treino para "aprender" a tarefa.
- **Relação com outros conceitos:** É mais amplo que Deep Learning (que é um subconjunto seu) e mais restrito que IA.
- **Exemplo conceitual:** Um algoritmo alimentado com e-mails rotulados como "spam" ou "não spam" que aprende a classificar novos e-mails.

---

### 5. Generative AI (GenAI)
- **Tradução:** IA Generativa
- **Definição:** Uso de redes neurais profundas para criar conteúdo novo — texto, imagem ou outras mídias.
- **Funcionalidade:** Gera saídas originais (não apenas classifica ou prevê categorias).
- **Relação com outros conceitos:** LLMs são a principal forma de GenAI voltada a texto.
- **Exemplo conceitual:** Um LLM escrevendo um poema inédito a partir de um comando do usuário.

---

### 6. Transformer
- **Tradução:** Transformador (arquitetura Transformer, geralmente mantida em inglês)
- **Definição:** Arquitetura de rede neural profunda introduzida em 2017 ("Attention Is All You Need"), composta originalmente por um encoder e um decoder, conectados por mecanismos de self-attention.
- **Funcionalidade:** Processa sequências de texto considerando a relação entre todas as palavras simultaneamente, em vez de sequencialmente palavra por palavra (como em RNNs).
- **Relação com outros conceitos:** É a base arquitetural da maioria dos LLMs modernos, incluindo GPT e BERT.
- **Exemplo conceitual:** Um sistema de tradução automática que converte uma frase em inglês para o alemão usando encoder (compreensão) e decoder (geração).

---

### 7. Self-Attention (Mecanismo de Atenção)
- **Tradução:** Autoatenção
- **Definição:** Mecanismo que permite ao modelo ponderar a importância relativa de diferentes palavras/tokens de uma sequência entre si.
- **Funcionalidade:** Captura dependências de longo alcance e relações contextuais no texto, mesmo entre palavras distantes na frase.
- **Relação com outros conceitos:** É o componente central do Transformer; sem ele, o modelo não conseguiria "decidir" quais partes do texto são mais relevantes para gerar cada palavra.
- **Exemplo conceitual:** Na frase "O gato que estava com fome comeu a ração", o mecanismo de atenção ajuda o modelo a associar "comeu" a "gato", mesmo com palavras entre eles.

---

### 8. Encoder
- **Tradução:** Codificador
- **Definição:** Submódulo do Transformer que processa o texto de entrada e o converte em representações numéricas (vetores) que capturam seu contexto.
- **Funcionalidade:** "Entende" o texto de entrada e produz embeddings que serão usados pelo decoder.
- **Relação com outros conceitos:** É a base de modelos como o BERT, que fazem previsão de palavras mascaradas.
- **Exemplo conceitual:** Ao traduzir "This is an example", o encoder transforma essa frase em vetores numéricos que representam seu significado.

---

### 9. Decoder
- **Tradução:** Decodificador
- **Definição:** Submódulo do Transformer que recebe representações vetoriais (do encoder ou do próprio histórico gerado) e produz o texto de saída, uma palavra por vez.
- **Funcionalidade:** Gera texto de forma autoregressiva, usando a saída anterior como parte da entrada seguinte.
- **Relação com outros conceitos:** Modelos GPT usam apenas essa parte do Transformer (arquitetura decoder-only).
- **Exemplo conceitual:** Dado o vetor de "This is an example" e a tradução parcial "Das ist ein", o decoder gera a última palavra "Beispiel".

---

### 10. BERT (Bidirectional Encoder Representations from Transformers)
- **Tradução:** Representações Bidirecionais de Codificador de Transformers (geralmente mantido como sigla)
- **Definição:** Modelo baseado apenas no encoder do Transformer, treinado para prever palavras mascaradas dentro de uma frase.
- **Funcionalidade:** Especializado em tarefas de classificação de texto, como análise de sentimento e categorização de documentos.
- **Relação com outros conceitos:** Contrasta com o GPT, que é decoder-only e voltado à geração de texto.
- **Exemplo conceitual:** O X (antigo Twitter) usa BERT para detectar conteúdo tóxico em publicações.

---

### 11. GPT (Generative Pretrained Transformer)
- **Tradução:** Transformador Pré-treinado Generativo (mantido como sigla)
- **Definição:** Modelo baseado apenas no decoder do Transformer, projetado para gerar texto de forma autoregressiva.
- **Funcionalidade:** Realiza completude de texto e, por consequência, tarefas como tradução, sumarização e escrita criativa.
- **Relação com outros conceitos:** É o foco de implementação do livro; representa a arquitetura decoder-only em oposição à encoder-only (BERT).
- **Exemplo conceitual:** Dado o início "Breakfast is the", o GPT completa com "...most important meal of the day."

---

### 12. Pretraining (Pré-treinamento)
- **Tradução:** Pré-treinamento
- **Definição:** Primeira fase de treino de um LLM, em que o modelo aprende com um grande corpus de texto não rotulado, usando a previsão da próxima palavra como tarefa.
- **Funcionalidade:** Constrói um "modelo base" (foundation model) com compreensão geral da linguagem.
- **Relação com outros conceitos:** Antecede o fine-tuning; usa self-supervised learning.
- **Exemplo conceitual:** O GPT-3 foi pré-treinado com cerca de 300 bilhões de tokens extraídos de textos da internet, livros e Wikipédia.

---

### 13. Fine-tuning
- **Tradução:** Ajuste fino
- **Definição:** Segunda fase de treino, em que um modelo pré-treinado é refinado em um dataset menor e rotulado, voltado a uma tarefa específica.
- **Funcionalidade:** Especializa o modelo base para aplicações como responder perguntas, classificar textos ou seguir instruções.
- **Relação com outros conceitos:** Divide-se em *instruction fine-tuning* e *classification fine-tuning*; depende de um modelo já pré-treinado.
- **Exemplo conceitual:** Pegar um modelo base e treiná-lo com pares de e-mails rotulados como "spam"/"não spam" para criar um classificador.

---

### 14. Self-Supervised Learning
- **Tradução:** Aprendizado autossupervisionado
- **Definição:** Forma de aprendizado em que o próprio modelo gera seus rótulos a partir da estrutura dos dados, sem anotação humana.
- **Funcionalidade:** Permite usar a próxima palavra do texto como "rótulo" automático durante o pré-treinamento.
- **Relação com outros conceitos:** É o que torna possível treinar LLMs com datasets massivos sem rotulagem manual, diferente do aprendizado supervisionado tradicional.
- **Exemplo conceitual:** Ao treinar com a frase "O céu é azul", o modelo usa "azul" como o rótulo que deveria prever ao ver "O céu é ___".

---

### 15. Foundation Model (Modelo Base)
- **Tradução:** Modelo Fundacional / Modelo Base
- **Definição:** Modelo resultante do pré-treinamento, com capacidades gerais de linguagem, que serve de ponto de partida para fine-tuning.
- **Funcionalidade:** Oferece completude de texto e capacidades básicas de few-shot antes de qualquer especialização.
- **Relação com outros conceitos:** Precede e viabiliza o fine-tuning; o GPT-3 é um exemplo clássico.
- **Exemplo conceitual:** Antes do ChatGPT existir, o GPT-3 já era um modelo base capaz de completar textos, mas ainda não ajustado para seguir instruções de forma conversacional.

---

### 16. Instruction Fine-Tuning
- **Tradução:** Ajuste fino por instrução
- **Definição:** Tipo de fine-tuning em que o dataset rotulado é formado por pares de instrução e resposta correta.
- **Funcionalidade:** Ensina o modelo a seguir comandos do usuário (ex.: "traduza este texto").
- **Relação com outros conceitos:** É a técnica usada para transformar o GPT-3 em algo como o ChatGPT (via InstructGPT).
- **Exemplo conceitual:** Par de treino: instrução "Traduza 'cachorro' para o inglês" → resposta "dog".

---

### 17. Classification Fine-Tuning
- **Tradução:** Ajuste fino por classificação
- **Definição:** Tipo de fine-tuning em que o dataset rotulado associa textos a categorias/classes específicas.
- **Funcionalidade:** Especializa o modelo para tarefas de categorização.
- **Relação com outros conceitos:** Contrasta com instruction fine-tuning, que lida com pares instrução-resposta livre.
- **Exemplo conceitual:** Associar o texto de um e-mail à classe "spam" ou "não spam".

---

### 18. Token / Tokenização
- **Tradução:** Token / Tokenização
- **Definição:** Token é a unidade básica de texto que o modelo processa (aproximadamente uma palavra ou parte dela); tokenização é o processo de converter texto bruto nessas unidades.
- **Funcionalidade:** Serve como a "moeda" de entrada e saída do modelo, permitindo contar e processar texto numericamente.
- **Relação com outros conceitos:** É o primeiro passo do pipeline de pré-processamento, antes de gerar embeddings.
- **Exemplo conceitual:** A frase "Eu gosto de café" pode ser dividida em tokens como ["Eu", "gosto", "de", "café"].

---

### 19. Zero-shot Learning
- **Tradução:** Aprendizado sem exemplos
- **Definição:** Capacidade do modelo de realizar uma tarefa totalmente nova sem receber nenhum exemplo prévio dela.
- **Funcionalidade:** Demonstra a generalização do modelo além do que foi explicitamente treinado.
- **Relação com outros conceitos:** Contrasta com few-shot learning; é um exemplo de comportamento emergente.
- **Exemplo conceitual:** Pedir ao modelo "traduza 'breakfast' para o alemão" sem fornecer nenhum exemplo de tradução anterior.

---

### 20. Few-shot Learning
- **Tradução:** Aprendizado com poucos exemplos
- **Definição:** Capacidade do modelo de aprender a realizar uma tarefa a partir de um número pequeno de exemplos fornecidos na própria entrada (prompt).
- **Funcionalidade:** Permite adaptar o comportamento do modelo "na hora", sem re-treinamento.
- **Relação com outros conceitos:** Contrasta com zero-shot; ambos são capacidades emergentes de modelos GPT.
- **Exemplo conceitual:** Fornecer os pares "gaot => goat", "sheo => shoe" e pedir ao modelo para corrigir "pohne".

---

### 21. Emergent Behavior (Comportamento Emergente)
- **Tradução:** Comportamento emergente
- **Definição:** Capacidade que surge no modelo sem ter sido explicitamente ensinada durante o treino, resultado da exposição a dados vastos e diversos.
- **Funcionalidade:** Explica por que um modelo treinado apenas para prever a próxima palavra consegue traduzir, resumir ou classificar textos.
- **Relação com outros conceitos:** Está ligado ao zero-shot e few-shot learning; é uma das razões pelas quais LLMs surpreenderam pesquisadores.
- **Exemplo conceitual:** Um GPT treinado só com previsão de próxima palavra "aprende sozinho" a traduzir inglês-francês, mesmo sem treino específico para tradução.

---

### 22. Autoregressive Model (Modelo Autorregressivo)
- **Tradução:** Modelo autorregressivo
- **Definição:** Tipo de modelo que usa suas próprias saídas anteriores como parte da entrada para gerar a próxima previsão.
- **Funcionalidade:** Gera texto token por token, de forma iterativa e sequencial (esquerda para direita).
- **Relação com outros conceitos:** É a forma como o GPT (decoder-only) funciona.
- **Exemplo conceitual:** Depois de gerar "O céu é", o modelo usa esse trecho inteiro como entrada para prever a próxima palavra, "azul".

---

### 23. Embedding
- **Tradução:** Vetor de incorporação / representação vetorial
- **Definição:** Representação numérica (vetor) de um texto ou palavra que captura seu significado e contexto em múltiplas dimensões.
- **Funcionalidade:** Serve de "ponte" entre texto legível por humanos e cálculos matemáticos que a rede neural consegue processar.
- **Relação com outros conceitos:** É produzido pelo encoder (ou pela camada de entrada do modelo) e consumido pelo decoder ou pelas camadas seguintes.
- **Exemplo conceitual:** A palavra "rei" e a palavra "rainha" teriam embeddings próximos no espaço vetorial, refletindo semelhança semântica.

---

### 24. Parâmetros (Parameters)
- **Tradução:** Parâmetros
- **Definição:** Pesos ajustáveis dentro da rede neural, otimizados durante o treino para minimizar o erro de previsão.
- **Funcionalidade:** Determinam como o modelo transforma a entrada em saída; quanto mais parâmetros, maior geralmente a capacidade do modelo.
- **Relação com outros conceitos:** É o que caracteriza o "tamanho" de um LLM (ex.: GPT-3 tem 175 bilhões de parâmetros).
- **Exemplo conceitual:** Cada "peso" em uma rede neural é como um botão que ajusta o quanto uma informação influencia a previsão final.

---
