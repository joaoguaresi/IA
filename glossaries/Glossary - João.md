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

# Glossário — Capítulo 2: Working with Text Data

---

### 1. Word Embedding
- **Tradução:** Incorporação de palavras / vetor de palavras
- **Definição:** Representação de uma palavra como um vetor de valores contínuos em um espaço vetorial multidimensional, no qual palavras semanticamente próximas ocupam posições próximas.
- **Funcionalidade:** Converte dados categóricos (texto) em dados numéricos compatíveis com as operações matemáticas das redes neurais — é a "ponte" obrigatória entre o texto bruto e o LLM.
- **Relação com outros conceitos:** É a forma mais comum de embedding de texto; existem também embeddings de sentenças, parágrafos e documentos inteiros (usados em RAG). Nos LLMs modernos, os word embeddings são produzidos pela própria *embedding layer* do modelo, e não por um modelo externo.
- **Exemplo conceitual:** Em um embedding bidimensional, "eagle", "duck" e "goose" formam um agrupamento (aves), enquanto "squirrel" fica distante — refletindo a proximidade semântica no espaço vetorial.

---

### 2. Embedding Model
- **Tradução:** Modelo de incorporação
- **Definição:** Modelo (ou camada de rede neural) responsável por transformar dados brutos em representações vetoriais densas.
- **Funcionalidade:** Recebe dados não rotulados (texto, áudio, vídeo) e devolve um vetor numérico que os representa.
- **Relação com outros conceitos:** Cada formato de dado exige um modelo de embedding próprio — um modelo treinado para texto não serve para áudio ou vídeo.
- **Exemplo conceitual:** Um trecho de áudio passa por um *audio embedding model* e vira o vetor `[-0.15, 0.45, 2.11]`; o mesmo trecho não poderia ser processado por um modelo de embedding de texto.

---

### 3. Word2Vec
- **Tradução:** Word2Vec (mantido em inglês)
- **Definição:** Abordagem clássica e anterior aos LLMs para gerar word embeddings, treinando uma rede neural para prever o contexto de uma palavra (ou a palavra a partir do contexto).
- **Funcionalidade:** Baseia-se na hipótese distribucional: palavras que aparecem em contextos semelhantes tendem a ter significados semelhantes.
- **Relação com outros conceitos:** É uma alternativa *pré-treinada e estática* às embedding layers dos LLMs. LLMs preferem gerar seus próprios embeddings porque estes são otimizados para a tarefa e os dados específicos durante o treino.
- **Exemplo conceitual:** Projetando embeddings do Word2Vec em duas dimensões, "Germany"–"Berlin" e "England"–"London" aparecem com relações espaciais análogas.

---

### 4. Embedding Dimensionality (dimensão do embedding)
- **Tradução:** Dimensionalidade do embedding
- **Definição:** Número de componentes (valores) que compõem o vetor de cada token; também chamada de dimensionalidade dos *hidden states* do modelo.
- **Funcionalidade:** Define quanta nuance semântica o vetor consegue capturar; é um trade-off direto entre desempenho e custo computacional.
- **Relação com outros conceitos:** Determina o número de colunas da matriz de pesos da embedding layer e precisa ser igual à dimensão dos positional embeddings para que a soma seja possível.
- **Exemplo conceitual:** Os menores modelos GPT-2 (117M e 125M parâmetros) usam 768 dimensões; o maior GPT-3 (175B parâmetros) usa 12.288 dimensões. No código do livro usa-se 256 dimensões por praticidade.

---

### 5. Retrieval-Augmented Generation (RAG)
- **Tradução:** Geração aumentada por recuperação
- **Definição:** Técnica que combina geração de texto com a busca em uma base de conhecimento externa para trazer informações relevantes no momento da geração.
- **Funcionalidade:** Permite ao modelo consultar dados que não estão em seus parâmetros, reduzindo alucinações e ampliando o conhecimento disponível.
- **Relação com outros conceitos:** Depende de embeddings de sentenças, parágrafos ou documentos (e não de word embeddings). O livro apenas menciona o conceito — está fora do seu escopo.
- **Exemplo conceitual:** Um assistente que, antes de responder, busca os parágrafos mais similares em um manual interno e os usa como contexto para gerar a resposta.

---

### 6. Tokenization / Tokenizer
- **Tradução:** Tokenização / tokenizador
- **Definição:** Processo (e o objeto que o implementa) de dividir um texto bruto em unidades individuais — tokens — que podem ser palavras, subpalavras ou caracteres especiais, incluindo pontuação.
- **Funcionalidade:** É o primeiro passo obrigatório do pipeline de pré-processamento, antes da conversão em token IDs e embeddings.
- **Relação com outros conceitos:** Todo tokenizer implementa dois métodos complementares: `encode` e `decode`. A escolha do esquema (regex simples vs. BPE) determina como palavras desconhecidas serão tratadas.
- **Exemplo conceitual:** `"Hello, world. Is this-- a test?"` é dividido em 10 tokens: `['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']`. Aplicado ao conto "The Verdict" (20.479 caracteres), o esquema gera 4.690 tokens.

---

### 7. Vocabulary (Vocabulário)
- **Tradução:** Vocabulário
- **Definição:** Dicionário que mapeia cada token único do conjunto de treino a um número inteiro exclusivo.
- **Funcionalidade:** Serve como tabela de tradução entre texto e números; seu tamanho define quantas linhas terá a matriz de embeddings.
- **Relação com outros conceitos:** É construído a partir dos tokens únicos, geralmente ordenados alfabeticamente; sua limitação motiva a existência dos tokens especiais e do BPE.
- **Exemplo conceitual:** Em "The Verdict" o vocabulário tem 1.130 entradas (`('!', 0)`, `('"', 1)`, …); ao adicionar `<|endoftext|>` e `<|unk|>`, passa a 1.132. O tokenizador BPE do GPT-2 tem 50.257 entradas.

---

### 8. Token ID
- **Tradução:** Identificador de token
- **Definição:** Inteiro único associado a um token pelo vocabulário.
- **Funcionalidade:** Representação intermediária entre o token em texto e o vetor de embedding — é o índice usado para buscar a linha correspondente na matriz de pesos.
- **Relação com outros conceitos:** Produzido pelo `encode`, revertido pelo `decode` e consumido pela embedding layer.
- **Exemplo conceitual:** A frase `"It's the last he painted, you know,"` vira `[1, 56, 2, 850, 988, 602, 533, 746, 5, ...]`.

---

### 9. Encode (método `encode`)
- **Tradução:** Codificar
- **Definição:** Método do tokenizador que recebe texto bruto, o divide em tokens e converte cada token no seu token ID via vocabulário.
- **Funcionalidade:** Executa, em uma única chamada, tokenização + mapeamento string→inteiro.
- **Relação com outros conceitos:** É a operação inversa do `decode`; no BPE (tiktoken) já entrega diretamente os IDs, dispensando etapas intermediárias.
- **Exemplo conceitual:** `tokenizer.encode("Hello, do you like tea?")` retorna a lista de IDs correspondente — ou lança `KeyError: 'Hello'` se a palavra não estiver no vocabulário (caso do `SimpleTokenizerV1`).

---

### 10. Decode (método `decode`) / Detokenização
- **Tradução:** Decodificar / destokenizar
- **Definição:** Método que faz o mapeamento inverso inteiro→string, reconstruindo o texto legível a partir de uma lista de token IDs.
- **Funcionalidade:** Usa um *vocabulário invertido* (`int_to_str`) e ainda aplica ajustes de formatação, como remover espaços antes de sinais de pontuação.
- **Relação com outros conceitos:** É indispensável para converter a saída numérica do LLM de volta em texto para o usuário.
- **Exemplo conceitual:** `tokenizer.decode(ids)` devolve `'" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'`

---

### 11. Special Context Tokens (Tokens especiais de contexto)
- **Tradução:** Tokens especiais de contexto
- **Definição:** Tokens artificiais adicionados ao vocabulário para sinalizar situações que não são palavras do texto — palavras desconhecidas, fronteiras entre documentos, preenchimento, início e fim de sequência.
- **Funcionalidade:** Ampliam a compreensão contextual do modelo e permitem o processamento robusto de textos heterogêneos.
- **Relação com outros conceitos:** Diferentes famílias de modelos usam conjuntos distintos; os modelos GPT usam apenas `<|endoftext|>`.
- **Exemplo conceitual:** Ao concatenar dois textos independentes: `"Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."`

---

### 12. `<|unk|>` (Unknown Token)
- **Tradução:** Token de desconhecido
- **Definição:** Token especial usado para representar qualquer palavra que não conste do vocabulário.
- **Funcionalidade:** Impede que o tokenizador falhe (`KeyError`) ao encontrar palavras fora do vocabulário, substituindo-as por um marcador genérico.
- **Relação com outros conceitos:** É a solução do `SimpleTokenizerV2`; os modelos GPT **não** o utilizam, pois o BPE resolve o mesmo problema sem perda de informação.
- **Exemplo conceitual:** Como "Hello" e "palace" não aparecem em "The Verdict", a saída decodificada vira `<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.` — evidenciando a perda de informação.

---

### 13. `<|endoftext|>`
- **Tradução:** Token de fim de texto
- **Definição:** Token especial inserido entre fontes de texto independentes e não relacionadas.
- **Funcionalidade:** Sinaliza ao LLM que, embora os textos estejam concatenados no corpus de treino, eles não têm continuidade semântica entre si; nos modelos GPT também cumpre o papel de padding.
- **Relação com outros conceitos:** É análogo ao `[EOS]`; no vocabulário do BPE do GPT-2 recebe o maior ID, 50256 (de um total de 50.257 tokens).
- **Exemplo conceitual:** Ao treinar com dois artigos da Wikipédia seguidos, insere-se `<|endoftext|>` entre eles para que o modelo não trate o fim de um como início natural do outro.

---

### 14. `[BOS]`, `[EOS]` e `[PAD]`
- **Tradução:** Início de sequência / fim de sequência / preenchimento
- **Definição:** Conjunto alternativo de tokens especiais usado por alguns modelos: `[BOS]` marca o início de um texto, `[EOS]` marca o fim, e `[PAD]` completa textos curtos.
- **Funcionalidade:** `[PAD]` garante que todos os exemplos de um *batch* tenham o mesmo comprimento, igualando-os ao texto mais longo do lote.
- **Relação com outros conceitos:** O tokenizador do GPT dispensa os três: usa apenas `<|endoftext|>`. Além disso, como no treino em lote se aplica uma *máscara* (o modelo não atende aos tokens de padding), a escolha específica do token de preenchimento se torna irrelevante.
- **Exemplo conceitual:** Em um batch com frases de 5 e 8 tokens, a primeira recebe três `[PAD]` ao final para alinhar os comprimentos.

---

### 15. Out-of-Vocabulary (OOV)
- **Tradução:** Fora do vocabulário
- **Definição:** Condição de uma palavra que aparece no texto de entrada, mas não existe no vocabulário construído a partir do conjunto de treino.
- **Funcionalidade (problema que gera):** Sem tratamento, quebra o `encode`; com `<|unk|>`, perde-se o significado; com BPE, a palavra é decomposta e preservada.
- **Relação com outros conceitos:** Motiva tanto o uso de conjuntos de treino grandes e diversos quanto a adoção de tokenização em subpalavras.
- **Exemplo conceitual:** `"Hello"` é OOV em relação a "The Verdict", gerando `KeyError: 'Hello'` no `SimpleTokenizerV1`.

---

### 16. Byte Pair Encoding (BPE)
- **Tradução:** Codificação por pares de bytes (geralmente mantida como sigla)
- **Definição:** Esquema avançado de tokenização que constrói o vocabulário fundindo iterativamente caracteres frequentes em subpalavras e subpalavras frequentes em palavras.
- **Funcionalidade:** Permite representar *qualquer* palavra, mesmo inédita, decompondo-a em subpalavras ou caracteres individuais — eliminando a necessidade do `<|unk|>`.
- **Relação com outros conceitos:** Foi usado no treino do GPT-2, GPT-3 e do modelo original do ChatGPT; as fusões são determinadas por um limiar de frequência.
- **Exemplo conceitual:** `"Akwirw ier"` é quebrado em `["Ak", "w", "ir", "w", " ", "ier"]` → `[33901, 86, 343, 86, 220, 959]`. Já `"de"` é uma subpalavra frequente por aparecer em "define", "depend", "made" e "hidden".

---

### 17. Subword Units (Unidades de subpalavra)
- **Tradução:** Unidades de subpalavra
- **Definição:** Fragmentos de palavra (maiores que um caractere e menores que uma palavra completa) que compõem o vocabulário de tokenizadores como o BPE.
- **Funcionalidade:** Equilibram o tamanho do vocabulário e a cobertura da língua: poucos milhares de unidades cobrem virtualmente qualquer palavra.
- **Relação com outros conceitos:** São o mecanismo que torna o BPE imune a palavras desconhecidas.
- **Exemplo conceitual:** `someunknownPlace` é codificada e decodificada corretamente pelo BPE, embora certamente não figure como palavra inteira em seu vocabulário.

---

### 18. tiktoken
- **Tradução:** tiktoken (nome de biblioteca, mantido em inglês)
- **Definição:** Biblioteca open source da OpenAI que implementa o algoritmo BPE de forma eficiente, com núcleo escrito em Rust.
- **Funcionalidade:** Fornece tokenizadores prontos (`tiktoken.get_encoding("gpt2")`) com a mesma interface `encode`/`decode` de um tokenizador manual.
- **Relação com outros conceitos:** Substitui, na prática, o `SimpleTokenizerV2` implementado didaticamente no capítulo.
- **Exemplo conceitual:** `tokenizer.encode(text, allowed_special={"<|endoftext|>"})` retorna `[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, ...]`.

---

### 19. Input–Target Pairs (Pares entrada–alvo)
- **Tradução:** Pares entrada–alvo
- **Definição:** Estrutura de dados de treino em que a entrada `x` é uma sequência de tokens e o alvo `y` é essa mesma sequência deslocada uma posição à direita.
- **Funcionalidade:** Materializa a tarefa de previsão da próxima palavra: para cada posição, o modelo vê o contexto anterior e deve prever o token seguinte.
- **Relação com outros conceitos:** São gerados pela abordagem de janela deslizante e entregues ao modelo como tensores; durante o treino, tudo o que está além do alvo é mascarado.
- **Exemplo conceitual:** `x: [290, 4920, 2241, 287]` e `y: [4920, 2241, 287, 257]`; em texto: `"and established himself in" ----> "a"`.

---

### 20. Sliding Window (Janela deslizante)
- **Tradução:** Janela deslizante
- **Definição:** Estratégia de amostragem que percorre o texto tokenizado extraindo blocos consecutivos de tamanho fixo, avançando uma quantidade definida de posições a cada extração.
- **Funcionalidade:** Gera múltiplos exemplos de treino a partir de um único texto contínuo, maximizando o aproveitamento do corpus.
- **Relação com outros conceitos:** É parametrizada por `max_length` (tamanho da janela) e `stride` (passo do deslocamento); implementada no `GPTDatasetV1`.
- **Exemplo conceitual:** Com janela 4 e passo 1, o lote 1 é `"In the heart of"` e o lote 2 é `"the heart of the"`.

---

### 21. Context Size / Context Length
- **Tradução:** Tamanho de contexto / comprimento de contexto
- **Definição:** Número máximo de tokens que o modelo consegue receber como entrada de uma só vez.
- **Funcionalidade:** Determina quantos tokens formam cada bloco de entrada e, portanto, quanta informação anterior o modelo pode considerar ao prever o próximo token.
- **Relação com outros conceitos:** No código corresponde a `max_length` e também define o número de linhas da camada de positional embeddings. Textos mais longos que o contexto suportado precisam ser truncados.
- **Exemplo conceitual:** O livro usa `context_size = 4` por simplicidade didática, mas observa que é comum treinar LLMs com entradas de pelo menos 256 tokens.

---

### 22. Stride (Passo)
- **Tradução:** Passo / deslocamento
- **Definição:** Número de posições que a janela deslizante avança entre um lote e o seguinte.
- **Funcionalidade:** Controla o grau de sobreposição entre exemplos de treino: passo igual ao tamanho da janela elimina a sobreposição; passo menor cria redundância.
- **Relação com outros conceitos:** Sobreposição excessiva pode aumentar o *overfitting*; por isso, usa-se `stride = max_length` para percorrer o dataset sem repetir nem pular palavras.
- **Exemplo conceitual:** Com `stride=1`, os IDs do segundo lote são os do primeiro deslocados em uma posição: `[40, 367, 2885, 1464]` → `[367, 2885, 1464, 1807]`.

---

### 23. Batch / Batch Size (Lote)
- **Tradução:** Lote / tamanho de lote
- **Definição:** Conjunto de exemplos de treino processados simultaneamente em uma única passagem pelo modelo.
- **Funcionalidade:** Lotes pequenos exigem menos memória, mas produzem atualizações de pesos mais ruidosas; lotes grandes suavizam as atualizações ao custo de memória.
- **Relação com outros conceitos:** É um hiperparâmetro clássico de deep learning; a necessidade de igualar comprimentos dentro de um lote é o que justifica o token `[PAD]`.
- **Exemplo conceitual:** Com `batch_size=8` e `max_length=4`, o tensor de entrada tem forma `torch.Size([8, 4])` — oito amostras de quatro tokens.

---

### 24. Tensor
- **Tradução:** Tensor
- **Definição:** Estrutura de dados multidimensional (generalização de vetores e matrizes) usada pelo PyTorch para representar entradas, alvos e pesos.
- **Funcionalidade:** Permite operações matemáticas vetorizadas e diferenciáveis, essenciais para o treino via backpropagation.
- **Relação com outros conceitos:** O data loader devolve um tensor `x` (entradas) e um tensor `y` (alvos); após a embedding layer, os dados ganham uma dimensão extra.
- **Exemplo conceitual:** Um lote de 8 amostras × 4 tokens × 256 dimensões forma um tensor `torch.Size([8, 4, 256])`.

---

### 25. Dataset (classe do PyTorch)
- **Tradução:** Conjunto de dados (classe `Dataset`)
- **Definição:** Classe do PyTorch que define como cada exemplo individual do conjunto de treino é armazenado e recuperado.
- **Funcionalidade:** Implementa `__len__` (quantidade total de linhas) e `__getitem__` (retorno de uma linha específica), encapsulando a lógica da janela deslizante.
- **Relação com outros conceitos:** É consumida pelo `DataLoader`, que se encarrega de agrupar as linhas em lotes.
- **Exemplo conceitual:** O `GPTDatasetV1` tokeniza todo o texto e preenche as listas `input_ids` e `target_ids` com pares de chunks deslocados em uma posição.

---

### 26. DataLoader (classe do PyTorch)
- **Tradução:** Carregador de dados
- **Definição:** Classe do PyTorch que itera sobre um `Dataset`, agrupando exemplos em lotes e devolvendo-os como tensores.
- **Funcionalidade:** Gerencia embaralhamento (`shuffle`), descarte do último lote incompleto (`drop_last`) e paralelismo de pré-processamento (`num_workers`).
- **Relação com outros conceitos:** `drop_last=True` evita picos de perda (*loss spikes*) causados por um lote final menor que os demais.
- **Exemplo conceitual:** `create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)` devolve lotes prontos para alimentar o modelo.

---

### 27. Embedding Layer (`torch.nn.Embedding`)
- **Tradução:** Camada de embedding
- **Definição:** Camada da rede neural que armazena uma matriz de pesos com uma linha por token do vocabulário e uma coluna por dimensão do embedding.
- **Funcionalidade:** Converte token IDs em vetores contínuos; seus pesos são inicializados aleatoriamente e otimizados durante o treino do LLM.
- **Relação com outros conceitos:** É a alternativa moderna ao Word2Vec, com a vantagem de ser otimizada para a tarefa específica; suas dimensões são definidas por `vocab_size` e `output_dim`.
- **Exemplo conceitual:** `torch.nn.Embedding(50257, 256)` cria a matriz de embeddings de um modelo com vocabulário BPE completo e vetores de 256 dimensões.

---

### 28. Weight Matrix (Matriz de pesos do embedding)
- **Tradução:** Matriz de pesos
- **Definição:** Tabela numérica interna da embedding layer, com formato `vocab_size × output_dim`, inicializada com valores aleatórios pequenos.
- **Funcionalidade:** Cada linha é o vetor de embedding de um token específico; esses valores são ajustados por backpropagation ao longo do treino.
- **Relação com outros conceitos:** É acessada por meio de uma *lookup operation* indexada pelo token ID.
- **Exemplo conceitual:** Com `vocab_size=6` e `output_dim=3`, a matriz tem 6 linhas e 3 colunas, começando em `[0.3374, -0.1778, -0.1690]` na linha 0.

---

### 29. Lookup Operation (Operação de busca)
- **Tradução:** Operação de consulta / busca em tabela
- **Definição:** Mecanismo pelo qual a embedding layer recupera diretamente a linha da matriz de pesos correspondente a um token ID.
- **Funcionalidade:** Torna a conversão ID→vetor extremamente eficiente, sem qualquer cálculo além do acesso indexado.
- **Relação com outros conceitos:** É matematicamente equivalente a uma codificação one-hot seguida de multiplicação matricial em uma camada totalmente conectada — apenas muito mais eficiente. Por isso, continua sendo uma camada treinável por backpropagation.
- **Exemplo conceitual:** `embedding_layer(torch.tensor([3]))` devolve `[-0.4015, 0.9666, -1.1481]`, que é exatamente a quarta linha da matriz (índice 3, pois Python conta a partir de 0).

---

### 30. One-Hot Encoding
- **Tradução:** Codificação one-hot
- **Definição:** Representação de uma categoria como um vetor de zeros com um único valor 1 na posição correspondente àquela categoria.
- **Funcionalidade:** Serve como referência conceitual para entender a embedding layer, mas é ineficiente na prática por gerar vetores enormes e esparsos.
- **Relação com outros conceitos:** One-hot + multiplicação matricial ≡ lookup na embedding layer; a segunda forma é apenas a implementação otimizada da primeira.
- **Exemplo conceitual:** Em um vocabulário de 6 palavras, o token de ID 3 seria `[0, 0, 0, 1, 0, 0]`; multiplicá-lo pela matriz de pesos seleciona exatamente a quarta linha.

---

### 31. Backpropagation
- **Tradução:** Retropropagação
- **Definição:** Algoritmo que calcula os gradientes do erro em relação a cada peso da rede, propagando-os da saída de volta para as camadas anteriores.
- **Funcionalidade:** Permite ajustar todos os parâmetros — inclusive os pesos da embedding layer — para reduzir o erro de previsão.
- **Relação com outros conceitos:** É a razão pela qual os embeddings precisam ser vetores contínuos: valores discretos não são diferenciáveis e, portanto, não poderiam ser otimizados.
- **Exemplo conceitual:** A cada lote processado, os vetores de embedding dos tokens envolvidos são levemente deslocados na direção que diminui o erro de previsão da próxima palavra.

---

### 32. Positional Embeddings
- **Tradução:** Embeddings posicionais
- **Definição:** Vetores que codificam a posição de cada token dentro da sequência, somados aos token embeddings.
- **Funcionalidade:** Corrigem uma limitação estrutural do mecanismo de self-attention, que é *position-agnostic* — sem eles, o modelo não distinguiria "o cão mordeu o homem" de "o homem mordeu o cão".
- **Relação com outros conceitos:** Precisam ter exatamente a mesma dimensionalidade dos token embeddings para que a soma seja válida. Dividem-se em absolutos e relativos.
- **Exemplo conceitual:** `torch.nn.Embedding(context_length, output_dim)` com `torch.arange(4)` gera 4 vetores de 256 dimensões, um para cada posição da janela.

---

### 33. Absolute Positional Embeddings
- **Tradução:** Embeddings posicionais absolutos
- **Definição:** Embeddings associados diretamente a posições específicas da sequência — a posição 1 tem um vetor próprio, a posição 2 outro, e assim por diante.
- **Funcionalidade:** Transmitem ao modelo a localização exata de cada token.
- **Relação com outros conceitos:** É a abordagem usada pelos modelos GPT da OpenAI, com a particularidade de serem *aprendidos durante o treino*, e não fixos/predefinidos como as codificações posicionais senoidais do Transformer original.
- **Exemplo conceitual:** Em uma janela de 4 tokens, existem exatamente 4 vetores posicionais, reutilizados em todos os lotes.

---

### 34. Relative Positional Embeddings
- **Tradução:** Embeddings posicionais relativos
- **Definição:** Embeddings que codificam a distância entre tokens em vez de sua posição absoluta.
- **Funcionalidade:** O modelo aprende relações do tipo "a que distância", o que favorece a generalização para sequências de comprimentos não vistos durante o treino.
- **Relação com outros conceitos:** Alternativa aos embeddings absolutos; a escolha entre os dois depende da aplicação e da natureza dos dados.
- **Exemplo conceitual:** Em vez de registrar que um token está na posição 37, o modelo registra que ele está 3 posições à esquerda do token atual.

---

### 35. Input Embeddings
- **Tradução:** Embeddings de entrada
- **Definição:** Resultado final do pipeline de pré-processamento: a soma elemento a elemento dos token embeddings com os positional embeddings.
- **Funcionalidade:** São o tensor efetivamente entregue às camadas principais do LLM (a partir do bloco de atenção).
- **Relação com outros conceitos:** Fecham o pipeline texto → tokens → token IDs → token embeddings → (+ posicionais) → input embeddings.
- **Exemplo conceitual:** `input_embeddings = token_embeddings + pos_embeddings` produz um tensor `torch.Size([8, 4, 256])` — o broadcasting do PyTorch soma o tensor posicional 4×256 a cada uma das oito amostras do lote.

---

### 36. Overfitting (no contexto da amostragem)
- **Tradução:** Sobreajuste
- **Definição:** Situação em que o modelo memoriza padrões específicos do conjunto de treino em vez de generalizar.
- **Funcionalidade (relevância no capítulo):** Justifica a escolha de `stride` igual ao `max_length` — sobreposição excessiva entre lotes significa mostrar os mesmos trechos repetidamente ao modelo, aumentando o risco de sobreajuste.
- **Relação com outros conceitos:** Conecta decisões de amostragem de dados (stride, janela deslizante) à qualidade final da generalização do LLM.
- **Exemplo conceitual:** Com `stride=1` e janela 4, cada token aparece em até 4 exemplos diferentes; com `stride=4`, aparece em apenas um.

---

### 37. Hyperparameter (Hiperparâmetro)
- **Tradução:** Hiperparâmetro
- **Definição:** Configuração definida pelo desenvolvedor antes do treino, que não é aprendida pelo modelo.
- **Funcionalidade:** Controla o comportamento do processo de treino e deve ser ajustado experimentalmente.
- **Relação com outros conceitos:** Contrasta com os *parâmetros* (pesos), que são aprendidos por backpropagation. No capítulo, são hiperparâmetros: `batch_size`, `max_length`, `stride` e `output_dim`.
- **Exemplo conceitual:** Escolher `batch_size=8` em vez de `batch_size=1` é um trade-off entre uso de memória e estabilidade das atualizações de peso.

---
