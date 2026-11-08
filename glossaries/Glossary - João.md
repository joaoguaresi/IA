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

# Glossário — Capítulo 3: Coding Attention Mechanisms

---

### 1. Attention Mechanism (Mecanismo de Atenção)
- **Tradução:** Mecanismo de atenção
- **Definição:** Componente que permite a um modelo ponderar dinamicamente a relevância de diferentes partes de uma sequência de entrada ao processar cada elemento, superando a limitação das RNNs encoder-decoder de comprimir toda a entrada em um único estado oculto.
- **Funcionalidade:** Dá ao decodificador acesso seletivo a todos os tokens de entrada, e não apenas ao último estado oculto do encoder, permitindo capturar dependências de longo alcance.
- **Relação com outros conceitos:** Motivado pela limitação das RNNs encoder-decoder (seção 3.1) e pelo mecanismo de Bahdanau Attention (2014); é generalizado pelo self-attention usado no Transformer (Capítulo 1).
- **Exemplo conceitual:** Ao traduzir "Kannst du mir helfen diesen Satz zu übersetzen", o modelo pode acessar diretamente palavras distantes da entrada alemã ao gerar cada palavra da tradução em inglês, em vez de depender só do estado oculto final do encoder.

---

### 2. Bahdanau Attention
- **Tradução:** Atenção de Bahdanau
- **Definição:** Mecanismo de atenção proposto em 2014 para RNNs encoder-decoder, que permite ao decoder acessar seletivamente diferentes partes da sequência de entrada a cada passo de decodificação.
- **Funcionalidade:** Resolveu o gargalo das RNNs, que precisavam codificar toda a frase de entrada em um único vetor de estado oculto antes de iniciar a decodificação.
- **Relação com outros conceitos:** Precursor histórico do self-attention; três anos depois, pesquisadores mostraram que RNNs nem eram necessárias, propondo a arquitetura Transformer com self-attention.
- **Exemplo conceitual:** Ao gerar o segundo token de saída de uma tradução, o modelo pondera com pesos diferentes (proporcionais à importância) cada token da frase alemã de entrada, em vez de usar só o último estado oculto.

---

### 3. Context Vector (Vetor de Contexto)
- **Tradução:** Vetor de contexto
- **Definição:** Embedding enriquecido de um token, obtido como soma ponderada de todos os vetores de entrada (ou de valor) da sequência, incorporando informação de todo o contexto.
- **Funcionalidade:** Substitui o embedding "estático" do token por uma representação que já reflete sua relação com os demais tokens da sequência.
- **Relação com outros conceitos:** É a saída de qualquer variante de self-attention; no livro, z(2) é o vetor de contexto do token "journey" na frase "Your journey starts with one step."
- **Exemplo conceitual:** No self-attention simplificado, `z(2) = tensor([0.4419, 0.6515, 0.5683])`, calculado como soma ponderada dos 6 vetores de entrada usando os pesos de atenção relativos a "journey".

---

### 4. Attention Score (Score de Atenção)
- **Tradução:** Score (pontuação) de atenção
- **Definição:** Valor intermediário e não normalizado que mede a similaridade entre um token de consulta e cada token da sequência, calculado via produto escalar.
- **Funcionalidade:** É a primeira das três etapas do self-attention (score → peso → vetor de contexto); quanto maior o produto escalar, maior a similaridade entre os vetores.
- **Relação com outros conceitos:** É normalizado via softmax para se tornar um attention weight; no self-attention com pesos treináveis, é calculado entre query e key em vez de entre inputs brutos.
- **Exemplo conceitual:** Os scores do token "journey" (query) contra os 6 tokens de "Your journey starts with one step" são `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]`.

---

### 5. Attention Weight (Peso de Atenção)
- **Tradução:** Peso de atenção
- **Definição:** Score de atenção normalizado (via softmax) de modo que os pesos relativos a cada consulta somem 1.
- **Funcionalidade:** Determina o quanto cada token de entrada contribui para o vetor de contexto do token atual; funciona como uma distribuição de probabilidade interpretável.
- **Relação com outros conceitos:** Não deve ser confundido com os *weight parameters* (Wq, Wk, Wv) — pesos de atenção são dinâmicos e dependem da entrada, enquanto os parâmetros são aprendidos e fixos após o treino.
- **Exemplo conceitual:** Os pesos de atenção de "journey" somam 1: `[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]`.

---

### 6. Self-Attention Simplificado (sem pesos treináveis)
- **Tradução:** Self-attention simplificado
- **Definição:** Variante didática do self-attention em que os vetores de contexto são calculados diretamente a partir dos produtos escalares entre os próprios embeddings de entrada, sem nenhuma matriz de pesos treinável.
- **Funcionalidade:** Introduz as três etapas centrais do self-attention — scores, pesos (softmax) e vetor de contexto — antes de adicionar complexidade.
- **Relação com outros conceitos:** Generalizado para todos os tokens simultaneamente via `attn_scores = inputs @ inputs.T`; é a base sobre a qual se constrói o self-attention com pesos treináveis. Implementado em `self_attention_simples`/`attention_scores_simples` em `src/attention.py`.
- **Exemplo conceitual:** Para os 6 tokens de "Your journey starts with one step", a matriz de pesos de atenção completa tem forma (6, 6), e `all_context_vecs = attn_weights @ inputs` gera os 6 vetores de contexto de uma vez.

---

### 7. Query, Key e Value (Consulta, Chave e Valor)
- **Tradução:** Consulta, chave e valor
- **Definição:** Três projeções vetoriais de cada token de entrada, obtidas via matrizes de pesos treináveis Wq, Wk e Wv, inspiradas em sistemas de recuperação de informação (bancos de dados).
- **Funcionalidade:** A query representa o token atual "procurando" informação; a key é usada para casar com a query (via produto escalar); a value é o conteúdo efetivamente recuperado e combinado no vetor de contexto.
- **Relação com outros conceitos:** São a base do scaled dot-product attention; substituem, no self-attention treinável, o uso direto dos embeddings de entrada do self-attention simplificado.
- **Exemplo conceitual:** Para "journey" (query) com `d_in=3` e `d_out=2`, `query_2 = x_2 @ W_query` resulta em `tensor([0.4306, 1.4551])`.

---

### 8. Weight Matrices Wq, Wk, Wv (Matrizes de Pesos de Atenção)
- **Tradução:** Matrizes de pesos de query, key e value
- **Definição:** Matrizes treináveis (de forma `d_in × d_out`) que projetam os embeddings de entrada nos espaços de query, key e value.
- **Funcionalidade:** São ajustadas por backpropagation durante o treino para que o modelo aprenda a produzir vetores de contexto úteis à tarefa de prever o próximo token.
- **Relação com outros conceitos:** Implementadas primeiro com `nn.Parameter(torch.rand(...))` (`SelfAttention_v1`) e depois com `nn.Linear` (`SelfAttention_v2`), que usa um esquema de inicialização mais estável para o treino.
- **Exemplo conceitual:** `W_query = torch.nn.Parameter(torch.rand(d_in, d_out))` inicializa a matriz de pesos de query com `d_in=3` linhas e `d_out=2` colunas.

---

### 9. Scaled Dot-Product Attention
- **Tradução:** Atenção por produto escalar escalonado
- **Definição:** Mecanismo de self-attention com pesos treináveis (Wq, Wk, Wv) em que os scores de atenção são divididos pela raiz quadrada da dimensão das keys antes do softmax.
- **Funcionalidade:** A escala evita que produtos escalares muito grandes (comuns quando a dimensão do embedding é alta, >1.000 em LLMs reais) levem o softmax a se comportar como uma função degrau, gerando gradientes próximos de zero e travando o treino.
- **Relação com outros conceitos:** É o self-attention usado na arquitetura Transformer original e nos modelos GPT; implementado em `scaled_dot_product_attention`/`SelfAttention_v1`/`SelfAttention_v2`, servindo de base para `CausalAttention`.
- **Exemplo conceitual:** `attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)`, com `d_k = keys.shape[-1] = 2`, produz `[0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]`.

---

### 10. Causal Attention (Atenção Causal) / Masked Attention
- **Tradução:** Atenção causal / atenção mascarada
- **Definição:** Forma especializada de self-attention em que cada token só pode atender a si mesmo e aos tokens anteriores da sequência, nunca aos futuros.
- **Funcionalidade:** Garante a ordem temporal necessária para a geração autorregressiva de texto — ao prever o próximo token, o modelo não pode "espiar" tokens que ainda não deveria conhecer.
- **Relação com outros conceitos:** Implementada zerando (ou definindo como `-inf`) os pesos/scores de atenção acima da diagonal principal da matriz de atenção; é essencial para os modelos GPT (decoder-only). Implementada em `CausalAttention` (`src/attention.py`).
- **Exemplo conceitual:** Para "journey" (linha 2), a atenção só é mantida para "Your" e "journey"; os pesos para "starts", "with", "one" e "step" são zerados e a linha é renormalizada para somar 1.

---

### 11. Causal Mask (Máscara Causal)
- **Tradução:** Máscara causal
- **Definição:** Matriz triangular usada para bloquear a atenção a posições futuras — via `torch.tril` (1s abaixo/na diagonal) combinada com multiplicação, ou via `torch.triu` (1s acima da diagonal) combinada com `masked_fill(-inf)`.
- **Funcionalidade:** A abordagem com `-inf` antes do softmax é mais eficiente: como e^-∞ = 0, o softmax já produz pesos zerados e normalizados em uma única etapa, sem precisar renormalizar depois.
- **Relação com outros conceitos:** Está ligada ao problema do information leakage; é armazenada como buffer (`register_buffer`) na classe `CausalAttention`, para ser movida automaticamente entre CPU/GPU junto com o modelo.
- **Exemplo conceitual:** `mask = torch.triu(torch.ones(6, 6), diagonal=1)` seguido de `attn_scores.masked_fill(mask.bool(), -torch.inf)` produz scores como `tensor([[0.2899, -inf, -inf, -inf, -inf, -inf], ...])`.

---

### 12. Information Leakage (Vazamento de Informação)
- **Tradução:** Vazamento de informação
- **Definição:** Risco aparente de que tokens futuros, mesmo mascarados, ainda influenciem o resultado por terem participado do cálculo original do softmax antes da renormalização.
- **Funcionalidade (por que não ocorre):** Matematicamente, renormalizar os pesos após zerar as posições futuras equivale a calcular o softmax apenas sobre o subconjunto de posições não mascaradas — como se as posições futuras nunca tivessem existido no cálculo.
- **Relação com outros conceitos:** Justifica por que a máscara causal (com renormalização, ou com `-inf` antes do softmax) é matematicamente correta e segura para geração autorregressiva.
- **Exemplo conceitual:** A linha de "journey" no exemplo do livro soma exatamente 1,0 mesmo depois de zerar os pesos das posições futuras e redividir os valores restantes pela soma da linha.

---

### 13. Dropout (na Atenção)
- **Tradução:** Dropout (na atenção)
- **Definição:** Técnica que zera aleatoriamente uma fração dos pesos de atenção durante o treino, reescalando os valores remanescentes pelo inverso da taxa mantida (ex.: ×2 para dropout de 50%).
- **Funcionalidade:** Reduz o overfitting evitando que o modelo dependa excessivamente de conexões de atenção específicas; é desativado na inferência.
- **Relação com outros conceitos:** Aplicado após o cálculo dos pesos de atenção (convenção adotada no livro) e integrado à `CausalAttention`/`MultiHeadAttention` via `self.dropout = nn.Dropout(dropout)`; no treino real de LLMs usa-se uma taxa baixa (0,1–0,2), bem menor que os 50% do exemplo didático.
- **Exemplo conceitual:** `torch.nn.Dropout(0.5)` aplicado a uma matriz de 1s produz aproximadamente metade dos valores zerados e a outra metade escalada para 2.0.

---

### 14. Multi-Head Attention
- **Tradução:** Atenção multi-cabeça
- **Definição:** Extensão do self-attention/causal attention em que o mecanismo é executado várias vezes em paralelo ("cabeças"), cada uma com suas próprias projeções Wq, Wk, Wv, e os resultados são combinados.
- **Funcionalidade:** Permite ao modelo atender simultaneamente a diferentes aspectos/subespaços de representação da entrada, em vez de se limitar a um único padrão de atenção.
- **Relação com outros conceitos:** É a última das quatro variantes construídas incrementalmente no capítulo (simplificado → treinável → causal → multi-head); é o componente usado diretamente nos blocos Transformer da Sprint 4. Implementada em `MultiHeadAttentionWrapper`/`MultiHeadAttention`.
- **Exemplo conceitual:** Com `num_heads=2` e `d_out=2` por cabeça, a saída final tem 4 dimensões (`d_out × num_heads`), combinando dois conjuntos de vetores de contexto Z1 e Z2.

---

### 15. Head (Cabeça de Atenção) e Head Dimension
- **Tradução:** Cabeça de atenção / dimensão da cabeça
- **Definição:** Cada "cabeça" é uma instância independente do mecanismo de atenção; `head_dim = d_out / num_heads` é a dimensão do vetor de contexto produzido por cada cabeça antes da combinação final.
- **Funcionalidade:** Múltiplas cabeças com dimensão reduzida (em vez de uma única cabeça com dimensão completa) permitem capturar padrões de atenção diferentes a um custo computacional comparável.
- **Relação com outros conceitos:** Na `MultiHeadAttention`, `d_out` precisa ser divisível por `num_heads` (`assert d_out % num_heads == 0`); nos modelos GPT-2, o menor tem 12 cabeças com embedding de 768 dimensões, o maior tem 25 cabeças com 1.600 dimensões.
- **Exemplo conceitual:** `MultiHeadAttention(d_in=3, d_out=2, ..., num_heads=2)` gera `head_dim = 1`: cada cabeça processa vetores de contexto de 1 dimensão antes de serem recombinados em 2.

---

### 16. MultiHeadAttentionWrapper (Empilhamento de Cabeças)
- **Tradução:** Wrapper de atenção multi-cabeça
- **Definição:** Implementação inicial e mais intuitiva de multi-head attention, que instancia múltiplos objetos `CausalAttention` independentes e concatena suas saídas ao longo da última dimensão.
- **Funcionalidade:** Cada cabeça processa a sequência inteira de forma sequencial (via list comprehension), com sua própria matriz de pesos — conceitualmente simples, mas computacionalmente repetitivo.
- **Relação com outros conceitos:** É substituído, por eficiência, pela classe `MultiHeadAttention`, que faz a mesma coisa com apenas uma projeção linear e um split de tensor.
- **Exemplo conceitual:** `torch.cat([head(x) for head in self.heads], dim=-1)` concatena as saídas de duas `CausalAttention`, produzindo `context_vecs.shape == torch.Size([2, 6, 4])` a partir de duas cabeças de saída 2.

---

### 17. MultiHeadAttention com Weight Splits (Implementação Eficiente)
- **Tradução:** Atenção multi-cabeça com divisão de pesos
- **Definição:** Implementação que integra todas as cabeças em uma única classe, projetando queries/keys/values uma única vez com matrizes maiores e depois "dividindo" (`.view`) e transpondo (`.transpose`) os tensores para separar as cabeças.
- **Funcionalidade:** Evita repetir a multiplicação de matrizes para cada cabeça (o passo computacionalmente mais caro); usa multiplicação de matrizes em lote (batched matrix multiplication) para calcular todas as cabeças de uma vez.
- **Relação com outros conceitos:** Reshape `(b, num_tokens, d_out)` → `(b, num_tokens, num_heads, head_dim)` → transpõe para `(b, num_heads, num_tokens, head_dim)`; ao final, transpõe de volta e achata para `(b, num_tokens, d_out)`. É a classe efetivamente usada na LLM do Capítulo 4.
- **Exemplo conceitual:** `MultiHeadAttention(d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2)` aplicada a um batch de forma `(2, 6, 3)` devolve vetores de contexto de forma `(2, 6, 2)`.

---

### 18. Output Projection Layer (`out_proj`)
- **Tradução:** Camada de projeção de saída
- **Definição:** Camada linear adicional (`nn.Linear(d_out, d_out)`) aplicada após a combinação das cabeças em `MultiHeadAttention`.
- **Funcionalidade:** Mistura as informações vindas de diferentes cabeças antes de passar adiante; não é estritamente necessária, mas é comum em arquiteturas de LLM.
- **Relação com outros conceitos:** Presente na `MultiHeadAttention`, mas ausente na `CausalAttention` de cabeça única — é uma diferença estrutural entre as duas classes.
- **Exemplo conceitual:** `context_vec = self.out_proj(context_vec)` é a última operação antes de retornar o vetor de contexto final da `MultiHeadAttention`.

---

### 19. Batched Matrix Multiplication (Multiplicação de Matrizes em Lote)
- **Tradução:** Multiplicação de matrizes em lote
- **Definição:** Operação em que o PyTorch aplica a multiplicação de matrizes às duas últimas dimensões de tensores de 4 dimensões (ou mais), repetindo automaticamente para cada "lote" das dimensões anteriores (batch e heads).
- **Funcionalidade:** Permite calcular os scores de atenção de todas as cabeças e de todas as amostras do batch em uma única operação vetorizada, sem loops explícitos.
- **Relação com outros conceitos:** Usada em `attn_scores = queries @ keys.transpose(2, 3)` dentro de `MultiHeadAttention`, onde queries/keys têm forma `(b, num_heads, num_tokens, head_dim)`.
- **Exemplo conceitual:** Para um tensor de forma `(1, 2, 3, 4)`, `a @ a.transpose(2, 3)` calcula, em uma única chamada, o produto interno de cada uma das 2 cabeças separadamente — equivalente a fazer `first_head @ first_head.T` e `second_head @ second_head.T` manualmente.

---

### 20. register_buffer
- **Tradução:** Registro de buffer (não treinável)
- **Definição:** Método do PyTorch (`nn.Module.register_buffer`) que registra um tensor como parte do estado do módulo sem torná-lo um parâmetro treinável.
- **Funcionalidade:** Garante que a máscara causal seja movida automaticamente entre CPU e GPU junto com o restante do modelo, evitando erros de dispositivo incompatível, sem participar do treino via backpropagation.
- **Relação com outros conceitos:** Usado em `CausalAttention` e `MultiHeadAttention` para armazenar a máscara triangular (`self.mask`), que é fixa e não aprendida.
- **Exemplo conceitual:** `self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))` cria a máscara uma única vez na inicialização, reutilizada em cada chamada de `forward`.

---

### 21. Weight Parameters vs. Attention Weights (Distinção)
- **Tradução:** Parâmetros de peso vs. pesos de atenção
- **Definição:** "Weight" nas matrizes Wq, Wk, Wv refere-se aos parâmetros aprendidos da rede (fixos após o treino); "attention weight" é o valor dinâmico e específico ao contexto que determina a importância relativa de cada token.
- **Funcionalidade:** Evita a confusão comum entre os dois usos da palavra "peso" no capítulo — um é estrutural (parâmetro), o outro é uma saída calculada a cada chamada do forward.
- **Relação com outros conceitos:** Complementa o termo "Parâmetros" já definido no Capítulo 1; os pesos de atenção dependem da entrada atual, enquanto os parâmetros (Wq, Wk, Wv) só mudam durante o treino.
- **Exemplo conceitual:** A mesma matriz `W_query` (parâmetro fixo) gera pesos de atenção diferentes para "journey" e para "step", pois os pesos de atenção dependem do token de consulta específico, não apenas da matriz.

---

# Glossário — Capítulo 4: Implementing a GPT Model from Scratch to Generate Text

---

### 1. DummyGPTModel (Arquitetura Placeholder)
- **Tradução:** Modelo GPT provisório / placeholder
- **Definição:** Esqueleto inicial da arquitetura GPT, em que o Transformer Block e o LayerNorm são substituídos por classes "vazias" (`DummyTransformerBlock` devolve a entrada sem alterá-la; `DummyLayerNorm` idem), só para fixar o formato de entrada e saída do modelo antes de implementar cada peça de verdade.
- **Funcionalidade:** Permite rodar um forward completo (Token IDs → logits) e conferir as formas dos tensores em cada etapa, mesmo sem nenhum componente interno funcional — uma estratégia de desenvolvimento incremental "de fora para dentro".
- **Relação com outros conceitos:** É substituído, passo a passo, pelos componentes reais do capítulo (LayerNorm, GELU, FeedForward, Shortcut Connections, Transformer Block), até virar o `GPTModel` funcional da seção 4.6.
- **Exemplo conceitual:** Com o `GPT_CONFIG_124M`, o `DummyGPTModel` já devolve logits de forma `(2, 4, 50257)` para um lote de 2 textos com 4 tokens — a forma certa, mesmo com os blocos internos sem fazer nada de útil ainda.

---

### 2. Layer Normalization (LayerNorm)
- **Tradução:** Normalização de camada
- **Definição:** Técnica que reescala as ativações de uma camada, ao longo da última dimensão (`emb_dim`), para terem média 0 e variância 1, seguida de uma escala (`scale`) e um deslocamento (`shift`) aprendidos.
- **Funcionalidade:** Estabiliza e acelera o treinamento, evitando que as ativações cresçam ou encolham demais à medida que atravessam muitas camadas — um dos problemas centrais no treino de redes profundas como os LLMs.
- **Relação com outros conceitos:** Diferente do Batch Normalization (que normaliza entre as amostras de um lote), o LayerNorm normaliza cada token independentemente, o que o torna adequado a lotes de tamanho variável; é aplicado antes da atenção e antes da FeedForward em cada Transformer Block (Pre-LayerNorm), e mais uma vez ao final de toda a pilha de blocos.
- **Exemplo conceitual:** Em `src/transformer.py`, `LayerNorm(emb_dim)` usa `unbiased=False` no cálculo da variância (divide por n, não por n−1) — a mesma convenção do TensorFlow usada no GPT-2 original, necessária para compatibilidade com pesos pré-treinados.

---

### 3. Vanishing Gradient Problem (Problema do Gradiente Desaparecendo)
- **Tradução:** Problema do gradiente que desaparece
- **Definição:** Fenômeno em que os gradientes ficam progressivamente menores à medida que se propagam para trás através das camadas durante a retropropagação, dificultando o ajuste dos pesos das camadas mais iniciais da rede.
- **Funcionalidade (por que importa):** Em redes profundas sem nenhum mecanismo de correção, as primeiras camadas praticamente param de aprender, pois o sinal de erro que chega a elas é quase nulo.
- **Relação com outros conceitos:** É o problema que motiva as conexões residuais (shortcut connections); o livro demonstra isso numericamente com uma rede de 5 camadas, comparando o gradiente médio de cada camada com e sem shortcuts.
- **Exemplo conceitual:** Sem shortcuts, o gradiente médio da primeira camada (`layers.0`) cai para 0,0002, quase 15 mil vezes menor que o da última camada (3,25); com shortcuts, a primeira camada mantém gradiente de 0,22 — a mesma ordem de grandeza da última.

---

### 4. GELU (Gaussian Error Linear Unit)
- **Tradução:** Unidade linear de erro gaussiano
- **Definição:** Função de ativação suave usada pelos modelos GPT no lugar da ReLU, definida (na versão exata) como `GELU(x) = x · Φ(x)`, onde `Φ` é a função de distribuição acumulada da gaussiana padrão; na prática, usa-se uma aproximação por `tanh`, mais barata de calcular.
- **Funcionalidade:** Ao contrário da ReLU, que zera abruptamente tudo que é negativo (derivada zero permanente, o problema dos "neurônios mortos"), a GELU deixa passar uma fração pequena, mas não nula, dos valores negativos, numa transição suave — o que favorece a otimização em redes profundas.
- **Relação com outros conceitos:** É a ativação usada dentro da `FeedForward`, entre as duas camadas lineares; o GPT-2 original foi treinado com exatamente essa aproximação por `tanh`, por isso o livro (e `src/transformer.py`) a reproduz em vez de usar a GELU exata.
- **Exemplo conceitual:** No `exp13`, `GELU(-1.0) ≈ -0,159` enquanto `ReLU(-1.0) = 0` — a GELU "deixa vazar" parte do sinal negativo; para `x=3.0`, as duas já ficam praticamente iguais (2,996 vs. 3,000).

---

### 5. FeedForward Network (Rede Feed-Forward)
- **Tradução:** Rede de alimentação direta (posição a posição)
- **Definição:** Pequena rede de duas camadas lineares com uma GELU no meio, aplicada independentemente a cada posição (token) da sequência: `emb_dim → 4×emb_dim → GELU → emb_dim`.
- **Funcionalidade:** Enquanto a atenção mistura informação ENTRE tokens, a FeedForward processa cada token individualmente, dando à rede "espaço" extra (a expansão de 4x) para recombinar, token a token, a informação que a atenção trouxe de outras posições.
- **Relação com outros conceitos:** É a segunda das duas sub-camadas de um Transformer Block (a primeira é a Multi-Head Attention); a uniformidade entre a forma de entrada e saída é o que permite empilhar blocos sem ajustes de dimensão.
- **Exemplo conceitual:** Para `emb_dim=768` (GPT-2 small), a primeira camada linear projeta para 3.072 dimensões e a segunda projeta de volta para 768 — no `exp13`, com `emb_dim=8`, isso vira 8 → 32 → 8, com 552 parâmetros no total.

---

### 6. Shortcut Connections (Conexões Residuais)
- **Tradução:** Conexões de atalho / conexões residuais
- **Definição:** Conexões que somam a entrada de uma camada (ou sub-bloco) diretamente à sua saída, criando um "atalho" por onde o gradiente pode fluir sem precisar atravessar as transformações internas daquela camada.
- **Funcionalidade:** Mitigam o problema do gradiente desaparecendo em redes profundas, garantindo que mesmo camadas muito distantes da saída recebam um sinal de erro significativo durante o treino.
- **Relação com outros conceitos:** Usadas duas vezes em cada Transformer Block — ao redor da atenção e ao redor da FeedForward; originalmente propostas para redes residuais de visão computacional, hoje são um componente padrão de LLMs.
- **Exemplo conceitual:** No `exp13` (13.5), zerando todos os pesos da atenção, da FeedForward e do `scale` do LayerNorm dentro de um `TransformerBlock`, a saída continua sendo exatamente igual à entrada — prova de que o atalho, sozinho, já garante a passagem do sinal mesmo quando as sub-camadas "não fazem nada".

---

### 7. Transformer Block
- **Tradução:** Bloco Transformer
- **Definição:** Unidade reutilizável que integra Multi-Head Attention, Layer Normalization, FeedForward e conexões residuais: `x → LayerNorm → Atenção → dropout → +x`, seguido de `→ LayerNorm → FeedForward → dropout → +resultado`.
- **Funcionalidade:** É o "tijolo" que se repete N vezes para formar a arquitetura GPT; entrada e saída têm sempre a mesma forma `(batch, tokens, emb_dim)`, o que permite empilhar quantos blocos forem necessários sem nenhum ajuste de dimensão entre eles.
- **Relação com outros conceitos:** Repetido 12 vezes no GPT-2 small (124M), 24 no medium, 36 no large e 48 no XL; implementado em `TransformerBlock` (`src/transformer.py`), que também repassa `ultimo_attn_weights` da sua `MultiHeadAttention` interna, no mesmo contrato da Sprint 3.
- **Exemplo conceitual:** `exp13` instancia um bloco com `emb_dim=8`, `num_heads=2` e mede 848 parâmetros totais: 264 na atenção, 552 na FeedForward e 32 nos dois LayerNorms.

---

### 8. Pre-LayerNorm vs. Post-LayerNorm
- **Tradução:** Normalização antes vs. depois da sub-camada
- **Definição:** Duas formas de posicionar o LayerNorm em relação à atenção/FeedForward: Pre-LayerNorm normaliza a entrada ANTES de passá-la pela sub-camada (`x → norm → atenção → +x`); Post-LayerNorm normaliza DEPOIS (`x → atenção → norm → +x`), como no Transformer original de 2017.
- **Funcionalidade:** O Pre-LayerNorm, adotado pelo GPT-2 e pelos LLMs modernos, produz uma dinâmica de treino mais estável que o Post-LayerNorm, especialmente em redes muito profundas.
- **Relação com outros conceitos:** É a convenção seguida por `TransformerBlock`, tanto no livro quanto em `src/transformer.py`: o `LayerNorm` sempre vem antes da `MultiHeadAttention`/`FeedForward`, nunca depois.
- **Exemplo conceitual:** `self.norm1(x)` é chamado antes de `self.atencao(x)` em `TransformerBlock.forward` — se a ordem fosse invertida (atenção primeiro, norm depois), seria Post-LayerNorm.

---

### 9. GPTModel (Arquitetura GPT Completa)
- **Tradução:** Modelo GPT
- **Definição:** Arquitetura completa que encadeia `GPTEmbedding` (token + positional embeddings) → dropout → N × `TransformerBlock` → LayerNorm final → `out_head` (camada linear sem viés), convertendo Token IDs em logits sobre o vocabulário.
- **Funcionalidade:** É o modelo que efetivamente aprende a prever o próximo token; todos os componentes dos capítulos 2 a 4 (tokenização, embeddings, atenção, normalização, FeedForward, shortcuts) se encontram aqui pela primeira vez em uma única classe.
- **Relação com outros conceitos:** Implementado em `GPTModel` (`src/transformer.py`); os valores padrão da classe reproduzem o `GPT_CONFIG_124M` do livro (768 dimensões, 12 cabeças, 12 camadas).
- **Exemplo conceitual:** `exp14` confirma que `modelo.forward(idx)` produz exatamente os mesmos logits que passar manualmente por `embeddings → cada bloco → norm_final → out_head` — a classe só organiza essas etapas, sem nenhuma lógica escondida.

---

### 10. Weight Tying (Amarração de Pesos)
- **Tradução:** Amarração (compartilhamento) de pesos
- **Definição:** Técnica em que a matriz de token embeddings (`tok_emb`, forma `vocab_size × emb_dim`) e a matriz da camada de saída (`out_head`, também `vocab_size × emb_dim`) são a MESMA matriz, reaproveitada nas duas pontas do modelo.
- **Funcionalidade:** Reduz a memória e o número de parâmetros do modelo, já que essas duas matrizes são, de longe, as maiores da arquitetura quando o vocabulário é grande (50.257 no GPT-2).
- **Relação com outros conceitos:** É exatamente a diferença entre os "124M" anunciados originalmente para o GPT-2 small e os 163M parâmetros reais de uma implementação sem tying, como a do livro e a deste projeto (que usa camadas separadas, por produzir, segundo o autor, melhores resultados de treino).
- **Exemplo conceitual:** `exp14` (14.5) calcula exatamente essa diferença: 163.009.536 parâmetros sem tying, menos 38.597.376 (o tamanho de `tok_emb`, igual ao de `out_head`), resultando em 124.412.160 — batendo com os números do livro.

---

### 11. GPT_CONFIG_124M (Dicionário de Configuração)
- **Tradução:** Configuração do GPT-2 small
- **Definição:** No livro, um dicionário Python que reúne os hiperparâmetros estruturais do menor modelo GPT-2: `vocab_size=50257`, `context_length=1024`, `emb_dim=768`, `n_heads=12`, `n_layers=12`, `drop_rate=0.1`, `qkv_bias=False`.
- **Funcionalidade:** Centraliza, em um único lugar, tudo que define o "tamanho" e a "forma" do modelo — o mesmo `GPTModel`/classe serve para qualquer uma das quatro variantes oficiais do GPT-2, bastando trocar esses valores.
- **Relação com outros conceitos:** Em `src/transformer.py`, os parâmetros equivalentes viraram argumentos nomeados do construtor de `GPTModel` (com os mesmos valores como padrão), em vez de um dicionário — mas o papel é idêntico.
- **Exemplo conceitual:** `exp14` (14.4) usa as quatro configurações oficiais (small/medium/large/XL) só variando `emb_dim`, `n_layers` e `n_heads`, confirmando que a mesma classe escala de 163M a mais de 1,6 bilhão de parâmetros.

---

### 12. Logits
- **Tradução:** Logits (mantido em inglês)
- **Definição:** Saída bruta e não normalizada da última camada do modelo (`out_head`), de forma `(batch, tokens, vocab_size)` — um vetor de 50.257 números reais por token, um por palavra possível do vocabulário.
- **Funcionalidade:** Só a posição de maior valor já indica, sozinha, qual token o modelo prefere (já que o softmax é monotônico); os logits precisam passar por softmax para virarem uma distribuição de probabilidade válida (soma 1, todos ≥ 0).
- **Relação com outros conceitos:** É a ponte entre o `GPTModel` (que produz logits) e `src/geracao.py` (que os transforma em texto); só a última posição da sequência interessa para prever o próximo token.
- **Exemplo conceitual:** No `exp16` (16.3), `logits[:, -1, :]` tem forma `(1, vocab_size)`; aplicando softmax, a maior probabilidade e a posição do `argmax` são idênticas antes e depois da normalização.

---

### 13. Language Model Head (`out_head`)
- **Tradução:** Cabeça de modelo de linguagem
- **Definição:** Camada linear final do `GPTModel` (`nn.Linear(emb_dim, vocab_size, bias=False)`) que projeta a representação de cada token, após o LayerNorm final, para o espaço do vocabulário.
- **Funcionalidade:** É a camada que converte "entendimento contextual" (um vetor de `emb_dim` dimensões) em uma "opinião sobre qual é o próximo token" (um vetor de `vocab_size` dimensões) — sem viés, seguindo a convenção do GPT-2.
- **Relação com outros conceitos:** Tem a mesma forma da matriz de token embeddings (`vocab_size × emb_dim`), o que torna o weight tying possível; domina, junto com `tok_emb`, o total de parâmetros em modelos com vocabulário grande e poucas camadas.
- **Exemplo conceitual:** `parametros_por_componente` (`src/transformer.py`) reporta, para um `GPTModel` pequeno de teste, que `out_head` e `token_embeddings` têm exatamente o mesmo número de parâmetros (800, no `exp14`) — prova direta de que as formas são iguais.

---

### 14. Geração Autorregressiva (generate / `gerar`)
- **Tradução:** Geração autorregressiva de texto
- **Definição:** Processo iterativo em que o modelo prevê um token por vez, acrescenta o token escolhido ao final do contexto e repete, usando a sequência (agora maior) como entrada da próxima rodada.
- **Funcionalidade:** É assim que um LLM "escreve": nunca produz a frase inteira de uma vez, mas token a token, cada novo token condicionado em tudo o que já foi gerado — o mesmo princípio da máscara causal da Sprint 3, agora aplicado à inferência.
- **Relação com outros conceitos:** Implementado em `gerar` (`src/geracao.py`), que generaliza o `generate_text_simple` do livro (o livro só cobre greedy; `gerar` acrescenta temperatura e top-k); a cada passo, só os logits da ÚLTIMA posição são usados.
- **Exemplo conceitual:** `exp16` (16.1) mostra, passo a passo, o contexto crescendo a cada iteração — começando em `[50, 1807, ...]` (prompt) e ganhando um Token ID novo por rodada, até completar `max_novos_tokens`.

---

### 15. Context Size Cropping (Recorte do Contexto)
- **Tradução:** Recorte do tamanho de contexto
- **Definição:** Etapa do laço de geração em que a sequência acumulada é recortada para os últimos `context_size` tokens (`idx[:, -context_size:]`) antes de ser enviada ao modelo.
- **Funcionalidade:** Garante que o modelo nunca receba mais tokens do que sua janela de contexto suporta — a cada novo token gerado, a sequência cresce, mas o que é efetivamente "visto" pelo modelo permanece limitado a `context_size`.
- **Relação com outros conceitos:** É necessário porque a camada de positional embeddings (Sprint 2) só tem vetores para `context_length` posições; sem o recorte, uma geração longa o suficiente provocaria um erro de índice fora do intervalo.
- **Exemplo conceitual:** `exp16` (16.7) usa `context_size=8` com um prompt de mais de 8 tokens — a sequência total cresce a cada rodada, mas o modelo sempre recebe só os últimos 8 tokens como entrada.

---

### 16. Dropout nas Múltiplas Camadas do GPT
- **Tradução:** Dropout distribuído pela arquitetura
- **Definição:** No `GPTModel`, o dropout aparece em três lugares distintos: logo após somar token e positional embeddings (`drop_emb`), depois da atenção dentro de cada bloco e depois da FeedForward dentro de cada bloco (`drop_shortcut`).
- **Funcionalidade:** Cada ponto de dropout reduz o overfitting de uma parte diferente do modelo: nos embeddings, na mistura de informação entre tokens (atenção) e na transformação individual de cada token (FeedForward).
- **Relação com outros conceitos:** Generaliza o dropout já visto na `CausalAttention`/`MultiHeadAttention` da Sprint 3 (que cobria só os pesos de atenção) para o modelo inteiro; todos usam a mesma taxa (`drop_rate`) por simplicidade, embora o livro proponha, como exercício, usar taxas diferentes em cada ponto.
- **Exemplo conceitual:** `exp13` (13.7) mede a variância da saída de um `TransformerBlock` entre chamadas repetidas em modo de treino: 0 com `drop_rate=0%`, crescendo para 0,238 com `drop_rate=50%` — prova de que o dropout torna a saída estocástica apenas durante o treino.

---

### 17. Temperatura (Sampling)
- **Tradução:** Temperatura de amostragem
- **Definição:** Fator que divide os logits antes do softmax (`logits / temperatura`); valores menores que 1 tornam a distribuição mais "afiada" (mais parecida com greedy), valores maiores que 1 a achatam, dando mais chance a tokens menos prováveis.
- **Funcionalidade:** Controla a aleatoriedade da geração sem mudar a ORDEM de preferência do modelo entre os candidatos — só a distância relativa entre eles antes de sortear.
- **Relação com outros conceitos:** Implementada em `aplicar_temperatura` (`src/geracao.py`); combinada com `torch.multinomial` (sorteio proporcional às probabilidades) em vez de `argmax` (que ignoraria completamente a temperatura).
- **Exemplo conceitual:** `exp17` mede a entropia da distribuição do próximo token para temperaturas diferentes, mostrando que a entropia cresce com a temperatura — mais tokens "no páreo", maior a incerteza da escolha.

---

### 18. Top-k Sampling
- **Tradução:** Amostragem pelos k maiores
- **Definição:** Técnica que mantém apenas os `k` tokens com maior logit, atribuindo `-infinito` a todos os demais antes do softmax — eliminando, assim, candidatos muito improváveis do sorteio.
- **Funcionalidade:** Evita que a amostragem, especialmente com temperatura alta, escolha por azar um token muito improvável e descarrile a coerência do texto gerado.
- **Relação com outros conceitos:** Implementado em `aplicar_top_k` (`src/geracao.py`); aplicado ANTES da temperatura em `distribuicao_do_proximo_token` — cortar primeiro e só depois reescalar, nunca o contrário, para que o corte reflita a distribuição original do modelo.
- **Exemplo conceitual:** Com `top_k=5`, apenas os 5 tokens mais prováveis de cada passo concorrem entre si; no `exp16` (16.2), os "cinco candidatos de cada passo" são exatamente essa vizinhança — o mesmo grupo que o top-k manteria.

---

### 19. Greedy Decoding (Decodificação Gulosa)
- **Tradução:** Decodificação gulosa
- **Definição:** Estratégia de geração em que, a cada passo, o token escolhido é sempre o de maior probabilidade (`argmax` dos logits) — sem nenhum sorteio envolvido.
- **Funcionalidade:** É determinística: a mesma entrada, com o mesmo modelo, produz sempre exatamente a mesma saída — útil para depuração e para comparar configurações de forma reprodutível.
- **Relação com outros conceitos:** Corresponde a `temperatura=0` em `gerar` (`src/geracao.py`), ou à função dedicada `gerar_texto_greedy`; é também o único modo implementado no `generate_text_simple` do livro (seção 4.7) — temperatura e top-k são extensões além do que o capítulo cobre.
- **Exemplo conceitual:** `exp16` (16.4) roda a mesma geração greedy três vezes seguidas e confirma que as três saídas são idênticas — diferente do que aconteceria com `temperatura > 0`.

---

### 20. Número de Parâmetros e Memória de um LLM
- **Tradução:** Contagem de parâmetros e memória do modelo
- **Definição:** O total de parâmetros treináveis de um `GPTModel` pode ser decomposto por componente (embeddings, blocos Transformer, LayerNorm final, `out_head`) e convertido em memória estimada (parâmetros × 4 bytes, assumindo float32).
- **Funcionalidade:** Permite projetar o custo de configurações maiores sem precisar instanciá-las na memória — basta uma fórmula fechada, validada contra um modelo pequeno de verdade.
- **Relação com outros conceitos:** `parametros_por_componente` e `memoria_em_mb` (`src/transformer.py`) implementam essa análise, reaproveitando `numero_de_parametros` já definido na Sprint 3; o livro faz a mesma conta manualmente para o GPT-2 small (163.009.536 parâmetros, 621,83 MB).
- **Exemplo conceitual:** `exp14` (14.3) valida uma fórmula fechada de contagem contra um `GPTModel` pequeno instanciado (bateram exatamente) e depois a usa, sem instanciar nada, para projetar o GPT-2 XL: 1.637.792.000 parâmetros, cerca de 6,1 GB em float32.

---

# Glossário — Capítulo 5: Pretraining on Unlabeled Data

---

### 1. text_to_token_ids / token_ids_to_text
- **Tradução:** Conversão entre texto e Token IDs
- **Definição:** Par de funções utilitárias (seção 5.1.1) que encapsulam, respectivamente, a tokenização de um texto em um tensor `(1, num_tokens)` pronto para o modelo (`unsqueeze(0)` acrescenta a dimensão de lote) e o caminho inverso, removendo a dimensão de lote e decodificando de volta para texto.
- **Funcionalidade:** Evitam repetir, em todo lugar do capítulo, a mesma sequência de três chamadas (`tokenizer.encode` → `torch.tensor` → `unsqueeze`); formalizam o "contrato" de forma que todas as demais funções do capítulo (geração, avaliação, treino) passam a assumir.
- **Relação com outros conceitos:** É exatamente o par `texto_para_ids`/`ids_para_texto` já implementado em `src/geracao.py` na Sprint 4 — o livro só os apresenta formalmente agora, no Capítulo 5, embora a necessidade já existisse desde a geração de texto do Capítulo 4.
- **Exemplo conceitual:** `text_to_token_ids("Every effort moves you", tokenizer)` produz um tensor de forma `(1, 4)`; `token_ids_to_text` desfaz essa transformação, retornando a string original (ou o texto gerado pelo modelo).

---

### 2. Text Generation Loss (Perda de Geração de Texto)
- **Tradução:** Perda de geração de texto
- **Definição:** Medida numérica de "quão longe" os tokens gerados por um LLM estão dos tokens-alvo corretos — o conceito geral que a cross-entropy loss formaliza matematicamente.
- **Funcionalidade:** Antes do Capítulo 5, só existia uma forma qualitativa de avaliar o modelo (ler o texto gerado e julgar se "parece bom"); a perda de geração introduz uma métrica numérica, necessária tanto para acompanhar o progresso quanto para servir de sinal de treino via backpropagation.
- **Relação com outros conceitos:** É calculada a partir das probabilidades que o softmax atribui aos tokens-alvo (ver "Probabilidades do Token-Alvo"); sem ela, não haveria nada para `loss.backward()` diferenciar.
- **Exemplo conceitual:** Um modelo ainda não treinado gera "Every effort moves you rentingetic wasn? refres RexMeCHicular stren" para o contexto "Every effort moves you" — claramente incoerente, mas só a perda numérica permite comparar objetivamente esse resultado com o de outra configuração.

---

### 3. Probabilidades do Token-Alvo (Target Token Probabilities)
- **Tradução:** Probabilidades do token-alvo
- **Definição:** Para cada posição de uma sequência, a probabilidade (após softmax) que o modelo atribuiu especificamente ao token que deveria ter sido gerado ali — não ao token de maior probabilidade, mas ao token correto.
- **Funcionalidade:** É o primeiro ingrediente do cálculo da perda: o objetivo do treino é aumentar exatamente essas probabilidades, uma de cada vez, a cada posição de cada exemplo.
- **Relação com outros conceitos:** Extraídas via indexação avançada (`probas[text_idx, [0, 1, 2], targets[text_idx]]`); alimentam diretamente o cálculo da log-probabilidade média negativa (cross-entropy).
- **Exemplo conceitual:** Para a frase-alvo " effort moves you", um modelo não treinado atribui probabilidades como `[7.45e-05, 3.11e-05, 1.16e-05]` a cada token correto — próximas de zero, já que o vocabulário tem 50.257 entradas e os pesos ainda são aleatórios.

---

### 4. Log-Probabilidade Média Negativa (Negative Average Log-Probability)
- **Tradução:** Log-probabilidade média negativa
- **Definição:** Resultado de aplicar logaritmo às probabilidades do token-alvo, calcular a média desses logs e inverter o sinal — o procedimento manual, passo a passo, que a cross-entropy loss do PyTorch executa internamente.
- **Funcionalidade:** Trabalhar com logaritmos de probabilidades é mais tratável matematicamente em otimização do que as probabilidades brutas (produtos de números muito pequenos tendem a *underflow* numérico; somas de logs, não); inverter o sinal transforma um valor que deveria subir até 0 (a probabilidade log, sempre ≤ 0) em um valor que deve descer até 0 — a convenção usual de uma função de perda a ser minimizada.
- **Relação com outros conceitos:** É numericamente idêntica à cross-entropy loss — "cross entropy" e "log-probabilidade média negativa" são termos usados de forma intercambiável na prática, segundo o próprio livro.
- **Exemplo conceitual:** Para seis probabilidades-alvo (duas sequências de três tokens), os logs são `[-9.50, -10.38, -11.37, -11.48, -9.78, -12.26]`; a média é `-10.7940`; invertendo o sinal, a perda é `10.7940` — exatamente o valor que `cross_entropy` devolve para a mesma entrada.

---

### 5. Cross-Entropy Loss
- **Tradução:** Perda de entropia cruzada
- **Definição:** Medida da diferença entre duas distribuições de probabilidade — aqui, a distribuição "verdadeira" (um token-alvo específico, com probabilidade 1) e a distribuição prevista pelo modelo (os 50.257 valores de probabilidade do softmax).
- **Funcionalidade:** É a função de perda padrão para classificação multi-classe, e um LLM prevendo o próximo token é, posição a posição, exatamente uma classificação multi-classe entre as `vocab_size` palavras possíveis; `torch.nn.functional.cross_entropy` calcula, em uma única chamada, os seis passos manuais (probabilidades → seleção do alvo → log → média → sinal trocado).
- **Relação com outros conceitos:** Exige achatar logits `(batch, tokens, vocab_size)` em `(batch·tokens, vocab_size)` e alvos `(batch, tokens)` em `(batch·tokens,)`, para que a função enxergue cada posição como um exemplo independente de classificação. Implementada em `calcular_perda_lote` (`src/treino.py`).
- **Exemplo conceitual:** `torch.nn.functional.cross_entropy(logits_flat, targets_flat)` sobre os mesmos dados do exemplo manual devolve `tensor(10.7940)` — idêntico ao cálculo passo a passo, confirmado também pelo `exp18` (18.4) deste projeto com valores próprios (perda manual = perda do PyTorch = `True`).

---

### 6. Perplexity
- **Tradução:** Perplexidade
- **Definição:** `e` elevado à perda (`torch.exp(perda)`) — uma transformação da cross-entropy loss para uma escala mais interpretável.
- **Funcionalidade:** Representa o tamanho efetivo do vocabulário sobre o qual o modelo está "em dúvida" a cada posição; uma perplexity de 48.725, por exemplo, significa que o modelo está tão incerto quanto se tivesse que escolher ao acaso entre 48.725 tokens igualmente prováveis.
- **Relação com outros conceitos:** Função direta da perda (quanto menor a perda, menor a perplexity); implementada em `perplexidade` (`src/treino.py`), usada ao lado de toda perda reportada nos experimentos deste projeto.
- **Exemplo conceitual:** O livro calcula perplexity de `48,725.82` para um modelo não treinado com perda `10.7940`; no `exp18` (18.6) deste projeto, a perda inicial de `10.98` sobre o corpus `amostra.txt` corresponde a uma perplexity de `58.623` — na mesma ordem de grandeza do vocabulário completo (50.257), como esperado de um modelo que ainda não aprendeu nada.

---

### 7. Divisão Treino/Validação (Training/Validation Split)
- **Tradução:** Divisão treino/validação
- **Definição:** Separação do corpus de treinamento em duas partes disjuntas — uma maior (tipicamente 90%), usada para ajustar os pesos do modelo, e uma menor (10%), reservada apenas para medir o desempenho em dados que o modelo nunca viu durante o treino.
- **Funcionalidade:** Sem essa separação, não haveria como distinguir um modelo que aprendeu um padrão geral de um modelo que apenas memorizou o texto de treino — a perda de validação é o termômetro dessa diferença.
- **Relação com outros conceitos:** Usa os mesmos `create_dataloader_v1`/`GPTDatasetV1` da Sprint 2, um para cada metade do texto; a Sprint 5 usa `drop_last=True` no loader de treino (evita lotes incompletos durante o treino) e `drop_last=False` no de validação (quer aproveitar todos os dados disponíveis para a medida).
- **Exemplo conceitual:** Com `train_ratio=0.90` sobre o corpus `amostra.txt` (7.180 caracteres), o `exp19` (19.1) produz 39 lotes de treino e 5 de validação, com `batch_size=2` e `context_length=32`.

---

### 8. calc_loss_batch / calc_loss_loader
- **Tradução:** Cálculo da perda de um lote / de um DataLoader inteiro
- **Definição:** Duas funções em camadas: `calc_loss_batch` calcula a cross-entropy de um único lote (chama o modelo, achata os tensores, aplica `cross_entropy`); `calc_loss_loader` itera sobre um `DataLoader` inteiro (ou só sobre os primeiros `num_batches`), acumulando e depois dividindo pelo número de lotes.
- **Funcionalidade:** Separar as duas responsabilidades permite reutilizar `calc_loss_batch` tanto dentro do loop de treino (onde o gradiente importa) quanto dentro de `evaluate_model` (onde ele é descartado); o parâmetro `num_batches` em `calc_loss_loader` evita o custo de varrer o conjunto de validação inteiro a cada avaliação durante o treino.
- **Relação com outros conceitos:** Implementadas como `calcular_perda_lote`/`calcular_perda_loader` (`src/treino.py`), usando exatamente a Cross-Entropy Loss definida acima.
- **Exemplo conceitual:** Sobre `amostra.txt`, com um `GPTModel` ainda não treinado, `calcular_perda_loader` devolve `10.9789` para o conjunto de treino e `11.0167` para a validação (`exp18`, 18.6) — próximos um do outro, como esperado antes de qualquer ajuste de pesos.

---

### 9. Loop de Treinamento (Training Loop)
- **Tradução:** Laço de treinamento
- **Definição:** Sequência de oito passos (Figura 5.11) que todo treino de rede neural profunda em PyTorch segue: iterar sobre épocas, iterar sobre lotes, zerar gradientes, calcular a perda, retropropagar, atualizar os pesos e, opcionalmente, registrar perdas e gerar amostras de texto.
- **Funcionalidade:** É o mecanismo que efetivamente transforma um `GPTModel` de pesos aleatórios em um modelo que prevê o próximo token com mais acerto — tudo o que vem antes (arquitetura, avaliação) só prepara o terreno para este loop.
- **Relação com outros conceitos:** Implementado em `treinar_modelo` (`src/treino.py`), que corresponde ao `train_model_simple` do livro; usa `calcular_perda_lote` no passo 4 e `avaliar_modelo`/`gerar_e_imprimir_amostra` nos passos opcionais de acompanhamento.
- **Exemplo conceitual:** O `exp19` roda esse loop por 10 épocas sobre `amostra.txt`, reduzindo a perda de treino de `10.962` para `5.264` e a de validação de `10.962` para `6.236` — a prova numérica de que o loop está, de fato, ajustando os pesos na direção certa.

---

### 10. Epoch (Época)
- **Tradução:** Época
- **Definição:** Uma passagem completa por todo o conjunto de treino — o laço externo do loop de treinamento.
- **Funcionalidade:** É a unidade em que o progresso do treino costuma ser medido e reportado (perda por época, amostra de texto por época); o número de lotes dentro de uma época é determinado pelo tamanho do conjunto de treino dividido pelo tamanho do lote.
- **Relação com outros conceitos:** Parâmetro `num_epocas` de `treinar_modelo`; mais épocas sobre o mesmo corpus pequeno tendem a aumentar o overfitting, como visto no `exp20`.
- **Exemplo conceitual:** Com 39 lotes de treino por época (`exp19`), 10 épocas correspondem a 390 passos de atualização de pesos ao todo.

---

### 11. zero_grad() (Reinício dos Gradientes)
- **Tradução:** Zerar os gradientes
- **Definição:** Chamada (`otimizador.zero_grad()`) que limpa os gradientes acumulados na iteração anterior antes de calcular os novos.
- **Funcionalidade:** Por padrão, o PyTorch ACUMULA gradientes a cada `.backward()` — útil para técnicas avançadas (acumulação de gradiente em lotes grandes), mas indesejável aqui, onde cada lote deve gerar uma atualização de peso independente. Sem `zero_grad()`, os gradientes de lotes sucessivos se somariam, distorcendo a direção da atualização.
- **Relação com outros conceitos:** Primeiro passo de cada iteração do loop de treinamento, antes de `calcular_perda_lote` e `loss.backward()`.
- **Exemplo conceitual:** `otimizador.zero_grad()` → `perda = calcular_perda_lote(...)` → `perda.backward()` → `otimizador.step()` é a sequência de quatro linhas que se repete a cada lote dentro de `treinar_modelo`.

---

### 12. optimizer.step() (Atualização dos Pesos)
- **Tradução:** Passo do otimizador / atualização dos pesos
- **Definição:** Chamada que efetivamente ajusta os parâmetros do modelo, na direção oposta ao gradiente calculado por `loss.backward()`, na magnitude determinada pela taxa de aprendizado.
- **Funcionalidade:** É o passo em que o "aprendizado" realmente acontece — tudo antes dele (forward, perda, backward) só calcula EM QUE DIREÇÃO mover os pesos; `step()` os move de fato.
- **Relação com outros conceitos:** Opera sobre todos os parâmetros registrados no otimizador (`modelo.parameters()`, passados na criação do `AdamW`); seu efeito acumulado ao longo de muitos passos é a própria curva de perda decrescente observada nos experimentos.
- **Exemplo conceitual:** Em 390 passos (`exp19`, 10 épocas × 39 lotes), cada chamada de `otimizador.step()` move os 6.634.240 parâmetros do modelo um pouco mais na direção que reduz a perda do lote atual.

---

### 13. AdamW
- **Tradução:** AdamW (mantido em inglês — nome do otimizador)
- **Definição:** Variante do otimizador Adam que separa o *weight decay* (penalização de pesos grandes, para reduzir a complexidade do modelo e evitar overfitting) da atualização baseada no gradiente, em vez de misturar os dois como o Adam tradicional.
- **Funcionalidade:** Essa separação produz uma regularização mais eficaz e melhor generalização — por isso é o otimizador mais usado no treinamento de LLMs atualmente, incluindo nesta implementação.
- **Relação com outros conceitos:** Criado via `torch.optim.AdamW(modelo.parameters(), lr=..., weight_decay=...)`; é o que efetivamente executa `optimizer.step()` a cada iteração do loop de treinamento.
- **Exemplo conceitual:** O livro treina seu modelo de 124M parâmetros com `lr=0.0004, weight_decay=0.1`; os experimentos deste projeto (`exp19`, `exp20`) usam os mesmos valores como ponto de partida, variando a taxa de aprendizado para observar seu efeito isoladamente.

---

### 14. Weight Decay (Decaimento de Peso)
- **Tradução:** Decaimento de peso
- **Definição:** Termo de regularização que penaliza pesos com magnitude grande, empurrando-os sutilmente em direção a zero a cada atualização, independentemente do gradiente da perda.
- **Funcionalidade:** Reduz a complexidade efetiva do modelo, desencorajando-o de depender fortemente de poucos pesos muito grandes — uma forma de combater o overfitting que atua durante a própria atualização dos pesos, não sobre os dados.
- **Relação com outros conceitos:** É o componente que o AdamW implementa de forma mais eficaz que o Adam tradicional; configurado pelo parâmetro `weight_decay` do otimizador (0.1 no exemplo do livro).
- **Exemplo conceitual:** `torch.optim.AdamW(modelo.parameters(), lr=0.0004, weight_decay=0.1)` aplica, a cada passo, tanto a atualização baseada no gradiente da perda quanto um pequeno "encolhimento" adicional de todos os pesos.

---

### 15. evaluate_model (Avaliação Durante o Treino)
- **Tradução:** Avaliação do modelo (durante o treino)
- **Definição:** Função que calcula a perda de treino e de validação com o modelo em modo `eval()` (dropout desligado) e `torch.no_grad()` (sem cálculo de gradiente), restaurando `train()` ao final.
- **Funcionalidade:** Sem desligar o dropout, a perda de validação oscilaria de uma medição para outra só por causa da aleatoriedade do dropout, mascarando o progresso real do treino; sem `no_grad()`, a avaliação gastaria memória e tempo calculando gradientes que não serão usados.
- **Relação com outros conceitos:** Implementada como `avaliar_modelo` (`src/treino.py`); chamada periodicamente (a cada `freq_avaliacao` passos) de dentro do loop de treinamento, usando `calc_loss_loader` sobre um número limitado de lotes (`eval_iter`/`lotes_avaliacao`) para não pesar o treino.
- **Exemplo conceitual:** No `exp20` (20.1), avaliar a cada época (em vez de a cada lote) permite acompanhar que a perda de validação para de cair por volta da época 8, enquanto a de treino continua caindo.

---

### 16. generate_and_print_sample (Acompanhamento Qualitativo)
- **Tradução:** Geração e impressão de amostra
- **Definição:** Função que, ao final de cada época, gera um trecho de texto a partir de um contexto inicial fixo, usando o modelo no estado atual do treino, para inspeção visual.
- **Funcionalidade:** A perda numérica mede o progresso, mas não é intuitiva de interpretar sozinha; ver o texto evoluir de "vírgulas repetidas" para frases gramaticalmente corretas é uma confirmação qualitativa complementar, mais acessível que qualquer número.
- **Relação com outros conceitos:** Implementada como `gerar_e_imprimir_amostra` (`src/treino.py`), que reaproveita `gerar_texto_greedy` (Sprint 4) em vez de reimplementar o laço de geração — alterna o modelo para `eval()` antes e `train()` depois, como `evaluate_model`.
- **Exemplo conceitual:** No `exp19` (19.4), a amostra da época 1 é um trecho degenerado de espaços repetidos; na época 10, já aparece alguma estrutura reconhecível de frase — evidência visual do mesmo progresso que a perda numérica documenta.

---

### 17. Overfitting Durante o Treinamento (Divergência Treino/Validação)
- **Tradução:** Sobreajuste durante o treinamento
- **Definição:** Situação em que a perda de treino continua caindo a cada época, mas a perda de validação estagna ou passa a subir — sinal de que o modelo está memorizando particularidades do conjunto de treino em vez de aprender um padrão que generaliza.
- **Funcionalidade:** É o principal risco ao treinar um modelo por muitas épocas sobre um corpus pequeno (como o deste projeto); o gap crescente entre as duas curvas (`perda_val - perda_treino`) é o sintoma numérico, e um trecho gerado que reaparece literalmente no corpus de treino é a confirmação qualitativa.
- **Relação com outros conceitos:** Estende o conceito de "Overfitting" já definido no Capítulo 2 (lá, ligado ao *stride* da janela deslizante) para o contexto do treino em si: agora a causa não é a amostragem dos dados, mas o número de vezes que o modelo revisita o mesmo texto pequeno.
- **Exemplo conceitual:** No `exp20` (20.1), o gap `perda_val - perda_treino` cresce de `-0.014` (época 1) para `+0.725` (época 8) de forma monotônica — a perda de validação deixa de acompanhar a de treino bem antes do fim do experimento, exatamente como o livro descreve para "The Verdict" (overfitting visível já a partir da época 2).

---

### 18. Custo de Pré-treinamento de LLMs (Cost of Pretraining)
- **Tradução:** Custo de pré-treinamento
- **Definição:** Estimativa do custo computacional (em horas de GPU e em dólares) de treinar um LLM em escala real, citada pelo livro para contextualizar por que este capítulo usa um modelo e um corpus tão pequenos.
- **Funcionalidade:** Justifica a escolha didática do livro (e deste projeto): treinar um modelo de 124M parâmetros sobre um conto curto leva minutos em um laptop; treinar um modelo real em escala de produção é um projeto de centenas de milhares de dólares.
- **Relação com outros conceitos:** Contextualiza por que a Sprint 5 usa um corpus pequeno (`amostra.txt`, 7.180 caracteres) e modelos reduzidos (`emb_dim` de 32 a 64 em vez de 768) — o objetivo é entender o MECANISMO do treino, não reproduzir sua escala real.
- **Exemplo conceitual:** O livro estima que treinar o Llama 2 de 7 bilhões de parâmetros (184.320 horas de GPU A100, 2 trilhões de tokens) custaria cerca de US$ 690.000 em nuvem — contra os poucos minutos e nenhum custo de nuvem que os experimentos desta Sprint exigem.

---

### 19. Salvamento e Carregamento de Pesos (state_dict)
- **Tradução:** Salvamento e carregamento de pesos / dicionário de estado
- **Definição:** O `state_dict` de um modelo PyTorch é um dicionário que mapeia cada camada aos seus parâmetros; `torch.save(modelo.state_dict(), caminho)` grava esse dicionário em disco, e `modelo.load_state_dict(torch.load(caminho))` o recarrega em uma nova instância do mesmo modelo.
- **Funcionalidade:** Permite preservar o resultado de um treino (que pode levar minutos, horas ou — em escala real — meses) sem precisar refazê-lo a cada nova sessão, e retomar o treino de onde parou ou simplesmente usar o modelo treinado para inferência depois.
- **Relação com outros conceitos:** É o Entregável 6 desta Sprint (parte da Laura); depende da arquitetura `GPTModel` já estar definida de forma idêntica entre o salvamento e o carregamento, já que o `state_dict` só guarda os números dos pesos, não a estrutura do modelo.
- **Exemplo conceitual:** `torch.save(modelo.state_dict(), "modelo.pth")` seguido, em uma sessão nova, de `modelo_novo = GPTModel(...); modelo_novo.load_state_dict(torch.load("modelo.pth"))` reproduz exatamente o mesmo modelo treinado, pronto para gerar texto sem precisar re-treinar.

---
