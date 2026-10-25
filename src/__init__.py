"""
Projeto LLM — Sprints 2 a 4.

Fluxo implementado:

    Texto -> Tokenização -> Tokens -> Token IDs -> Sequências (entrada, alvo)
          -> Token Embeddings -> Positional Embeddings -> Lote        (Sprint 2)
          -> Self-Attention -> Causal Attention -> Multi-Head Attention (Sprint 3)
          -> N x TransformerBlock -> LayerNorm final -> logits          (Sprint 4)
          -> geração autoregressiva de texto                            (Sprint 4)

Uso típico:

    from src import GPTModel, gerar_texto, obter_tokenizador_bpe

    tokenizador = obter_tokenizador_bpe()

    modelo = GPTModel(vocab_size=tokenizador.n_vocab, context_length=256,
                      emb_dim=768, n_layers=12, num_heads=12)

    logits = modelo(entradas)             # entradas: (8, 256) -> (8, 256, 50257)

    texto_gerado = gerar_texto(modelo, tokenizador, prompt="Era uma vez",
                               max_novos_tokens=20, context_size=256)

Requer: torch e tiktoken.
"""

from .attention import (
    CausalAttention,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    SelfAttention_v1,
    SelfAttention_v2,
    attention_scores_simples,
    context_vector_passo_a_passo,
    entropia_dos_pesos,
    numero_de_parametros,
    scaled_dot_product_attention,
    self_attention_simples,
)
from .bpe import BPETokenizerSimples, TiktokenTokenizer, obter_tokenizador_bpe
from .geracao import (
    aplicar_temperatura,
    aplicar_top_k,
    distribuicao_do_proximo_token,
    entropia_da_distribuicao,
    gerar,
    gerar_passo_a_passo,
    gerar_texto,
    gerar_texto_greedy,
    ids_para_texto,
    taxa_de_repeticao,
    texto_para_ids,
    tokens_para_cobrir,
)
from .dataset import (
    GPTDatasetV1,
    create_dataloader_v1,
    criar_pares_janela_deslizante,
    numero_de_amostras,
)
from .embeddings import GPTEmbedding, comparar_posicoes, embedding_de_token, tamanho_em_mb
from .pipeline import ResultadoPipeline, executar_pipeline
from .tokenizer import (
    TOKEN_DESCONHECIDO,
    TOKEN_FIM_DE_TEXTO,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    construir_vocabulario,
    destokenizar,
    inverter_vocabulario,
    juntar_textos,
    tokenizar,
)
from .transformer import (
    GELU,
    FeedForward,
    GPTModel,
    LayerNorm,
    TransformerBlock,
    memoria_em_mb,
    parametros_por_componente,
)

__all__ = [
    # tokenização e vocabulário
    "tokenizar",
    "destokenizar",
    "construir_vocabulario",
    "inverter_vocabulario",
    "juntar_textos",
    "SimpleTokenizerV1",
    "SimpleTokenizerV2",
    "TOKEN_FIM_DE_TEXTO",
    "TOKEN_DESCONHECIDO",
    # BPE
    "obter_tokenizador_bpe",
    "TiktokenTokenizer",
    "BPETokenizerSimples",
    # sequências e lotes
    "criar_pares_janela_deslizante",
    "numero_de_amostras",
    "GPTDatasetV1",
    "create_dataloader_v1",
    # embeddings
    "GPTEmbedding",
    "embedding_de_token",
    "comparar_posicoes",
    "tamanho_em_mb",
    # pipeline
    "executar_pipeline",
    "ResultadoPipeline",
    # atenção (Sprint 3)
    "self_attention_simples",
    "attention_scores_simples",
    "context_vector_passo_a_passo",
    "scaled_dot_product_attention",
    "SelfAttention_v1",
    "SelfAttention_v2",
    "CausalAttention",
    "MultiHeadAttentionWrapper",
    "MultiHeadAttention",
    "entropia_dos_pesos",
    "numero_de_parametros",
    # geração de texto (Sprint 4)
    "gerar",
    "gerar_texto_greedy",
    "gerar_texto",
    "gerar_passo_a_passo",
    "texto_para_ids",
    "ids_para_texto",
    "aplicar_temperatura",
    "aplicar_top_k",
    "distribuicao_do_proximo_token",
    "entropia_da_distribuicao",
    "tokens_para_cobrir",
    "taxa_de_repeticao",
    # Transformer Block e modelo GPT (Sprint 4)
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "GPTModel",
    "memoria_em_mb",
    "parametros_por_componente",
]
