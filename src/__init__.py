"""
Sprint 2 — preparação de dados de texto para um LLM (Capítulo 2).

Fluxo implementado:

    Texto -> Tokenização -> Tokens -> Token IDs -> Sequências (entrada, alvo)
          -> Token Embeddings -> Positional Embeddings -> Lote -> Entrada do modelo

Uso típico:

    from src import create_dataloader_v1, GPTEmbedding, obter_tokenizador_bpe

    tokenizador = obter_tokenizador_bpe()
    dataloader = create_dataloader_v1(texto, tokenizer=tokenizador,
                                      batch_size=8, max_length=256, stride=128)
    entradas, alvos = next(iter(dataloader))

    camada = GPTEmbedding(vocab_size=tokenizador.n_vocab,
                          output_dim=768, context_length=256)
    input_embeddings = camada(entradas)   # (8, 256, 768)

Requer: torch e tiktoken.
"""

from .bpe import BPETokenizerSimples, TiktokenTokenizer, obter_tokenizador_bpe
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
]
