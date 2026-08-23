"""
Itens 3.4 e 3.5 da Sprint: token embeddings e positional embeddings.

Corresponde às seções 2.7 (Criando token embeddings) e 2.8 (Codificando
posições de palavras) do Capítulo 2.

    Token ID (inteiro)  --lookup-->  vetor de `output_dim` dimensões
    input_embedding = token_embedding + positional_embedding
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


class GPTEmbedding(nn.Module):
    """
    Camada de entrada de um LLM do tipo GPT.

    - `tok_emb`: matriz (vocab_size, output_dim) — lookup por Token ID.
    - `pos_emb`: matriz (context_length, output_dim) — positional embeddings
      absolutos, aprendidos durante o treinamento (como no GPT).

    A soma dos dois produz os *input embeddings* enviados ao bloco de atenção
    da Sprint 3.

    Exemplo
    -------
    >>> camada = GPTEmbedding(vocab_size=50257, output_dim=768, context_length=256)
    >>> entradas = torch.randint(0, 50257, (8, 256))
    >>> camada(entradas).shape
    torch.Size([8, 256, 768])
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        context_length: int,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.tok_emb = nn.Embedding(vocab_size, output_dim)
        self.pos_emb = nn.Embedding(context_length, output_dim)
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.context_length = context_length

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"sequência de {seq_len} tokens excede context_length={self.context_length}"
            )
        token_embeddings = self.tok_emb(in_idx)                    # (b, T, d)
        posicoes = torch.arange(seq_len, device=in_idx.device)     # (T,)
        pos_embeddings = self.pos_emb(posicoes)                    # (T, d)
        return token_embeddings + pos_embeddings                   # broadcast -> (b, T, d)

    def num_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Utilidades de análise
# --------------------------------------------------------------------------- #
def tamanho_em_mb(vocab_size: int, output_dim: int, bytes_por_parametro: int = 4) -> float:
    """Memória ocupada pela matriz de token embeddings, em MB (float32)."""
    return vocab_size * output_dim * bytes_por_parametro / (1024**2)


@torch.no_grad()
def embedding_de_token(camada: GPTEmbedding, token_id: int) -> torch.Tensor:
    """Vetor de embedding (sem informação de posição) de um único Token ID."""
    indice = torch.tensor([token_id], dtype=torch.long, device=camada.tok_emb.weight.device)
    return camada.tok_emb(indice)[0]


@torch.no_grad()
def comparar_posicoes(
    camada: GPTEmbedding,
    token_id: int,
    posicoes: Sequence[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[Tuple[int, int], float]]:
    """
    Demonstra experimentalmente a necessidade dos positional embeddings.

    Coloca o MESMO Token ID em posições diferentes da sequência e devolve os
    vetores finais (token + posição) e as distâncias entre eles. Sem positional
    embeddings, todos seriam idênticos e a distância seria zero.
    """
    comprimento = max(posicoes) + 1
    dispositivo = camada.tok_emb.weight.device
    entrada = torch.full((1, comprimento), token_id, dtype=torch.long, device=dispositivo)
    saida = camada(entrada)[0]

    vetores = {p: saida[p] for p in posicoes}
    distancias = {
        (a, b): float(torch.norm(vetores[a] - vetores[b]))
        for i, a in enumerate(posicoes)
        for b in posicoes[i + 1 :]
    }
    return vetores, distancias
