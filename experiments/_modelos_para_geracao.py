"""
Modelos usados pelos experimentos de geração (exp16 e exp17).

A função de geração de `src/geracao.py` não depende de arquitetura: aceita
qualquer chamável que receba (batch, tokens) e devolva (batch, tokens, vocab).
Este módulo oferece duas opções enquanto o GPTModel do Entregável 3 não existe:

  * `ModeloProvisorio` — mini-GPT com as peças já prontas (embeddings da Sprint 2
    + Multi-Head Attention da Sprint 3 + camada de saída). Pesos aleatórios, sem
    treinamento: serve para medir formas, custo e o comportamento do laço.
  * `ModeloBigrama` — modelo de contagem de bigramas montado a partir do próprio
    corpus. Não é uma rede neural, mas devolve logits de verdade, o que torna
    visível no texto o efeito da temperatura e do top-k.

`obter_modelo_gpt()` usa automaticamente o GPTModel do Entregável 3 assim que ele
estiver disponível em `src`.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence

import torch
from torch import nn

from src.attention import MultiHeadAttention
from src.embeddings import GPTEmbedding


class ModeloProvisorio(nn.Module):
    """Embeddings -> Multi-Head Attention -> camada linear de saída (logits)."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int = 64,
        num_heads: int = 4,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        self.embeddings = GPTEmbedding(vocab_size, emb_dim, context_length, seed=seed)
        self.atencao = MultiHeadAttention(
            emb_dim, emb_dim, context_length, dropout=0.0, num_heads=num_heads, seed=seed
        )
        self.saida = nn.Linear(emb_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.num_heads = num_heads

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(idx)
        x = self.atencao(x)
        return self.saida(x)

    @property
    def nome(self) -> str:
        return f"ModeloProvisorio(emb={self.emb_dim}, heads={self.num_heads}, sem treinamento)"


class ModeloBigrama:
    """
    Modelo de linguagem por contagem de bigramas, usado como referência legível.

    Para cada token, guarda quantas vezes cada token apareceu logo depois dele no
    corpus. Os logits são o logaritmo dessas contagens (com suavização), de modo
    que a interface fica igual à de uma rede: (batch, tokens) -> (batch, tokens, vocab).

    As linhas são montadas sob demanda, a cada chamada. Guardar a matriz inteira
    exigiria vocab_size² valores — com o vocabulário do GPT-2 isso daria mais de
    2,5 bilhões de números, cerca de 10 GB. Como só as posições do contexto atual
    são necessárias, o custo real é batch × tokens × vocab_size.

    A suavização precisa ser pequena: ela é somada a TODOS os tokens do
    vocabulário, então um valor alto faz a cauda de tokens nunca observados
    somar mais massa que os sucessores reais, e a amostragem vira ruído.
    """

    def __init__(self, token_ids: Sequence[int], vocab_size: int, suavizacao: float = 1e-6) -> None:
        self.vocab_size = vocab_size
        self.context_length = 10_000          # não tem limite real de contexto
        self.suavizacao = suavizacao
        self.base = math.log(suavizacao)

        self.seguintes: dict = {}
        ids = list(token_ids)
        for anterior, seguinte in zip(ids, ids[1:]):
            self.seguintes.setdefault(anterior, Counter())[seguinte] += 1
        self.pares_distintos = sum(len(c) for c in self.seguintes.values())

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        b, n = idx.shape
        logits = torch.full((b, n, self.vocab_size), self.base)
        for linha in range(b):
            for posicao in range(n):
                contagens = self.seguintes.get(int(idx[linha, posicao]))
                if not contagens:
                    continue
                for token, vezes in contagens.items():
                    logits[linha, posicao, token] = math.log(vezes + self.suavizacao)
        return logits

    @property
    def nome(self) -> str:
        return f"ModeloBigrama({self.pares_distintos} pares distintos)"


def obter_modelo_gpt(vocab_size: int, context_length: int, **kwargs):
    """
    Devolve o GPTModel do Entregável 3 se ele já existir; senão, o provisório.

    Assim que o módulo da arquitetura for adicionado a `src`, os experimentos de
    geração passam a usá-lo sem precisar de nenhuma alteração.
    """
    for modulo, classe in (
        ("src.modelo", "GPTModel"),
        ("src.gpt", "GPTModel"),
        ("src.transformer", "GPTModel"),
        ("src.arquitetura", "GPTModel"),
    ):
        try:
            importado = __import__(modulo, fromlist=[classe])
            GPTModel = getattr(importado, classe)
        except (ImportError, AttributeError):
            continue
        try:
            return GPTModel(
                vocab_size=vocab_size, context_length=context_length, **kwargs
            )
        except TypeError:
            cfg = {
                "vocab_size": vocab_size,
                "context_length": context_length,
                "emb_dim": kwargs.get("emb_dim", 64),
                "n_heads": kwargs.get("num_heads", 4),
                "n_layers": kwargs.get("n_layers", 2),
                "drop_rate": kwargs.get("drop_rate", 0.0),
                "qkv_bias": False,
            }
            return GPTModel(cfg)
    return ModeloProvisorio(vocab_size, context_length, **kwargs)
