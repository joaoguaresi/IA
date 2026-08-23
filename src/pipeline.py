"""
Integração das etapas: do texto bruto até os input embeddings.

    Texto -> Tokenização -> Tokens -> Token IDs -> Sequências de treinamento
          -> Embeddings -> Positional Embeddings -> Lote -> Entrada do modelo

O tensor produzido aqui, de forma (batch_size, context_length, output_dim), é
exatamente a entrada esperada pelo self-attention da Sprint 3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .bpe import obter_tokenizador_bpe
from .dataset import create_dataloader_v1
from .embeddings import GPTEmbedding


@dataclass
class ResultadoPipeline:
    """Resumo das dimensões produzidas em cada etapa do pipeline."""

    n_caracteres: int
    n_tokens: int
    tamanho_vocabulario: int
    n_amostras: int
    n_lotes: int
    batch_size: int
    context_length: int
    output_dim: int
    forma_lote_entradas: tuple
    forma_lote_alvos: tuple
    forma_token_embeddings: tuple
    forma_input_embeddings: tuple
    parametros_embedding: int
    tempo_segundos: float
    extras: Dict[str, Any] = field(default_factory=dict)

    def imprimir(self) -> None:
        print("\n--- Pipeline de dados da Sprint 2 -------------------------------")
        print(f"Texto bruto ......................... {self.n_caracteres:,} caracteres")
        print("  |")
        print(f"  v Tokenização + Token IDs .......... {self.n_tokens:,} tokens")
        print(f"    (vocabulário: {self.tamanho_vocabulario:,} tokens)")
        print("  |")
        print(f"  v Janela deslizante ................ {self.n_amostras:,} amostras (entrada, alvo)")
        print("  |")
        print(f"  v Lotes (DataLoader) ............... {self.n_lotes:,} lotes de {self.batch_size}")
        print(f"    entradas {self.forma_lote_entradas}  alvos {self.forma_lote_alvos}")
        print("  |")
        print(f"  v Token embeddings ................. {self.forma_token_embeddings}")
        print("  |")
        print(f"  v + Positional embeddings .......... {self.forma_input_embeddings}")
        print("  |")
        print(f"  v Entrada do modelo (Sprint 3) ..... {self.forma_input_embeddings}")
        print(f"\nParâmetros das camadas de embedding: {self.parametros_embedding:,}")
        print(f"Tempo total: {self.tempo_segundos:.3f} s")
        print("-----------------------------------------------------------------")


@torch.no_grad()
def executar_pipeline(
    texto: str,
    context_length: int = 256,
    stride: int = 128,
    batch_size: int = 8,
    output_dim: int = 256,
    tokenizer: Optional[object] = None,
    shuffle: bool = False,
    seed: int = 123,
    device: Optional[str] = None,
) -> ResultadoPipeline:
    """Executa o fluxo completo e devolve o resumo das dimensões."""
    inicio = time.perf_counter()

    if tokenizer is None:
        tokenizer = obter_tokenizador_bpe()

    dataloader = create_dataloader_v1(
        texto,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=context_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=True,
    )

    entradas, alvos = next(iter(dataloader))

    camada = GPTEmbedding(
        vocab_size=tokenizer.n_vocab,
        output_dim=output_dim,
        context_length=context_length,
        seed=seed,
    )
    if device is not None:
        camada = camada.to(device)
        entradas = entradas.to(device)
        alvos = alvos.to(device)

    token_embeddings = camada.tok_emb(entradas)
    input_embeddings = camada(entradas)

    dataset = dataloader.dataset

    return ResultadoPipeline(
        n_caracteres=len(texto),
        n_tokens=dataset.n_tokens,
        tamanho_vocabulario=tokenizer.n_vocab,
        n_amostras=len(dataset),
        n_lotes=len(dataloader),
        batch_size=batch_size,
        context_length=context_length,
        output_dim=output_dim,
        forma_lote_entradas=tuple(entradas.shape),
        forma_lote_alvos=tuple(alvos.shape),
        forma_token_embeddings=tuple(token_embeddings.shape),
        forma_input_embeddings=tuple(input_embeddings.shape),
        parametros_embedding=camada.num_parametros(),
        tempo_segundos=time.perf_counter() - inicio,
        extras={
            "tokenizador": tokenizer.nome,
            "stride": stride,
            "device": str(input_embeddings.device),
        },
    )
