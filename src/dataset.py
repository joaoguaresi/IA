"""
Itens 3.3 e 3.6 da Sprint: preparação das sequências (entrada, alvo) com janela
deslizante e organização dos dados em lotes (Dataset + DataLoader do PyTorch).

Corresponde à seção 2.6 do capítulo (Amostragem com janela deslizante).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# 3.3 Preparação das sequências: pares (entrada, alvo)
# --------------------------------------------------------------------------- #
def criar_pares_janela_deslizante(
    token_ids: Sequence[int],
    max_length: int,
    stride: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Desliza uma janela de tamanho `max_length` sobre os Token IDs.

    Para cada posição i:
        entrada = token_ids[i : i + max_length]
        alvo    = token_ids[i + 1 : i + max_length + 1]   (deslocado em 1)

    O alvo é a própria entrada deslocada de uma posição, porque a tarefa de
    pré-treinamento é prever o próximo token.

    `stride` controla o avanço da janela:
        stride = 1            -> sobreposição máxima entre amostras
        stride = max_length   -> nenhuma sobreposição

    Número de amostras:
        n_amostras = floor((len(token_ids) - max_length - 1) / stride) + 1
    """
    if max_length <= 0 or stride <= 0:
        raise ValueError("max_length e stride devem ser positivos")
    if len(token_ids) <= max_length:
        raise ValueError(
            f"Texto curto demais: {len(token_ids)} tokens para max_length={max_length}. "
            "Use um texto maior ou um contexto menor."
        )

    entradas: List[List[int]] = []
    alvos: List[List[int]] = []
    for i in range(0, len(token_ids) - max_length, stride):
        entradas.append(list(token_ids[i : i + max_length]))
        alvos.append(list(token_ids[i + 1 : i + max_length + 1]))
    return entradas, alvos


def numero_de_amostras(n_tokens: int, max_length: int, stride: int) -> int:
    """Quantidade de amostras sem construir o dataset (útil para análises)."""
    disponivel = n_tokens - max_length
    if disponivel <= 0:
        return 0
    return (disponivel - 1) // stride + 1


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class GPTDatasetV1(Dataset):
    """
    Dataset de pares (entrada, alvo) — versão do capítulo.

    Cada item é uma tupla de tensores `torch.long` de forma (max_length,).
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int) -> None:
        token_ids = tokenizer.encode(txt)
        entradas, alvos = criar_pares_janela_deslizante(token_ids, max_length, stride)

        self.input_ids = [torch.tensor(x, dtype=torch.long) for x in entradas]
        self.target_ids = [torch.tensor(y, dtype=torch.long) for y in alvos]

        self.max_length = max_length
        self.stride = stride
        self.n_tokens = len(token_ids)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


# --------------------------------------------------------------------------- #
# 3.6 Organização dos dados: DataLoader
# --------------------------------------------------------------------------- #
def create_dataloader_v1(
    txt: str,
    tokenizer=None,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Cria o DataLoader que entrega lotes de entradas e alvos.

    Formas produzidas:
        entradas: (batch_size, max_length)
        alvos:    (batch_size, max_length)

    `drop_last=True` descarta o último lote incompleto, evitando picos de perda
    durante o treinamento.
    """
    if tokenizer is None:
        from .bpe import obter_tokenizador_bpe

        tokenizer = obter_tokenizador_bpe()

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
