"""
Seção 2.5 do capítulo: Byte Pair Encoding (BPE).

Duas implementações:

1. `TiktokenTokenizer` — o tokenizador oficial do GPT-2 (biblioteca `tiktoken`),
   com vocabulário de 50.257 tokens, encapsulado na mesma interface dos
   tokenizadores de `src.tokenizer` (encode / decode / n_vocab / nome).

2. `BPETokenizerSimples` — implementação didática e treinável do algoritmo BPE
   em Python puro (sem dependências), para demonstrar como o vocabulário de
   subpalavras é construído por fusões (merges) sucessivas dos pares mais
   frequentes.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import tiktoken

TOKEN_FIM_DE_TEXTO = "<|endoftext|>"


# --------------------------------------------------------------------------- #
# 1. Tokenizador BPE do GPT-2 (tiktoken)
# --------------------------------------------------------------------------- #
class TiktokenTokenizer:
    """Adaptador de `tiktoken` para a interface usada no projeto."""

    def __init__(self, codificacao: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(codificacao)
        self._codificacao = codificacao

    @property
    def nome(self) -> str:
        return f"tiktoken/{self._codificacao}"

    @property
    def n_vocab(self) -> int:
        return self._enc.n_vocab

    def encode(self, texto: str) -> List[int]:
        return self._enc.encode(texto, allowed_special={TOKEN_FIM_DE_TEXTO})

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode([int(i) for i in ids])

    def tokens(self, texto: str) -> List[str]:
        """Mostra o pedaço de texto correspondente a cada Token ID."""
        return [self.decode([i]) for i in self.encode(texto)]


def obter_tokenizador_bpe(codificacao: str = "gpt2") -> TiktokenTokenizer:
    """Tokenizador BPE do GPT-2 (50.257 tokens)."""
    return TiktokenTokenizer(codificacao)


# --------------------------------------------------------------------------- #
# 2. BPE treinável (implementação didática)
# --------------------------------------------------------------------------- #
def _contar_pares(ids: Sequence[int]) -> Counter:
    return Counter(zip(ids, ids[1:]))


def _fundir(ids: Sequence[int], par: Tuple[int, int], novo_id: int) -> List[int]:
    saida: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == par:
            saida.append(novo_id)
            i += 2
        else:
            saida.append(ids[i])
            i += 1
    return saida


class BPETokenizerSimples:
    """
    BPE treinável em Python puro.

    Começa com os 256 bytes possíveis e funde iterativamente o par de tokens
    mais frequente, criando tokens de subpalavra — a ideia descrita na seção 2.5.
    Palavras desconhecidas nunca quebram o tokenizador: no pior caso são
    decompostas em bytes individuais (por isso o GPT não precisa de `<|unk|>`).
    """

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.id_fim_de_texto: Optional[int] = None

    # -- treinamento -------------------------------------------------------- #
    def treinar(
        self, texto: str, tamanho_vocab: int, verbose: bool = False
    ) -> "BPETokenizerSimples":
        if tamanho_vocab < 256:
            raise ValueError("tamanho_vocab deve ser >= 256 (bytes iniciais)")
        num_fusoes = tamanho_vocab - 256
        ids = list(texto.encode("utf-8"))

        for i in range(num_fusoes):
            pares = _contar_pares(ids)
            if not pares:
                break
            par = max(pares, key=pares.get)
            if pares[par] < 2:
                break
            novo_id = 256 + i
            ids = _fundir(ids, par, novo_id)
            self.merges[par] = novo_id
            self.vocab[novo_id] = self.vocab[par[0]] + self.vocab[par[1]]
            if verbose:
                pedaco = self.vocab[novo_id].decode("utf-8", errors="replace")
                print(f"  fusão {i + 1:>3}: {par} -> {novo_id} ({pedaco!r}), freq={pares[par]}")

        self.id_fim_de_texto = 256 + len(self.merges)
        self.vocab[self.id_fim_de_texto] = TOKEN_FIM_DE_TEXTO.encode("utf-8")
        return self

    # -- interface padrão --------------------------------------------------- #
    @property
    def nome(self) -> str:
        return f"BPETokenizerSimples({self.n_vocab} tokens)"

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)

    def _encode_bytes(self, texto: str) -> List[int]:
        ids = list(texto.encode("utf-8"))
        while len(ids) >= 2:
            pares = _contar_pares(ids)
            par = min(pares, key=lambda p: self.merges.get(p, float("inf")))
            if par not in self.merges:
                break
            ids = _fundir(ids, par, self.merges[par])
        return ids

    def encode(self, texto: str) -> List[int]:
        if self.id_fim_de_texto is None or TOKEN_FIM_DE_TEXTO not in texto:
            return self._encode_bytes(texto)
        ids: List[int] = []
        partes = texto.split(TOKEN_FIM_DE_TEXTO)
        for indice, parte in enumerate(partes):
            ids.extend(self._encode_bytes(parte))
            if indice < len(partes) - 1:
                ids.append(self.id_fim_de_texto)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, texto: str) -> List[str]:
        return [self.decode([i]) for i in self.encode(texto)]
