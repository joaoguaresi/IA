"""
Itens 3.1 e 3.2 da Sprint: tokenização, vocabulário e Token IDs.

Corresponde às seções 2.2 (Tokenizando texto), 2.3 (Convertendo tokens em
token IDs) e 2.4 (Tokens de contexto especiais) do Capítulo 2.

Relação fundamental demonstrada aqui:

    Token  <->  Token ID  <->  Vocabulário
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

# Expressão regular do capítulo: separa pontuação, travessão e espaços,
# preservando maiúsculas/minúsculas (para o LLM distinguir nomes próprios).
PADRAO_TOKENIZACAO = r'([,.:;?_!"()\']|--|\s)'

TOKEN_FIM_DE_TEXTO = "<|endoftext|>"
TOKEN_DESCONHECIDO = "<|unk|>"
TOKENS_ESPECIAIS_PADRAO = (TOKEN_FIM_DE_TEXTO, TOKEN_DESCONHECIDO)


# --------------------------------------------------------------------------- #
# 3.1 Tokenização
# --------------------------------------------------------------------------- #
def tokenizar(texto: str, manter_espacos: bool = False) -> List[str]:
    """
    Divide um texto em tokens (palavras e sinais de pontuação).

    >>> tokenizar("Olá, mundo! Isso é um teste.")
    ['Olá', ',', 'mundo', '!', 'Isso', 'é', 'um', 'teste', '.']

    Parâmetros
    ----------
    texto : str
        Trecho de texto bruto.
    manter_espacos : bool
        Se True, os espaços em branco também são devolvidos como tokens
        (útil para mostrar que a tokenização é reversível caractere a caractere).
    """
    partes = re.split(PADRAO_TOKENIZACAO, texto)
    if manter_espacos:
        return [p for p in partes if p != ""]
    return [p.strip() for p in partes if p.strip() != ""]


def destokenizar(tokens: Sequence[str]) -> str:
    """Reconstrói um texto a partir de tokens, colando a pontuação à palavra."""
    texto = " ".join(tokens)
    return re.sub(r'\s+([,.:;?_!"()\'])', r"\1", texto)


# --------------------------------------------------------------------------- #
# 3.2 Vocabulário
# --------------------------------------------------------------------------- #
def construir_vocabulario(
    texto: str,
    tokens_especiais: Iterable[str] = TOKENS_ESPECIAIS_PADRAO,
) -> Dict[str, int]:
    """
    Constrói o vocabulário: dicionário {token -> token ID}.

    Os tokens únicos são ordenados alfabeticamente e recebem IDs sequenciais.
    Os tokens especiais são acrescentados no final, com os maiores IDs.
    """
    tokens_unicos = sorted(set(tokenizar(texto)))
    for especial in tokens_especiais:
        if especial not in tokens_unicos:
            tokens_unicos.append(especial)
    return {token: indice for indice, token in enumerate(tokens_unicos)}


def inverter_vocabulario(vocabulario: Dict[str, int]) -> Dict[int, str]:
    """Vocabulário inverso: {token ID -> token}, usado na decodificação."""
    return {indice: token for token, indice in vocabulario.items()}


# --------------------------------------------------------------------------- #
# Tokenizadores baseados em vocabulário
# --------------------------------------------------------------------------- #
class SimpleTokenizerV1:
    """
    Tokenizador simples do capítulo (seção 2.3).

    - `encode`: texto -> lista de Token IDs (via vocabulário)
    - `decode`: lista de Token IDs -> texto (via vocabulário inverso)

    Levanta KeyError diante de uma palavra fora do vocabulário — limitação
    que motiva a versão V2 com o token `<|unk|>`.
    """

    def __init__(self, vocabulario: Dict[str, int]) -> None:
        self.str_para_int = vocabulario
        self.int_para_str = inverter_vocabulario(vocabulario)

    @property
    def n_vocab(self) -> int:
        return len(self.str_para_int)

    @property
    def nome(self) -> str:
        return "SimpleTokenizerV1"

    def encode(self, texto: str) -> List[int]:
        tokens = tokenizar(texto)
        faltantes = [t for t in tokens if t not in self.str_para_int]
        if faltantes:
            raise KeyError(
                f"Tokens fora do vocabulário: {faltantes[:5]} "
                f"(use SimpleTokenizerV2 ou o BPE)"
            )
        return [self.str_para_int[t] for t in tokens]

    def decode(self, ids: Sequence[int]) -> str:
        return destokenizar([self.int_para_str[i] for i in ids])


class SimpleTokenizerV2(SimpleTokenizerV1):
    """
    Versão com tokens de contexto especiais (seção 2.4).

    Palavras fora do vocabulário são substituídas por `<|unk|>`; textos
    independentes podem ser concatenados com `<|endoftext|>`.
    """

    def __init__(
        self,
        vocabulario: Dict[str, int],
        token_desconhecido: str = TOKEN_DESCONHECIDO,
    ) -> None:
        if token_desconhecido not in vocabulario:
            vocabulario = dict(vocabulario)
            vocabulario[token_desconhecido] = len(vocabulario)
        super().__init__(vocabulario)
        self.token_desconhecido = token_desconhecido

    @property
    def nome(self) -> str:
        return "SimpleTokenizerV2"

    def encode(self, texto: str) -> List[int]:
        tokens = [
            t if t in self.str_para_int else self.token_desconhecido
            for t in tokenizar(texto)
        ]
        return [self.str_para_int[t] for t in tokens]


def juntar_textos(*textos: str) -> str:
    """Concatena textos independentes usando `<|endoftext|>` como separador."""
    return f" {TOKEN_FIM_DE_TEXTO} ".join(textos)
