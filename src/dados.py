"""Carregamento do corpus de treinamento."""

from __future__ import annotations

import pathlib
from typing import Optional, Union

PASTA_DADOS = pathlib.Path(__file__).resolve().parents[1] / "data"
VERDICT = PASTA_DADOS / "the-verdict.txt"
AMOSTRA = PASTA_DADOS / "amostra.txt"


def carregar_texto(caminho: Optional[Union[str, pathlib.Path]] = None) -> str:
    """
    Devolve o texto de treinamento.

    Prioridade: caminho informado > data/the-verdict.txt (se baixado) > data/amostra.txt.
    Ao integrar em outro projeto, passe sempre o caminho do seu próprio corpus.
    """
    if caminho is not None:
        return pathlib.Path(caminho).read_text(encoding="utf-8")
    if VERDICT.exists():
        return VERDICT.read_text(encoding="utf-8")
    if AMOSTRA.exists():
        return AMOSTRA.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Nenhum corpus encontrado em {PASTA_DADOS}. "
        "Informe `caminho` ou execute `python data/baixar_dados.py`."
    )


def nome_do_corpus(caminho: Optional[Union[str, pathlib.Path]] = None) -> str:
    if caminho is not None:
        return pathlib.Path(caminho).name
    return VERDICT.name if VERDICT.exists() else AMOSTRA.name
