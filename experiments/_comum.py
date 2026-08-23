"""Utilidades compartilhadas pelos scripts de experimento."""

from __future__ import annotations

import pathlib
import sys
from typing import Iterable, Sequence

RAIZ = pathlib.Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

PASTA_FIGURAS = RAIZ / "experiments" / "figuras"


def titulo(texto: str) -> None:
    import torch

    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)
    print(f"[torch {torch.__version__} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}]")


def subtitulo(texto: str) -> None:
    print(f"\n--- {texto} " + "-" * max(0, 72 - len(texto)))


def tabela(cabecalho: Sequence[str], linhas: Iterable[Sequence]) -> None:
    linhas = [[str(c) for c in linha] for linha in linhas]
    larguras = [len(h) for h in cabecalho]
    for linha in linhas:
        for i, celula in enumerate(linha):
            larguras[i] = max(larguras[i], len(celula))
    fmt = "  ".join("{:<" + str(w) + "}" for w in larguras)
    print(fmt.format(*cabecalho))
    print("  ".join("-" * w for w in larguras))
    for linha in linhas:
        print(fmt.format(*linha))


def salvar_figura(fig, nome: str) -> None:
    """Salva uma figura matplotlib em experiments/figuras/ (silencioso se falhar)."""
    try:
        PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
        caminho = PASTA_FIGURAS / nome
        fig.savefig(caminho, dpi=140, bbox_inches="tight")
        print(f"[figura salva] {caminho.relative_to(RAIZ)}")
    except Exception as erro:  # pragma: no cover
        print(f"[aviso] não foi possível salvar a figura: {erro}")
