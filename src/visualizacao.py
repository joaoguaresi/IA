"""
Visualização das matrizes de atenção (entregável 7 da Sprint 3).

Uma matriz de atenção é uma grade tokens × tokens em que a célula [i, j] diz
quanto peso o token da linha i deu ao token da coluna j. Cada linha soma 1.

Convenções usadas aqui:
  * escala sequencial de um único tom (claro = pouco peso, escuro = muito peso);
  * células bloqueadas pela máscara causal aparecem em cinza, marcadas com ×,
    para que a estrutura triangular se leia mesmo em impressão preto e branco;
  * valores são escritos dentro das células quando a matriz é pequena.
"""

from __future__ import annotations

import pathlib
from typing import Optional, Sequence

import numpy as np

TOM_SEQUENCIAL = "Blues"      # sequencial de um único tom: claro -> escuro
COR_BLOQUEADO = "#E8EAED"     # cinza neutro para as células mascaradas
COR_TEXTO = "#1F2430"
COR_TEXTO_CLARO = "#FFFFFF"
COR_SECUNDARIA = "#6B7280"


def _para_numpy(tensor) -> np.ndarray:
    """Aceita tensor do PyTorch ou array do NumPy."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return np.asarray(tensor, dtype=float)


def mapa_de_atencao(
    ax,
    pesos,
    rotulos: Optional[Sequence[str]] = None,
    titulo: str = "",
    causal: bool = False,
    vmax: Optional[float] = None,
    anotar: Optional[bool] = None,
):
    """
    Desenha uma matriz de atenção em um eixo do matplotlib.

    Parâmetros
    ----------
    pesos : matriz (num_tokens, num_tokens) com os pesos de atenção.
    rotulos : os tokens, usados nos dois eixos.
    causal : se True, marca as células bloqueadas (peso zero acima da diagonal).
    vmax : valor máximo da escala de cor — informe o mesmo em vários mapas para
           que eles fiquem comparáveis entre si.
    anotar : escreve o valor dentro da célula (padrão: apenas se a matriz é pequena).
    """
    m = _para_numpy(pesos)
    n = m.shape[0]
    if anotar is None:
        anotar = n <= 12
    if vmax is None:
        vmax = float(m.max())

    dados = np.ma.masked_where(np.triu(np.ones_like(m), k=1) > 0, m) if causal else m

    imagem = ax.imshow(dados, cmap=TOM_SEQUENCIAL, vmin=0.0, vmax=vmax, aspect="equal")
    imagem.cmap.set_bad(COR_BLOQUEADO)

    if rotulos is not None:
        ax.set_xticks(range(n), rotulos, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), rotulos, fontsize=8)
    else:
        ax.set_xticks(range(n), [str(i) for i in range(n)], fontsize=8)
        ax.set_yticks(range(n), [str(i) for i in range(n)], fontsize=8)

    ax.set_xlabel("token atendido (key)", fontsize=8.5, color=COR_SECUNDARIA)
    ax.set_ylabel("token que consulta (query)", fontsize=8.5, color=COR_SECUNDARIA)
    if titulo:
        ax.set_title(titulo, fontsize=10.5, pad=9)

    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.tick_params(length=0)

    # linhas finas de separação, na cor do fundo (regra do espaçador de 2px)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    if anotar:
        for i in range(n):
            for j in range(n):
                if causal and j > i:
                    ax.text(j, i, "×", ha="center", va="center",
                            fontsize=9, color=COR_SECUNDARIA)
                    continue
                valor = m[i, j]
                cor = COR_TEXTO_CLARO if valor > vmax * 0.6 else COR_TEXTO
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                        fontsize=7.5, color=cor)
    return imagem


def salvar_mapa(
    pesos,
    caminho,
    rotulos: Optional[Sequence[str]] = None,
    titulo: str = "",
    subtitulo: str = "",
    causal: bool = False,
):
    """Salva um único mapa de atenção como PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    imagem = mapa_de_atencao(ax, pesos, rotulos, titulo, causal=causal)
    barra = fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("peso de atenção", fontsize=8.5, color=COR_SECUNDARIA)
    barra.outline.set_visible(False)
    if subtitulo:
        # abaixo do eixo x, fora da área do gráfico (o bbox_inches="tight" inclui)
        ax.annotate(subtitulo, xy=(0, 0), xytext=(0, -62), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8, color=COR_SECUNDARIA,
                    annotation_clip=False)
    return _salvar(fig, caminho)


def grade_de_mapas(
    lista_pesos: Sequence,
    titulos: Sequence[str],
    caminho,
    rotulos: Optional[Sequence[str]] = None,
    causal: bool = False,
    titulo_geral: str = "",
    colunas: int = 4,
):
    """
    Small multiples: vários mapas lado a lado com a MESMA escala de cor.

    Usada para as cabeças do multi-head attention e para comparar configurações.
    A escala compartilhada é o que torna os quadros comparáveis entre si.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(lista_pesos)
    colunas = min(colunas, n)
    linhas = (n + colunas - 1) // colunas
    vmax = max(float(_para_numpy(p).max()) for p in lista_pesos)

    fig, eixos = plt.subplots(linhas, colunas, figsize=(3.5 * colunas, 3.5 * linhas))
    eixos = np.atleast_1d(eixos).ravel()

    imagem = None
    for i, (pesos, titulo) in enumerate(zip(lista_pesos, titulos)):
        imagem = mapa_de_atencao(
            eixos[i], pesos, rotulos, titulo, causal=causal, vmax=vmax,
            anotar=_para_numpy(pesos).shape[0] <= 8,
        )
        if i % colunas:                      # rótulos do eixo y só na 1ª coluna
            eixos[i].set_ylabel("")
            eixos[i].set_yticklabels([])
        if i < len(lista_pesos) - colunas:   # rótulo do eixo x só na última linha
            eixos[i].set_xlabel("")
    for j in range(n, len(eixos)):
        eixos[j].axis("off")

    if imagem is not None:
        barra = fig.colorbar(imagem, ax=eixos[:n].tolist(), fraction=0.02, pad=0.02)
        barra.set_label("peso de atenção", fontsize=8.5, color=COR_SECUNDARIA)
        barra.outline.set_visible(False)
    if titulo_geral:
        fig.suptitle(titulo_geral, fontsize=12.5, y=0.99)
    return _salvar(fig, caminho)


def _salvar(fig, caminho, apertar: bool = True):
    import matplotlib.pyplot as plt

    caminho = pathlib.Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight" if apertar else None,
                facecolor="white")
    plt.close(fig)
    print(f"[figura salva] {caminho}")
    return caminho
