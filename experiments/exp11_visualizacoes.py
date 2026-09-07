"""
Experimento 11 — Visualização das matrizes de atenção (entregável 7).

Gera os PNGs em experiments/figuras/:

  exp11_self_attention.png        atenção sem máscara, todos veem todos
  exp11_causal.png                a mesma sequência com a máscara causal
  exp11_com_e_sem_escala.png      efeito do sqrt(d_k) sobre a distribuição
  exp11_cabecas.png               uma matriz por cabeça do multi-head
  exp11_sequencias.png            a mesma camada em sequências diferentes
"""

import pathlib

import torch
from _comum import PASTA_FIGURAS, subtitulo, titulo

from src.attention import (
    CausalAttention,
    MultiHeadAttention,
    entropia_dos_pesos,
    scaled_dot_product_attention,
    self_attention_simples,
)
from src.visualizacao import grade_de_mapas, salvar_mapa

TOKENS = ["o", "velho", "consertou", "o", "relógio", "quebrado"]


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 11 — VISUALIZAÇÃO DAS MATRIZES DE ATENÇÃO")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[aviso] matplotlib não está instalado — nenhuma figura foi gerada.")
        print("        instale com: pip install matplotlib")
        return

    pasta = pathlib.Path(PASTA_FIGURAS)
    torch.manual_seed(123)
    n = len(TOKENS)
    inputs = torch.rand(n, 3)
    lote = inputs.unsqueeze(0)

    subtitulo("11.1 Self-attention sem máscara")
    _, pesos_simples = self_attention_simples(inputs)
    salvar_mapa(
        pesos_simples, pasta / "exp11_self_attention.png", TOKENS,
        titulo="Self-attention simplificado (sem máscara)",
        subtitulo="cada linha soma 1 · sem pesos treináveis",
    )

    subtitulo("11.2 A mesma sequência com máscara causal")
    causal = CausalAttention(d_in=3, d_out=2, context_length=n, dropout=0.0)
    causal(lote)
    salvar_mapa(
        causal.ultimo_attn_weights[0], pasta / "exp11_causal.png", TOKENS,
        titulo="Causal attention", causal=True,
        subtitulo="× = posição bloqueada (futuro) · a atenção se acumula à esquerda",
    )

    subtitulo("11.3 Com e sem a escala por raiz de d_k")
    torch.manual_seed(7)
    d_k = 128
    q, k, v = torch.randn(n, d_k), torch.randn(n, d_k), torch.randn(n, d_k)
    _, com = scaled_dot_product_attention(q, k, v, escalar=True)
    _, sem = scaled_dot_product_attention(q, k, v, escalar=False)
    grade_de_mapas(
        [com, sem],
        [f"com escala 1/√{d_k}  ·  entropia {entropia_dos_pesos(com):.2f} bits",
         f"sem escala  ·  entropia {entropia_dos_pesos(sem):.2f} bits"],
        pasta / "exp11_com_e_sem_escala.png", TOKENS,
        titulo_geral=f"Efeito da escala com d_k = {d_k}", colunas=2,
    )

    subtitulo("11.4 Uma matriz por cabeça")
    mha = MultiHeadAttention(d_in=16, d_out=16, context_length=n, num_heads=4)
    entrada = torch.rand(1, n, 16)
    mha(entrada)
    pesos = mha.ultimo_attn_weights[0]        # (heads, tokens, tokens)
    grade_de_mapas(
        [pesos[h] for h in range(pesos.shape[0])],
        [f"cabeça {h + 1}" for h in range(pesos.shape[0])],
        pasta / "exp11_cabecas.png", TOKENS, causal=True,
        titulo_geral="Multi-head attention: uma matriz de atenção por cabeça",
    )
    print(f"cabeças: {pesos.shape[0]}  |  matriz por cabeça: {tuple(pesos[0].shape)}")
    diferenca = float((pesos[0] - pesos[1]).abs().max())
    print(f"maior diferença entre a cabeça 1 e a cabeça 2: {diferenca:.4f}")

    subtitulo("11.5 A mesma camada em sequências diferentes")
    camada = CausalAttention(d_in=3, d_out=2, context_length=12, dropout=0.0)
    matrizes, rotulos = [], []
    for tamanho in (3, 6, 9, 12):
        torch.manual_seed(tamanho)
        camada(torch.rand(1, tamanho, 3))
        matrizes.append(camada.ultimo_attn_weights[0])
        rotulos.append(f"{tamanho} tokens")
    grade_de_mapas(
        matrizes, rotulos, pasta / "exp11_sequencias.png", causal=True,
        titulo_geral="A mesma camada causal em sequências de tamanhos diferentes",
    )

    print(f"\nTodas as figuras estão em: {pasta}")


if __name__ == "__main__":
    main()
