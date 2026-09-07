"""
Experimento 10 — Multi-head attention (seção 3.6).

Objetivos:
  * comparar a versão empilhada (wrapper) com a versão eficiente;
  * variar o número de cabeças e a dimensão de cada cabeça;
  * comparar self-attention de uma cabeça com multi-head;
  * medir parâmetros e tempo.
"""

import time

import torch
from _comum import subtitulo, tabela, titulo

from src.attention import (
    CausalAttention,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    numero_de_parametros,
)

TOKENS = ["o", "velho", "consertou", "o", "relógio", "quebrado"]


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 10 — MULTI-HEAD ATTENTION")
    torch.manual_seed(123)
    n = len(TOKENS)
    lote = torch.rand(2, n, 8)     # (batch=2, tokens, d_in=8)
    print(f"entrada: {tuple(lote.shape)}  (batch, tokens, d_in)")

    subtitulo("10.1 As duas implementações")
    wrapper = MultiHeadAttentionWrapper(d_in=8, d_out=4, context_length=n, num_heads=2)
    eficiente = MultiHeadAttention(d_in=8, d_out=8, context_length=n, num_heads=2)
    s_wrapper = wrapper(lote)
    s_eficiente = eficiente(lote)
    tabela(
        ["implementação", "saída", "parâmetros", "multiplicações Q/K/V"],
        [
            ["MultiHeadAttentionWrapper", str(tuple(s_wrapper.shape)),
             f"{numero_de_parametros(wrapper):,}", "3 por cabeça (6 no total)"],
            ["MultiHeadAttention", str(tuple(s_eficiente.shape)),
             f"{numero_de_parametros(eficiente):,}", "3 no total, divididas por reshape"],
        ],
    )

    subtitulo("10.2 A divisão em cabeças, passo a passo")
    b, tokens, _ = lote.shape
    q = eficiente.W_query(lote)
    print(f"1. projeção Q ................... {tuple(q.shape)}   (b, tokens, d_out)")
    q4 = q.view(b, tokens, eficiente.num_heads, eficiente.head_dim)
    print(f"2. view em cabeças .............. {tuple(q4.shape)} (b, tokens, heads, head_dim)")
    q4t = q4.transpose(1, 2)
    print(f"3. transpose(1, 2) .............. {tuple(q4t.shape)} (b, heads, tokens, head_dim)")
    print(f"4. scores = Q @ Kᵀ .............. {tuple(eficiente.ultimo_attn_weights.shape)}"
          "  (b, heads, tokens, tokens)")
    print(f"5. volta e reconsolida .......... {tuple(s_eficiente.shape)}   (b, tokens, d_out)")

    subtitulo("10.3 Número de cabeças (mantendo d_out fixo em 96)")
    linhas = []
    for heads in (1, 2, 4, 6, 8, 12):
        camada = MultiHeadAttention(d_in=96, d_out=96, context_length=n, num_heads=heads)
        entrada = torch.rand(1, n, 96)
        inicio = time.perf_counter()
        saida = camada(entrada)
        ms = (time.perf_counter() - inicio) * 1000
        linhas.append([
            heads, 96 // heads, str(tuple(saida.shape)),
            str(tuple(camada.ultimo_attn_weights.shape)),
            f"{numero_de_parametros(camada):,}", f"{ms:.2f} ms",
        ])
    tabela(["nº de cabeças", "head_dim", "saída", "matrizes de atenção",
            "parâmetros", "tempo"], linhas)

    subtitulo("10.4 Dimensão da cabeça (mantendo 4 cabeças)")
    linhas = []
    for head_dim in (8, 16, 32, 64):
        d_out = head_dim * 4
        camada = MultiHeadAttention(d_in=d_out, d_out=d_out, context_length=n, num_heads=4)
        entrada = torch.rand(1, n, d_out)
        saida = camada(entrada)
        linhas.append([
            head_dim, d_out, str(tuple(saida.shape)),
            f"{numero_de_parametros(camada):,}", f"{head_dim ** 0.5:.2f}",
        ])
    tabela(["head_dim", "d_out", "saída", "parâmetros", "divisor √head_dim"], linhas)

    subtitulo("10.5 d_out precisa ser divisível pelo número de cabeças")
    try:
        MultiHeadAttention(d_in=8, d_out=10, context_length=n, num_heads=4)
    except ValueError as erro:
        print(f"ValueError: {erro}")

    subtitulo("10.6 Uma cabeça vs. várias")
    uma = CausalAttention(d_in=8, d_out=8, context_length=n)
    varias = MultiHeadAttention(d_in=8, d_out=8, context_length=n, num_heads=4)
    p_uma, p_varias = uma(lote), varias(lote)
    tabela(
        ["mecanismo", "saída", "matrizes de atenção", "parâmetros"],
        [
            ["CausalAttention (1 cabeça)", str(tuple(p_uma.shape)),
             str(tuple(uma.ultimo_attn_weights.shape)), f"{numero_de_parametros(uma):,}"],
            ["MultiHeadAttention (4 cabeças)", str(tuple(p_varias.shape)),
             str(tuple(varias.ultimo_attn_weights.shape)), f"{numero_de_parametros(varias):,}"],
        ],
    )


if __name__ == "__main__":
    main()
