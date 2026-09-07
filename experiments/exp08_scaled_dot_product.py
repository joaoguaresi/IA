"""
Experimento 8 — Scaled dot-product attention (seção 3.4).

Objetivos:
  * comparar SelfAttention_v1 (nn.Parameter) e SelfAttention_v2 (nn.Linear);
  * medir o efeito da divisão por raiz de d_k;
  * variar a dimensão de embedding e observar parâmetros e formas.
"""

import torch
from _comum import salvar_figura, subtitulo, tabela, titulo

from src.attention import (
    SelfAttention_v1,
    SelfAttention_v2,
    entropia_dos_pesos,
    numero_de_parametros,
    scaled_dot_product_attention,
)

TOKENS = ["o", "velho", "consertou", "o", "relógio", "quebrado"]


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 8 — SCALED DOT-PRODUCT ATTENTION")
    torch.manual_seed(123)
    inputs = torch.rand(len(TOKENS), 3)
    d_in, d_out = 3, 2

    subtitulo("8.1 As três projeções: query, key e value")
    camada = SelfAttention_v1(d_in, d_out)
    saida = camada(inputs)
    print(f"entrada .......... {tuple(inputs.shape)}   (tokens, d_in)")
    print(f"Wq, Wk, Wv ....... {tuple(camada.W_query.shape)} cada uma  (d_in, d_out)")
    print(f"context vectors .. {tuple(saida.shape)}   (tokens, d_out)")
    print(f"parâmetros ....... {numero_de_parametros(camada)}")

    subtitulo("8.2 V1 (nn.Parameter) vs. V2 (nn.Linear)")
    v1 = SelfAttention_v1(d_in, d_out, seed=123)
    v2 = SelfAttention_v2(d_in, d_out, seed=123)
    tabela(
        ["implementação", "saída", "parâmetros", "média dos pesos Wq", "desvio padrão"],
        [
            ["SelfAttention_v1 (torch.rand)", str(tuple(v1(inputs).shape)),
             numero_de_parametros(v1), f"{float(v1.W_query.mean()):+.4f}",
             f"{float(v1.W_query.std()):.4f}"],
            ["SelfAttention_v2 (nn.Linear)", str(tuple(v2(inputs).shape)),
             numero_de_parametros(v2), f"{float(v2.W_query.weight.mean()):+.4f}",
             f"{float(v2.W_query.weight.std()):.4f}"],
        ],
    )

    subtitulo("8.3 O efeito da escala por raiz de d_k")
    linhas = []
    for d_k in (2, 8, 32, 128, 512):
        torch.manual_seed(7)
        q = torch.randn(6, d_k)
        k = torch.randn(6, d_k)
        v = torch.randn(6, d_k)
        _, com = scaled_dot_product_attention(q, k, v, escalar=True)
        _, sem = scaled_dot_product_attention(q, k, v, escalar=False)
        linhas.append([
            d_k,
            f"{float((q @ k.T).abs().max()):8.2f}",
            f"{float(com.max()):.3f}", f"{entropia_dos_pesos(com):.3f}",
            f"{float(sem.max()):.3f}", f"{entropia_dos_pesos(sem):.3f}",
        ])
    tabela(
        ["d_k", "maior score bruto", "maior peso (com escala)", "entropia (com)",
         "maior peso (sem escala)", "entropia (sem)"],
        linhas,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dks = [2, 8, 32, 128, 512]
        ent_com, ent_sem = [], []
        for d_k in dks:
            torch.manual_seed(7)
            q, k, v = torch.randn(6, d_k), torch.randn(6, d_k), torch.randn(6, d_k)
            ent_com.append(entropia_dos_pesos(scaled_dot_product_attention(q, k, v, escalar=True)[1]))
            ent_sem.append(entropia_dos_pesos(scaled_dot_product_attention(q, k, v, escalar=False)[1]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(dks, ent_com, "o-", color="#1C5D86", linewidth=2, label="com escala 1/√d_k")
        ax.plot(dks, ent_sem, "s--", color="#9A5312", linewidth=2, label="sem escala")
        ax.axhline(2.585, color="#9CA3AF", linewidth=1, linestyle=":",
                   label="atenção uniforme (log₂ 6)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("dimensão das keys (d_k)")
        ax.set_ylabel("entropia média dos pesos (bits)")
        ax.set_title("A escala por √d_k impede o softmax de saturar")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.legend(frameon=False, fontsize=9)
        salvar_figura(fig, "exp08_efeito_da_escala.png")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib não instalado — gráfico não gerado.")

    subtitulo("8.4 Diferentes dimensões de embedding")
    linhas = []
    for d_in_i, d_out_i in ((3, 2), (16, 16), (64, 64), (256, 256), (768, 768)):
        camada = SelfAttention_v2(d_in_i, d_out_i)
        torch.manual_seed(0)
        x = torch.rand(6, d_in_i)
        y = camada(x)
        linhas.append([
            d_in_i, d_out_i, str(tuple(x.shape)), str(tuple(y.shape)),
            f"{numero_de_parametros(camada):,}",
            f"{numero_de_parametros(camada) * 4 / 1024:.1f} KB",
        ])
    tabela(["d_in", "d_out", "entrada", "saída", "parâmetros", "memória dos pesos"], linhas)


if __name__ == "__main__":
    main()
