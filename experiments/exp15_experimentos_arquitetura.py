"""
Experimento 15 — Experimentação de arquitetura (parte do Entregável 5).

Varia número de blocos, número de cabeças, dimensão do embedding, taxa de
dropout e tamanho do contexto, observando o efeito sobre o número de
parâmetros, a memória e o custo computacional (tempo de forward) do modelo.
"""

import time

import torch
from _comum import salvar_figura, subtitulo, tabela, titulo

from src.transformer import GPTModel, memoria_em_mb

VOCAB = 1000  # vocabulário pequeno: o foco aqui é o corpo do modelo, não a saída


def _tempo_forward(modelo, idx, repeticoes: int = 5) -> float:
    """Tempo médio (ms) de um forward, descartando a primeira chamada (warm-up)."""
    modelo(idx)
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        modelo(idx)
    return (time.perf_counter() - inicio) * 1000 / repeticoes


@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 15 — EXPERIMENTAÇÃO DE ARQUITETURA")
    torch.manual_seed(123)

    subtitulo("15.1 Número de blocos (Transformer Blocks)")
    emb_dim, heads, ctx = 64, 4, 32
    idx = torch.randint(0, VOCAB, (4, ctx))
    linhas = []
    for n_layers in (1, 2, 4, 8, 12):
        modelo = GPTModel(VOCAB, ctx, emb_dim, n_layers, heads, drop_rate=0.0)
        modelo.eval()
        ms = _tempo_forward(modelo, idx)
        linhas.append([n_layers, f"{sum(p.numel() for p in modelo.parameters()):,}",
                        f"{memoria_em_mb(modelo):.2f} MB", f"{ms:.2f} ms"])
    tabela(["n_layers", "parâmetros", "memória", "tempo de forward"], linhas)
    print("mais blocos = mais parâmetros e mais tempo, ambos de forma aproximadamente "
          "linear em n_layers — cada bloco adicional custa sempre o mesmo (12·emb_dim² "
          "+ 10·emb_dim parâmetros, visto no exp14).")

    subtitulo("15.2 Número de cabeças (emb_dim fixo em 64)")
    linhas = []
    for n_heads in (1, 2, 4, 8, 16):
        modelo = GPTModel(VOCAB, ctx, emb_dim, n_layers=2, num_heads=n_heads, drop_rate=0.0)
        modelo.eval()
        ms = _tempo_forward(modelo, idx)
        linhas.append([n_heads, emb_dim // n_heads,
                        f"{sum(p.numel() for p in modelo.parameters()):,}", f"{ms:.2f} ms"])
    tabela(["num_heads", "head_dim", "parâmetros", "tempo de forward"], linhas)
    print("assim como medido para a MultiHeadAttention isolada na Sprint 3, o número "
          "de parâmetros NÃO muda com num_heads — só a forma como as mesmas dimensões "
          "são divididas em cabeças.")

    subtitulo("15.3 Dimensão do embedding (crescimento quadrático)")
    linhas = []
    for emb in (32, 64, 128, 256, 512):
        modelo = GPTModel(VOCAB, ctx, emb, n_layers=2, num_heads=4, drop_rate=0.0)
        modelo.eval()
        total = sum(p.numel() for p in modelo.parameters())
        linhas.append([emb, f"{total:,}", f"{memoria_em_mb(modelo):.2f} MB"])
    tabela(["emb_dim", "parâmetros", "memória"], linhas)
    razao = linhas[-1][1].replace(",", "")
    razao_inicial = linhas[0][1].replace(",", "")
    print(f"emb_dim cresceu 16× (32 -> 512); parâmetros cresceram "
          f"{int(razao) / int(razao_inicial):.0f}× — o corpo do modelo (blocos "
          "Transformer) domina o total e escala com emb_dim², o mesmo padrão já "
          "visto nos relatórios das Sprints 2 e 3.")

    subtitulo("15.4 Taxa de dropout: não muda parâmetros, muda a saída em treino")
    linhas = []
    modelo_ref = GPTModel(VOCAB, ctx, emb_dim, n_layers=2, num_heads=heads, drop_rate=0.0)
    params_ref = sum(p.numel() for p in modelo_ref.parameters())
    for taxa in (0.0, 0.1, 0.3, 0.5):
        modelo = GPTModel(VOCAB, ctx, emb_dim, n_layers=2, num_heads=heads, drop_rate=taxa)
        modelo.train()
        torch.manual_seed(7)
        saidas = torch.stack([modelo(idx[:1]) for _ in range(10)])
        linhas.append([
            f"{taxa:.0%}",
            f"{sum(p.numel() for p in modelo.parameters()):,}",
            f"{float(saidas.var(dim=0).mean()):.4f}",
        ])
    tabela(["drop_rate", "parâmetros (igual em todas)", "variância dos logits entre chamadas"], linhas)
    print(f"parâmetros do modelo de referência: {params_ref:,} — idêntico nas 4 linhas acima; "
          "dropout é um comportamento de execução, não uma mudança estrutural.")

    subtitulo("15.5 Tamanho do contexto")
    linhas = []
    tempos = []
    contextos = (8, 16, 32, 64, 128)
    for c in contextos:
        modelo = GPTModel(VOCAB, c, emb_dim, n_layers=2, num_heads=heads, drop_rate=0.0)
        modelo.eval()
        entrada = torch.randint(0, VOCAB, (4, c))
        ms = _tempo_forward(modelo, entrada)
        tempos.append(ms)
        linhas.append([c, f"{sum(p.numel() for p in modelo.parameters()):,}", f"{ms:.2f} ms"])
    tabela(["context_length", "parâmetros", "tempo de forward"], linhas)
    print("o contexto só afeta diretamente os positional embeddings (context_length × "
          "emb_dim parâmetros) — um crescimento pequeno perto do total do modelo —, mas "
          "domina o CUSTO COMPUTACIONAL do forward, pelo termo T² da atenção (Sprint 3).")

    subtitulo("15.6 Figuras")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_layers_vals = [1, 2, 4, 8, 12]
        params_por_camada = []
        for n in n_layers_vals:
            m = GPTModel(VOCAB, ctx, emb_dim, n, heads, drop_rate=0.0)
            params_por_camada.append(sum(p.numel() for p in m.parameters()) / 1e6)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(n_layers_vals, params_por_camada, "o-", color="#1C5D86", linewidth=2)
        ax1.set_xlabel("n_layers")
        ax1.set_ylabel("parâmetros (milhões)")
        ax1.set_title("Parâmetros cresce ~linearmente com n_layers")
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.grid(alpha=0.25, linewidth=0.7)

        ax2.plot(list(contextos), tempos, "s-", color="#9A5312", linewidth=2)
        ax2.set_xlabel("context_length")
        ax2.set_ylabel("tempo de forward (ms)")
        ax2.set_title("Tempo de forward cresce com o contexto (custo T² da atenção)")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.grid(alpha=0.25, linewidth=0.7)

        fig.tight_layout()
        salvar_figura(fig, "exp15_custo_arquitetura.png")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib não instalado — gráfico não gerado.")


if __name__ == "__main__":
    main()
