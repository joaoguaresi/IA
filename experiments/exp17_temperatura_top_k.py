"""
Experimento 17 — Temperatura e top-k (Entregável 4 / parte do 5).

Mede como as duas estratégias de amostragem alteram a distribuição do próximo
token e o texto produzido: concentração da distribuição, entropia, quantidade de
candidatos efetivos, variedade e repetição do texto gerado.
"""

import time

import torch
from _comum import salvar_figura, subtitulo, tabela, titulo
from _modelos_para_geracao import ModeloBigrama

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto
from src.geracao import (
    aplicar_top_k,
    distribuicao_do_proximo_token,
    entropia_da_distribuicao,
    gerar,
    ids_para_texto,
    taxa_de_repeticao,
    texto_para_ids,
    tokens_para_cobrir,
)

PROMPT = "O velho consertou"
TEMPERATURAS = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
TOP_KS = (1, 2, 5, 10, 50, 200)
CORES = ("#1C5D86", "#B45309", "#6B4E9B")


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 17 — TEMPERATURA E TOP-K")

    tokenizador = obter_tokenizador_bpe()
    bigrama = ModeloBigrama(tokenizador.encode(carregar_texto()), tokenizador.n_vocab)
    idx = texto_para_ids(PROMPT, tokenizador)
    logits = bigrama(idx)[:, -1, :]
    print(f"prompt: {PROMPT!r}  |  modelo: {bigrama.nome}")
    print(f"vocabulário: {tokenizador.n_vocab:,}  |  entropia máxima: "
          f"{torch.log2(torch.tensor(float(tokenizador.n_vocab))):.2f} bits")

    subtitulo("17.1 Efeito da temperatura sobre a distribuição")
    linhas = []
    for t in TEMPERATURAS:
        p = distribuicao_do_proximo_token(logits, temperatura=t)
        ordenadas, _ = torch.sort(p[0], descending=True)
        linhas.append([
            f"{t:.1f}", f"{float(p.max()):.4f}", f"{entropia_da_distribuicao(p):6.3f}",
            tokens_para_cobrir(p, 0.9), tokens_para_cobrir(p, 0.99),
            f"{float(ordenadas[0] / ordenadas[1]):.2f}x",
        ])
    tabela(["temperatura", "maior prob.", "entropia (bits)", "tokens p/ 90%",
            "tokens p/ 99%", "1º / 2º"], linhas)

    subtitulo("17.2 Os cinco candidatos sob temperaturas diferentes")
    _, ids_topo = torch.topk(logits[0], 5)
    cabecalho = ["temperatura"] + [repr(tokenizador.decode([int(i)])) for i in ids_topo]
    linhas = []
    for t in (0.1, 0.5, 1.0, 2.0, 3.0):
        p = distribuicao_do_proximo_token(logits, temperatura=t)[0]
        linhas.append([f"{t:.1f}"] + [f"{float(p[int(i)]):.4f}" for i in ids_topo])
    tabela(cabecalho, linhas)

    subtitulo("17.3 Texto gerado em cada temperatura")
    linhas = []
    for t in (0.0, 0.3, 0.7, 1.0, 1.5, 3.0):
        inicio = time.perf_counter()
        saida = gerar(bigrama, idx, 18, 16, temperatura=t, seed=42)
        ms = (time.perf_counter() - inicio) * 1000
        gerados = saida[0, idx.shape[1]:].tolist()
        linhas.append([
            "0 (greedy)" if t == 0 else f"{t:.1f}",
            repr(ids_para_texto(saida, tokenizador)[:54]),
            len(set(gerados)), f"{taxa_de_repeticao(gerados):.2f}",
            f"{ms:6.1f} ms", f"{ms / 18:5.2f} ms",
        ])
    tabela(["temperatura", "texto", "tokens distintos", "repetição",
            "18 tokens", "por token"], linhas)

    subtitulo("17.4 Efeito do top-k")
    linhas = []
    for k in TOP_KS:
        cortado = aplicar_top_k(logits, k)
        p = torch.softmax(cortado, dim=-1)
        inicio = time.perf_counter()
        saida = gerar(bigrama, idx, 18, 16, temperatura=1.0, top_k=k, seed=42)
        ms = (time.perf_counter() - inicio) * 1000
        gerados = saida[0, idx.shape[1]:].tolist()
        linhas.append([
            k, int((p[0] > 0).sum()), f"{entropia_da_distribuicao(p):6.3f}",
            f"{float(p.max()):.4f}", len(set(gerados)), f"{ms / 18:5.2f} ms",
            repr(ids_para_texto(saida, tokenizador)[:40]),
        ])
    tabela(["top_k", "tokens com prob. > 0", "entropia (bits)", "maior prob.",
            "tokens distintos", "por token", "texto"], linhas)

    subtitulo("17.5 Temperatura × top-k (entropia em bits)")
    ks = (1, 5, 50, None)
    linhas = []
    for t in (0.3, 0.7, 1.0, 2.0):
        linha = [f"{t:.1f}"]
        for k in ks:
            p = distribuicao_do_proximo_token(logits, temperatura=t, top_k=k)
            linha.append(f"{entropia_da_distribuicao(p):.3f}")
        linhas.append(linha)
    tabela(["temperatura"] + [f"top_k={k if k else 'sem'}" for k in ks], linhas)

    subtitulo("17.6 Custo computacional da escolha do token")
    repeticoes = 200
    estrategias = (
        ("greedy (argmax puro)", dict(temperatura=0.0, top_k=None)),
        ("amostragem, T = 1.0", dict(temperatura=1.0, top_k=None)),
        ("amostragem, T = 1.0 + top-k 50", dict(temperatura=1.0, top_k=50)),
        ("amostragem, T = 0.7 + top-k 5", dict(temperatura=0.7, top_k=5)),
    )
    referencia = None
    linhas = []
    for rotulo, cfg in estrategias:
        torch.manual_seed(0)
        inicio = time.perf_counter()
        for _ in range(repeticoes):
            if cfg["temperatura"] == 0.0:
                torch.argmax(logits, dim=-1, keepdim=True)
            else:
                prob = distribuicao_do_proximo_token(logits, **cfg)
                torch.multinomial(prob, num_samples=1)
        us = (time.perf_counter() - inicio) / repeticoes * 1e6
        if referencia is None:
            referencia = us
        linhas.append([rotulo, f"{us:8.1f} µs", f"{us / referencia:.2f}x"])
    tabela([f"estratégia (média de {repeticoes} escolhas)", "por escolha",
            "vs. greedy"], linhas)
    print(f"logits de uma posição: {tuple(logits.shape)}  "
          f"({logits.shape[-1]:,} candidatos)")

    subtitulo("17.7 Custo do top-k conforme k cresce")
    linhas = []
    for k in (1, 10, 50, 500, 5_000, 50_000):
        torch.manual_seed(0)
        inicio = time.perf_counter()
        for _ in range(repeticoes):
            aplicar_top_k(logits, k)
        us = (time.perf_counter() - inicio) / repeticoes * 1e6
        linhas.append([k, f"{us:8.1f} µs"])
    tabela(["top_k", "tempo do corte"], linhas)

    subtitulo("17.8 Reprodutibilidade")
    a = ids_para_texto(gerar(bigrama, idx, 12, 16, temperatura=1.2, seed=7), tokenizador)
    b = ids_para_texto(gerar(bigrama, idx, 12, 16, temperatura=1.2, seed=7), tokenizador)
    c = ids_para_texto(gerar(bigrama, idx, 12, 16, temperatura=1.2, seed=99), tokenizador)
    print(f"seed=7  (1ª vez): {a[:60]!r}")
    print(f"seed=7  (2ª vez): {b[:60]!r}")
    print(f"seed=99 .........: {c[:60]!r}")
    print(f"mesma seed produz o mesmo texto: {a == b}")
    print(f"seeds diferentes produzem textos diferentes: {a != c}")

    subtitulo("17.9 Figuras")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # (a) distribuição dos 10 candidatos mais prováveis sob três temperaturas
        _, ids10 = torch.topk(logits[0], 10)
        rotulos = [tokenizador.decode([int(i)]).strip() or "·" for i in ids10]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        largura, posicoes = 0.27, np.arange(len(ids10))
        for desloc, (t, cor) in enumerate(zip((0.5, 1.0, 2.0), CORES)):
            p = distribuicao_do_proximo_token(logits, temperatura=t)[0]
            valores = [float(p[int(i)]) for i in ids10]
            ax.bar(posicoes + (desloc - 1) * largura, valores, largura,
                   color=cor, label=f"T = {t}")
        ax.set_xticks(posicoes, rotulos, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("probabilidade")
        ax.set_title("Distribuição do próximo token sob três temperaturas")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.legend(frameon=False, fontsize=9)
        salvar_figura(fig, "exp17_distribuicao_temperatura.png")
        plt.close(fig)

        # (b) entropia em função da temperatura, com e sem top-k
        ts = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for (k, rotulo), cor in zip(((None, "sem top-k"), (10, "top-k = 10"),
                                     (3, "top-k = 3")), CORES):
            ys = [entropia_da_distribuicao(
                distribuicao_do_proximo_token(logits, temperatura=t, top_k=k)) for t in ts]
            ax.plot(ts, ys, "o-", color=cor, linewidth=2, markersize=5, label=rotulo)
        ax.set_xlabel("temperatura")
        ax.set_ylabel("entropia da distribuição (bits)")
        ax.set_title("O top-k limita o quanto a temperatura espalha a escolha")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.legend(frameon=False, fontsize=9)
        salvar_figura(fig, "exp17_entropia_temperatura.png")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib não instalado — figuras não geradas.")


if __name__ == "__main__":
    main()
