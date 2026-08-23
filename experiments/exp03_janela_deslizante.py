"""
Experimento 3 — Janela deslizante: pares (entrada, alvo) (item 3.3 do enunciado).

Investiga:
  * como o alvo é a entrada deslocada em uma posição;
  * o efeito de diferentes tamanhos de contexto (context_size);
  * o efeito do stride sobre a sobreposição e a quantidade de amostras;
  * a relação entre tamanho do contexto e quantidade de amostras de treinamento.
"""

from _comum import salvar_figura, subtitulo, tabela, titulo

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto
from src.dataset import criar_pares_janela_deslizante, numero_de_amostras


def main() -> None:
    titulo("EXPERIMENTO 3 — JANELA DESLIZANTE (PARES ENTRADA/ALVO)")
    texto = carregar_texto()
    tokenizador = obter_tokenizador_bpe()
    token_ids = tokenizador.encode(texto)
    print(f"tokenizador: {tokenizador.nome}  |  tokens no corpus: {len(token_ids):,}")

    subtitulo("3.1 A tarefa de previsão do próximo token")
    contexto = 4
    amostra = token_ids[: contexto + 1]
    entrada, alvo = amostra[:contexto], amostra[1 : contexto + 1]
    print(f"entrada (x): {entrada}")
    print(f"alvo    (y): {alvo}   <- x deslocado em 1 posição\n")
    for i in range(1, contexto + 1):
        ctx, previsto = amostra[:i], amostra[i]
        print(f"  {ctx}  ---->  {previsto}")
    print("\nEm texto:")
    for i in range(1, contexto + 1):
        ctx, previsto = amostra[:i], amostra[i]
        print(f"  {tokenizador.decode(ctx)!r}  ---->  {tokenizador.decode([previsto])!r}")

    subtitulo("3.2 Efeito do stride sobre a sobreposição")
    for stride in (1, 2, 4):
        entradas, _ = criar_pares_janela_deslizante(token_ids, max_length=4, stride=stride)
        print(f"\nstride={stride}  (primeiras 4 amostras)")
        for x in entradas[:4]:
            print(f"  {x}")
    print("\nstride=1 -> sobreposição máxima (mais amostras, dados repetidos)")
    print("stride=context_size -> sem sobreposição (menos amostras, sem repetição)")

    subtitulo("3.3 Tamanho do contexto vs. quantidade de amostras")
    contextos = [2, 4, 8, 16, 32, 64, 128, 256]
    linhas = []
    for ctx in contextos:
        if ctx >= len(token_ids):
            continue
        n_stride1 = numero_de_amostras(len(token_ids), ctx, 1)
        n_stride_ctx = numero_de_amostras(len(token_ids), ctx, ctx)
        n_stride_meio = numero_de_amostras(len(token_ids), ctx, max(1, ctx // 2))
        linhas.append(
            [
                ctx,
                f"{n_stride1:,}",
                f"{n_stride_meio:,}",
                f"{n_stride_ctx:,}",
                f"{n_stride_ctx * ctx:,}",
            ]
        )
    tabela(
        [
            "context_size",
            "amostras (stride=1)",
            "amostras (stride=ctx/2)",
            "amostras (stride=ctx)",
            "tokens vistos",
        ],
        linhas,
    )
    print("\nCom stride=1 a quantidade de amostras é praticamente independente do contexto")
    print("(N - contexto). Com stride=context_size ela cai de forma inversamente")
    print("proporcional ao contexto: dobrar o contexto reduz as amostras pela metade,")
    print("mas cada amostra passa a conter o dobro de tokens.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ctxs = [int(l[0]) for l in linhas]
        s1 = [numero_de_amostras(len(token_ids), c, 1) for c in ctxs]
        sc = [numero_de_amostras(len(token_ids), c, c) for c in ctxs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ctxs, s1, "o-", label="stride = 1")
        ax.plot(ctxs, sc, "s-", label="stride = context_size")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("tamanho do contexto (tokens)")
        ax.set_ylabel("número de amostras de treinamento")
        ax.set_title("Contexto vs. quantidade de amostras")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        salvar_figura(fig, "exp03_contexto_vs_amostras.png")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib não instalado — gráfico não gerado.")


if __name__ == "__main__":
    main()
