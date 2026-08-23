"""
Experimento 5 — Token embeddings e Positional embeddings (itens 3.4 e 3.5).

Investiga:
  * a transformação Token ID (inteiro) -> vetor de `output_dim` dimensões;
  * o impacto da dimensão do embedding sobre parâmetros e memória;
  * a prova experimental de que, sem positional embeddings, o mesmo token
    recebe exatamente a mesma representação em qualquer posição.
"""

from _comum import salvar_figura, subtitulo, tabela, titulo

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto
from src.dataset import create_dataloader_v1
from src.embeddings import GPTEmbedding, comparar_posicoes, embedding_de_token, tamanho_em_mb


def main() -> None:
    titulo("EXPERIMENTO 5 — EMBEDDINGS E POSITIONAL EMBEDDINGS")
    texto = carregar_texto()
    tokenizador = obter_tokenizador_bpe()
    vocab_size = tokenizador.n_vocab

    subtitulo("5.1 De Token ID para vetor")
    context_length, output_dim = 4, 8
    camada = GPTEmbedding(vocab_size, output_dim, context_length, seed=123)
    token_id = tokenizador.encode("relógio")[0]
    vetor = embedding_de_token(camada, token_id)
    print(f"token ID {token_id} -> vetor de dimensão {tuple(vetor.shape)}:")
    print(vetor)
    print("\nO inteiro deixa de ser um rótulo arbitrário e passa a ser um ponto")
    print("num espaço contínuo de", output_dim, "dimensões, ajustável por gradiente.")

    subtitulo("5.2 A matriz de embedding é uma tabela de consulta (lookup)")
    matriz = camada.tok_emb.weight
    print(f"forma da matriz de token embeddings: {tuple(matriz.shape)}  (vocab_size, output_dim)")
    print("A linha de índice i é o embedding do token de ID i — não há multiplicação")
    print("de matriz envolvida, apenas uma indexação.")

    subtitulo("5.3 Dimensão do embedding: parâmetros e memória")
    linhas = []
    for dim in (8, 64, 128, 256, 768, 1024):
        n_tok = vocab_size * dim
        n_pos = 256 * dim
        linhas.append(
            [
                dim,
                f"{n_tok:,}",
                f"{n_pos:,}",
                f"{n_tok + n_pos:,}",
                f"{tamanho_em_mb(vocab_size, dim):.2f} MB",
            ]
        )
    tabela(
        [
            "output_dim",
            "params token emb.",
            "params pos. emb. (ctx=256)",
            "total",
            "memória (float32)",
        ],
        linhas,
    )
    print(f"\nvocab_size usado: {vocab_size:,} ({tokenizador.nome})")
    print("A dimensão do embedding multiplica linearmente o número de parâmetros e a")
    print("memória de TODAS as estruturas seguintes do modelo (Q, K, V, feed-forward).")
    print("GPT-2 usa 768 dimensões; GPT-3 usa 12.288.")

    subtitulo("5.4 Formas dos tensores ao longo do pipeline")
    linhas = []
    for batch_size in (4, 8):
        for context_length in (4, 16, 64):
            for output_dim in (8, 256):
                dl = create_dataloader_v1(
                    texto,
                    tokenizer=tokenizador,
                    batch_size=batch_size,
                    max_length=context_length,
                    stride=context_length,
                    shuffle=False,
                )
                entradas, _ = next(iter(dl))
                camada = GPTEmbedding(vocab_size, output_dim, context_length, seed=123)
                saida = camada(entradas)
                n_valores = 1
                for d in saida.shape:
                    n_valores *= int(d)
                linhas.append(
                    [
                        batch_size,
                        context_length,
                        output_dim,
                        str(tuple(entradas.shape)),
                        str(tuple(saida.shape)),
                        f"{n_valores:,}",
                        f"{n_valores * 4 / 1024:.1f} KB",
                    ]
                )
    tabela(
        [
            "batch",
            "contexto",
            "dim",
            "lote de IDs",
            "input embeddings",
            "valores",
            "memória do tensor",
        ],
        linhas,
    )

    subtitulo("5.5 Por que precisamos de positional embeddings")
    context_length, output_dim = 8, 16
    camada = GPTEmbedding(vocab_size, output_dim, context_length, seed=123)
    posicoes = [0, 1, 5]
    vetores, distancias = comparar_posicoes(camada, token_id, posicoes)

    print(f"O MESMO token (ID {token_id}) repetido em várias posições da sequência.\n")
    print("Token embedding puro (sem posição) — 5 primeiras dimensões:")
    vetor_puro = embedding_de_token(camada, token_id)
    for p in posicoes:
        print(f"  posição {p}: {[round(float(v), 4) for v in vetor_puro[:5]]}")
    print("  -> idêntico em todas as posições: o self-attention não saberia a ordem.\n")

    print("Input embedding (token + positional) — 5 primeiras dimensões:")
    for p in posicoes:
        print(f"  posição {p}: {[round(float(v), 4) for v in vetores[p][:5]]}")
    print("\nDistância euclidiana entre as representações do mesmo token:")
    for (a, b), d in distancias.items():
        print(f"  posição {a} vs. posição {b}: {d:.4f}")
    print("\nA distância é maior que zero: a posição passou a fazer parte da representação.")

    subtitulo("5.6 Soma final: token embeddings + positional embeddings")
    dl = create_dataloader_v1(
        texto, tokenizer=tokenizador, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    entradas, _ = next(iter(dl))
    camada = GPTEmbedding(vocab_size, 256, 4, seed=123)
    tok = camada.tok_emb(entradas)
    saida = camada(entradas)
    print(f"lote de Token IDs ....... {tuple(entradas.shape)}")
    print(f"token embeddings ........ {tuple(tok.shape)}")
    print(f"positional embeddings ... {tuple(camada.pos_emb.weight.shape)} (somados por broadcast)")
    print(f"input embeddings ........ {tuple(saida.shape)}  <- entrada do bloco de atenção")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dims = [8, 64, 128, 256, 768, 1024]
        mem = [tamanho_em_mb(vocab_size, d) for d in dims]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(dims, mem, "o-")
        ax.set_xlabel("dimensão do embedding")
        ax.set_ylabel("memória da matriz de token embeddings (MB)")
        ax.set_title(f"Custo do embedding (vocab_size = {vocab_size:,})")
        ax.grid(alpha=0.3)
        salvar_figura(fig, "exp05_dimensao_vs_memoria.png")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib não instalado — gráfico não gerado.")


if __name__ == "__main__":
    main()
