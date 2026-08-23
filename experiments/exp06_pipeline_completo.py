"""
Experimento 6 — Pipeline completo texto -> entrada do modelo.

Reproduz o fluxo exigido no enunciado com diferentes configurações e mostra
que a saída final tem sempre a forma (batch_size, context_length, output_dim),
que é exatamente a entrada esperada pelo mecanismo de atenção da Sprint 3.
"""

from _comum import subtitulo, tabela, titulo

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto, nome_do_corpus
from src.pipeline import executar_pipeline


def main() -> None:
    titulo("EXPERIMENTO 6 — PIPELINE COMPLETO")
    texto = carregar_texto()
    tokenizador = obter_tokenizador_bpe()
    print(f"corpus: {nome_do_corpus()}  |  tokenizador: {tokenizador.nome}")

    subtitulo("6.1 Configuração de referência do capítulo (dim=256, contexto=4)")
    resultado = executar_pipeline(
        texto,
        context_length=4,
        stride=4,
        batch_size=8,
        output_dim=256,
        tokenizer=tokenizador,
    )
    resultado.imprimir()

    subtitulo("6.2 Varredura de configurações")
    linhas = []
    configuracoes = [
        # (contexto, stride, batch, dim)
        (4, 4, 8, 256),
        (8, 8, 8, 256),
        (16, 16, 4, 256),
        (64, 64, 4, 256),
        (256, 256, 2, 256),
        (16, 16, 4, 64),
        (16, 16, 4, 768),
        (16, 8, 4, 256),
        (16, 1, 4, 256),
    ]
    for contexto, stride, batch, dim in configuracoes:
        r = executar_pipeline(
            texto,
            context_length=contexto,
            stride=stride,
            batch_size=batch,
            output_dim=dim,
            tokenizer=tokenizador,
        )
        linhas.append(
            [
                contexto,
                stride,
                batch,
                dim,
                f"{r.n_amostras:,}",
                f"{r.n_lotes:,}",
                str(r.forma_input_embeddings),
                f"{r.parametros_embedding:,}",
                f"{r.tempo_segundos * 1000:.0f} ms",
            ]
        )
    tabela(
        [
            "contexto",
            "stride",
            "batch",
            "dim",
            "amostras",
            "lotes",
            "input embeddings",
            "params emb.",
            "tempo",
        ],
        linhas,
    )

    subtitulo("6.3 O que segue para a Sprint 3")
    print("O tensor de input embeddings (batch, contexto, dim) é a entrada direta do")
    print("mecanismo de self-attention. Nele:")
    print("  * 'batch'    -> sequências processadas em paralelo;")
    print("  * 'contexto' -> número de posições que podem atender umas às outras;")
    print("  * 'dim'      -> d_in das matrizes Wq, Wk e Wv (queries, keys, values).")
    print("Os alvos (targets) produzidos pela janela deslizante serão usados na função")
    print("de perda quando o modelo completo for treinado.")


if __name__ == "__main__":
    main()
