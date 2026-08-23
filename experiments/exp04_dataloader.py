"""
Experimento 4 — DataLoader e organização em lotes (item 3.6 do enunciado).

Investiga:
  * a forma dos tensores produzidos: (batch_size, context_length);
  * o efeito de diferentes tamanhos de lote;
  * a relação entre stride e sobreposição entre lotes consecutivos;
  * o custo (tempo) de percorrer o dataset com configurações diferentes.
"""

import time

from _comum import subtitulo, tabela, titulo

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto
from src.dataset import create_dataloader_v1


def main() -> None:
    titulo("EXPERIMENTO 4 — DATALOADER E LOTES")
    texto = carregar_texto()
    tokenizador = obter_tokenizador_bpe()

    subtitulo("4.1 Um lote com batch_size=1 e context_length=4 (sem embaralhar)")
    dl = create_dataloader_v1(
        texto, tokenizer=tokenizador, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    it = iter(dl)
    primeiro = next(it)
    segundo = next(it)
    print(f"lote 1 -> entradas: {primeiro[0].tolist()}  alvos: {primeiro[1].tolist()}")
    print(f"lote 2 -> entradas: {segundo[0].tolist()}  alvos: {segundo[1].tolist()}")
    print("Com stride=1, o lote 2 é o lote 1 deslocado em uma posição.")

    subtitulo("4.2 stride = context_length elimina a sobreposição")
    dl = create_dataloader_v1(
        texto, tokenizer=tokenizador, batch_size=2, max_length=4, stride=4, shuffle=False
    )
    entradas, alvos = next(iter(dl))
    print(f"entradas:\n{entradas}")
    print(f"alvos:\n{alvos}")
    print(f"formas: entradas {tuple(entradas.shape)}  alvos {tuple(alvos.shape)}")

    subtitulo("4.3 Efeito do tamanho do lote e do contexto sobre as estruturas")
    linhas = []
    for batch_size in (1, 2, 4, 8, 16):
        for context_length in (4, 16, 64, 256):
            inicio = time.perf_counter()
            dl = create_dataloader_v1(
                texto,
                tokenizer=tokenizador,
                batch_size=batch_size,
                max_length=context_length,
                stride=context_length,
                shuffle=False,
                drop_last=True,
            )
            n_lotes = len(dl)
            if n_lotes == 0:
                continue
            entradas, _ = next(iter(dl))
            tempo = time.perf_counter() - inicio
            n_amostras = len(dl.dataset)
            linhas.append(
                [
                    batch_size,
                    context_length,
                    f"{n_amostras:,}",
                    f"{n_lotes:,}",
                    str(tuple(entradas.shape)),
                    batch_size * context_length,
                    f"{tempo * 1000:.1f} ms",
                ]
            )
    tabela(
        [
            "batch_size",
            "context_length",
            "amostras",
            "lotes",
            "forma do lote",
            "tokens/lote",
            "tempo de criação",
        ],
        linhas,
    )
    print("\nO número de tokens processados por lote é batch_size * context_length —")
    print("é esse produto que determina a memória exigida em cada passo de treinamento.")

    subtitulo("4.4 Percorrendo o dataset completo (custo de iteração)")
    for batch_size in (1, 8, 32):
        dl = create_dataloader_v1(
            texto,
            tokenizer=tokenizador,
            batch_size=batch_size,
            max_length=64,
            stride=64,
            shuffle=True,
        )
        inicio = time.perf_counter()
        total = sum(1 for _ in dl)
        tempo = time.perf_counter() - inicio
        print(
            f"batch_size={batch_size:>3}  lotes={total:>4}  "
            f"tempo total={tempo * 1000:7.1f} ms  tempo/lote={tempo / max(total, 1) * 1000:.2f} ms"
        )
    print("\nLotes maiores reduzem o número de iterações (menos overhead por token),")
    print("ao custo de mais memória por passo.")

    subtitulo("4.5 drop_last=True vs. drop_last=False")
    for drop_last in (True, False):
        dl = create_dataloader_v1(
            texto,
            tokenizer=tokenizador,
            batch_size=16,
            max_length=64,
            stride=64,
            shuffle=False,
            drop_last=drop_last,
        )
        tamanhos = [len(x) for x, _ in dl]
        print(f"drop_last={drop_last!s:<5} -> lotes={len(tamanhos)}  tamanhos={tamanhos}")
    print("\ndrop_last=True descarta o último lote incompleto, evitando picos de loss.")


if __name__ == "__main__":
    main()
