"""
Experimento 2 — Vocabulário e Token IDs (item 3.2 do enunciado).

Demonstra a relação:  Token  <->  Token ID  <->  Vocabulário
"""

from _comum import subtitulo, tabela, titulo

from src.dados import carregar_texto, nome_do_corpus
from src.tokenizer import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    construir_vocabulario,
    inverter_vocabulario,
    juntar_textos,
    tokenizar,
)


def main() -> None:
    titulo("EXPERIMENTO 2 — VOCABULÁRIO E TOKEN IDs")
    texto = carregar_texto()

    subtitulo("2.1 Construção do vocabulário")
    tokens = tokenizar(texto)
    vocabulario = construir_vocabulario(texto)
    print(f"corpus ................ {nome_do_corpus()}")
    print(f"tokens totais ......... {len(tokens):,}")
    print(f"tokens únicos ......... {len(set(tokens)):,}")
    print(f"tamanho do vocabulário  {len(vocabulario):,} (inclui tokens especiais)")
    print("\nPrimeiras 10 entradas do vocabulário (token -> ID):")
    for token, indice in list(vocabulario.items())[:10]:
        print(f"  {token!r:<18} -> {indice}")
    print("\nÚltimas 3 entradas (tokens especiais):")
    for token, indice in list(vocabulario.items())[-3:]:
        print(f"  {token!r:<18} -> {indice}")

    subtitulo("2.2 encode / decode (ida e volta)")
    tokenizador = SimpleTokenizerV1(vocabulario)
    frase = "O velho consertou o relógio."
    ids = tokenizador.encode(frase)
    reconstruido = tokenizador.decode(ids)
    tabela(
        ["token", "token ID"],
        [[t, i] for t, i in zip(tokenizar(frase), ids)],
    )
    print(f"\ndecode(ids) = {reconstruido!r}")
    print(f"ida e volta preservou o texto: {reconstruido == frase}")

    subtitulo("2.3 Vocabulário inverso (Token ID -> Token)")
    inverso = inverter_vocabulario(vocabulario)
    for i in ids[:5]:
        print(f"  {i:>4} -> {inverso[i]!r}")

    subtitulo("2.4 Palavra fora do vocabulário: V1 falha, V2 usa <|unk|>")
    desconhecida = "O velho consertou o helicóptero."
    try:
        tokenizador.encode(desconhecida)
    except KeyError as erro:
        print(f"SimpleTokenizerV1 -> KeyError: {erro}")

    tokenizador_v2 = SimpleTokenizerV2(vocabulario)
    ids_v2 = tokenizador_v2.encode(desconhecida)
    print(f"SimpleTokenizerV2 -> {ids_v2}")
    print(f"decode            -> {tokenizador_v2.decode(ids_v2)!r}")

    subtitulo("2.5 Token especial <|endoftext|> separando textos independentes")
    juntos = juntar_textos("O velho fechou o livro.", "A biblioteca fechou.")
    ids_juntos = tokenizador_v2.encode(juntos)
    print(f"texto : {juntos}")
    print(f"ids   : {ids_juntos}")
    print(f"decode: {tokenizador_v2.decode(ids_juntos)!r}")

    subtitulo("2.6 Crescimento do vocabulário com o tamanho do corpus")
    linhas = []
    for fracao in (0.1, 0.25, 0.5, 0.75, 1.0):
        parcial = texto[: int(len(texto) * fracao)]
        n_tokens = len(tokenizar(parcial))
        n_vocab = len(construir_vocabulario(parcial))
        linhas.append(
            [f"{fracao:.0%}", n_tokens, n_vocab, f"{n_vocab / max(n_tokens, 1):.3f}"]
        )
    tabela(["fração do corpus", "tokens", "vocabulário", "vocab/tokens"], linhas)
    print("\nO vocabulário cresce mais devagar que o corpus (lei de Heaps): tokens novos")
    print("ficam cada vez mais raros, mas nunca deixam de aparecer — o que motiva o BPE.")


if __name__ == "__main__":
    main()
