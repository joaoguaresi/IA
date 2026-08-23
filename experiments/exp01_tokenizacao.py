"""
Experimento 1 — Tokenização (item 3.1 e 4 do enunciado).

Objetivos:
  * observar como diferentes frases são divididas em tokens;
  * comparar a tokenização por expressão regular (palavra) com o BPE (subpalavra);
  * medir a quantidade de tokens produzida para textos diferentes.
"""

from _comum import subtitulo, tabela, titulo

from src.bpe import BPETokenizerSimples, obter_tokenizador_bpe
from src.dados import carregar_texto, nome_do_corpus
from src.tokenizer import destokenizar, tokenizar

FRASES = [
    "Olá, mundo!",
    "O relojoeiro consertou o relógio -- e não cobrou nada.",
    "Large language models processam texto uma palavra por vez.",
    "supercalifragilisticexpialidocious",
    "A inteligência artificial generativa transformará a educação em 2026.",
    "email: aluno@universidade.br  telefone: (11) 98765-4321",
]


def main() -> None:
    titulo("EXPERIMENTO 1 — TOKENIZAÇÃO")

    subtitulo("1.1 Tokenização por expressão regular (palavras e pontuação)")
    for frase in FRASES:
        tokens = tokenizar(frase)
        print(f"\ntexto  : {frase}")
        print(f"tokens : {tokens}")
        print(f"n      : {len(tokens)}")

    subtitulo("1.2 A tokenização é reversível?")
    for frase in FRASES[:3]:
        reconstruido = destokenizar(tokenizar(frase))
        igual = "sim" if reconstruido == frase else "não"
        print(f"original      : {frase}")
        print(f"reconstruído  : {reconstruido}")
        print(f"idêntico      : {igual}\n")

    subtitulo("1.3 Regex (palavra) vs. BPE (subpalavra)")
    bpe = obter_tokenizador_bpe()
    linhas = []
    for frase in FRASES:
        n_regex = len(tokenizar(frase))
        n_bpe = len(bpe.encode(frase))
        linhas.append([frase[:44], n_regex, n_bpe, f"{n_bpe / n_regex:.2f}x"])
    tabela(["texto", "tokens regex", "tokens BPE", "razão"], linhas)

    subtitulo("1.4 Como o BPE trata uma palavra desconhecida")
    palavra = "supercalifragilisticexpialidocious"
    print(f"palavra: {palavra}")
    print(f"ids    : {bpe.encode(palavra)}")
    print(f"pedaços: {bpe.tokens(palavra)}")
    print("Observação: a palavra é decomposta em subpalavras/caracteres — por isso")
    print("o tokenizador do GPT não precisa do token <|unk|>.")

    subtitulo("1.5 BPE treinado no próprio corpus (implementação didática)")
    texto = carregar_texto()
    for tamanho in (300, 500, 1000):
        tok = BPETokenizerSimples().treinar(texto, tamanho_vocab=tamanho)
        ids = tok.encode(texto)
        print(
            f"vocab={tok.n_vocab:>5}  tokens no corpus={len(ids):>6}  "
            f"compressão vs. bytes={len(texto.encode('utf-8')) / len(ids):.2f}x"
        )
    print("\nQuanto maior o vocabulário, menos tokens são necessários para o mesmo texto:")
    print("é o compromisso entre tamanho do vocabulário e comprimento da sequência.")

    subtitulo("1.6 Quantidade de tokens no corpus completo")
    tabela(
        ["corpus", "caracteres", "tokens regex", "tokens BPE"],
        [[nome_do_corpus(), len(texto), len(tokenizar(texto)), len(bpe.encode(texto))]],
    )


if __name__ == "__main__":
    main()
