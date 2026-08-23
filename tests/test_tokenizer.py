"""Testes da tokenização, do vocabulário e dos Token IDs.

Executar com:  python -m unittest discover -s tests   (ou pytest tests/)
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.bpe import BPETokenizerSimples  # noqa: E402
from src.tokenizer import (  # noqa: E402
    TOKEN_DESCONHECIDO,
    TOKEN_FIM_DE_TEXTO,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    construir_vocabulario,
    destokenizar,
    inverter_vocabulario,
    tokenizar,
)

TEXTO = "O velho consertou o relógio, mas não cobrou nada. A loja fechou."


class TestTokenizacao(unittest.TestCase):
    def test_separa_pontuacao(self):
        self.assertEqual(tokenizar("Olá, mundo!"), ["Olá", ",", "mundo", "!"])

    def test_preserva_maiusculas(self):
        self.assertIn("O", tokenizar(TEXTO))
        self.assertIn("o", tokenizar(TEXTO))

    def test_travessao_duplo(self):
        self.assertEqual(tokenizar("a -- b"), ["a", "--", "b"])

    def test_destokenizar_reconstroi(self):
        self.assertEqual(destokenizar(tokenizar(TEXTO)), TEXTO)

    def test_texto_vazio(self):
        self.assertEqual(tokenizar(""), [])


class TestVocabulario(unittest.TestCase):
    def setUp(self):
        self.vocab = construir_vocabulario(TEXTO)

    def test_contem_tokens_especiais(self):
        self.assertIn(TOKEN_FIM_DE_TEXTO, self.vocab)
        self.assertIn(TOKEN_DESCONHECIDO, self.vocab)

    def test_ids_sao_unicos_e_sequenciais(self):
        ids = sorted(self.vocab.values())
        self.assertEqual(ids, list(range(len(self.vocab))))

    def test_inversao(self):
        inverso = inverter_vocabulario(self.vocab)
        for token, indice in self.vocab.items():
            self.assertEqual(inverso[indice], token)


class TestSimpleTokenizer(unittest.TestCase):
    def setUp(self):
        self.vocab = construir_vocabulario(TEXTO)

    def test_ida_e_volta(self):
        tok = SimpleTokenizerV1(self.vocab)
        frase = "A loja fechou."
        self.assertEqual(tok.decode(tok.encode(frase)), frase)

    def test_v1_falha_com_desconhecido(self):
        tok = SimpleTokenizerV1(self.vocab)
        with self.assertRaises(KeyError):
            tok.encode("palavra inexistente xyz")

    def test_v2_usa_unk(self):
        tok = SimpleTokenizerV2(self.vocab)
        ids = tok.encode("xyzabc")
        self.assertEqual(ids, [self.vocab[TOKEN_DESCONHECIDO]])
        self.assertEqual(tok.decode(ids), TOKEN_DESCONHECIDO)

    def test_n_vocab(self):
        tok = SimpleTokenizerV2(self.vocab)
        self.assertEqual(tok.n_vocab, len(self.vocab))


class TestBPESimples(unittest.TestCase):
    def setUp(self):
        self.tok = BPETokenizerSimples().treinar(TEXTO * 20, tamanho_vocab=320)

    def test_ida_e_volta_com_palavra_nova(self):
        frase = "palavra jamais vista no treinamento"
        self.assertEqual(self.tok.decode(self.tok.encode(frase)), frase)

    def test_reduz_numero_de_tokens(self):
        n_bytes = len(TEXTO.encode("utf-8"))
        self.assertLess(len(self.tok.encode(TEXTO)), n_bytes)

    def test_endoftext(self):
        texto = f"a{TOKEN_FIM_DE_TEXTO}b"
        self.assertIn(self.tok.id_fim_de_texto, self.tok.encode(texto))
        self.assertEqual(self.tok.decode(self.tok.encode(texto)), texto)


if __name__ == "__main__":
    unittest.main()
