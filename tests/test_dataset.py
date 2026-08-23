"""Testes da janela deslizante, do dataset e do DataLoader."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.bpe import obter_tokenizador_bpe  # noqa: E402
from src.dados import carregar_texto  # noqa: E402
from src.dataset import (  # noqa: E402
    GPTDatasetV1,
    create_dataloader_v1,
    criar_pares_janela_deslizante,
    numero_de_amostras,
)


class TestJanelaDeslizante(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(100))

    def test_alvo_e_entrada_deslocada(self):
        entradas, alvos = criar_pares_janela_deslizante(self.ids, max_length=4, stride=1)
        for x, y in zip(entradas[:10], alvos[:10]):
            self.assertEqual(y, [v + 1 for v in x])

    def test_formas(self):
        entradas, alvos = criar_pares_janela_deslizante(self.ids, max_length=8, stride=8)
        self.assertTrue(all(len(x) == 8 for x in entradas))
        self.assertEqual(len(entradas), len(alvos))

    def test_stride_controla_sobreposicao(self):
        e1, _ = criar_pares_janela_deslizante(self.ids, max_length=4, stride=1)
        e4, _ = criar_pares_janela_deslizante(self.ids, max_length=4, stride=4)
        self.assertGreater(len(e1), len(e4))
        self.assertEqual(e4[1][0], e4[0][0] + 4)

    def test_contagem_bate_com_formula(self):
        for max_length in (2, 4, 16):
            for stride in (1, 2, 4):
                entradas, _ = criar_pares_janela_deslizante(self.ids, max_length, stride)
                self.assertEqual(
                    len(entradas), numero_de_amostras(len(self.ids), max_length, stride)
                )

    def test_texto_curto_levanta_erro(self):
        with self.assertRaises(ValueError):
            criar_pares_janela_deslizante([1, 2, 3], max_length=10, stride=1)


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texto = carregar_texto()
        cls.tokenizador = obter_tokenizador_bpe()

    def test_forma_do_lote(self):
        dl = create_dataloader_v1(
            self.texto,
            tokenizer=self.tokenizador,
            batch_size=4,
            max_length=8,
            stride=8,
            shuffle=False,
        )
        entradas, alvos = next(iter(dl))
        self.assertEqual(tuple(entradas.shape), (4, 8))
        self.assertEqual(tuple(alvos.shape), (4, 8))

    def test_alvo_deslocado_no_lote(self):
        dl = create_dataloader_v1(
            self.texto,
            tokenizer=self.tokenizador,
            batch_size=2,
            max_length=6,
            stride=6,
            shuffle=False,
        )
        entradas, alvos = next(iter(dl))
        self.assertEqual(entradas[0][1:].tolist(), alvos[0][:-1].tolist())

    def test_drop_last(self):
        comum = dict(
            tokenizer=self.tokenizador, max_length=16, stride=16, shuffle=False, batch_size=7
        )
        com = create_dataloader_v1(self.texto, drop_last=True, **comum)
        sem = create_dataloader_v1(self.texto, drop_last=False, **comum)
        self.assertLessEqual(len(com), len(sem))
        self.assertTrue(all(len(x) == 7 for x, _ in com))

    def test_dataset_len(self):
        ds = GPTDatasetV1(self.texto, self.tokenizador, max_length=16, stride=16)
        self.assertEqual(
            len(ds), numero_de_amostras(ds.n_tokens, 16, 16)
        )


if __name__ == "__main__":
    unittest.main()
