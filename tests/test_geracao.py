"""Testes da inferência e geração de texto (Sprint 4, Entregável 4)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.geracao import (  # noqa: E402
    aplicar_temperatura,
    aplicar_top_k,
    distribuicao_do_proximo_token,
    entropia_da_distribuicao,
    gerar,
    gerar_passo_a_passo,
    gerar_texto,
    gerar_texto_greedy,
    ids_para_texto,
    taxa_de_repeticao,
    texto_para_ids,
    tokens_para_cobrir,
)

VOCAB = 10


class ModeloSucessor(nn.Module):
    """Modelo determinístico: prevê sempre o token seguinte ao atual (i -> i+1)."""

    def __init__(self, vocab_size: int = VOCAB) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.contextos_vistos = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.contextos_vistos.append(int(idx.shape[1]))
        b, n = idx.shape
        logits = torch.zeros(b, n, self.vocab_size)
        for linha in range(b):
            for posicao in range(n):
                alvo = (int(idx[linha, posicao]) + 1) % self.vocab_size
                logits[linha, posicao, alvo] = 10.0
        return logits


class ModeloFixo(nn.Module):
    """Devolve sempre os mesmos logits, decrescentes: [9, 8, 7, ...]."""

    def __init__(self, vocab_size: int = VOCAB) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.base = torch.tensor([float(vocab_size - i) for i in range(vocab_size)])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, n = idx.shape
        return self.base.repeat(b * n).reshape(b, n, self.vocab_size)


class TokenizadorFalso:
    """Tokenizador de brinquedo: cada caractere vira um Token ID."""

    def __init__(self) -> None:
        self.letras = "abcdefghij"
        self.n_vocab = len(self.letras)

    def encode(self, texto):
        return [self.letras.index(c) for c in texto if c in self.letras]

    def decode(self, ids):
        return "".join(self.letras[int(i)] for i in ids)


class TestTransformacoesDeLogits(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])

    def test_temperatura_preserva_a_ordem(self):
        for t in (0.1, 0.5, 2.0, 5.0):
            escalados = aplicar_temperatura(self.logits, t)
            self.assertEqual(int(torch.argmax(escalados)), int(torch.argmax(self.logits)))

    def test_temperatura_baixa_concentra(self):
        fria = torch.softmax(aplicar_temperatura(self.logits, 0.2), dim=-1)
        quente = torch.softmax(aplicar_temperatura(self.logits, 3.0), dim=-1)
        self.assertGreater(float(fria.max()), float(quente.max()))
        self.assertLess(entropia_da_distribuicao(fria), entropia_da_distribuicao(quente))

    def test_temperatura_invalida(self):
        for t in (0.0, -1.0):
            with self.assertRaises(ValueError):
                aplicar_temperatura(self.logits, t)

    def test_top_k_mantem_apenas_k_valores(self):
        cortado = aplicar_top_k(self.logits, 2)
        finitos = int((cortado > -torch.inf).sum())
        self.assertEqual(finitos, 2)
        self.assertEqual(int(torch.argmax(cortado)), int(torch.argmax(self.logits)))

    def test_top_k_zera_os_descartados(self):
        p = torch.softmax(aplicar_top_k(self.logits, 3), dim=-1)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertEqual(int((p[0] == 0).sum()), 2)

    def test_top_k_maior_que_o_vocabulario(self):
        cortado = aplicar_top_k(self.logits, 50)
        self.assertEqual(int((cortado > -torch.inf).sum()), self.logits.shape[-1])

    def test_top_k_invalido(self):
        with self.assertRaises(ValueError):
            aplicar_top_k(self.logits, 0)

    def test_distribuicao_soma_um(self):
        for t, k in ((1.0, None), (0.5, 3), (2.0, 1)):
            p = distribuicao_do_proximo_token(self.logits, temperatura=t, top_k=k)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=6)


class TestGeracaoGreedy(unittest.TestCase):
    def setUp(self):
        self.modelo = ModeloSucessor()
        self.idx = torch.tensor([[0]])

    def test_acrescenta_o_numero_pedido_de_tokens(self):
        saida = gerar_texto_greedy(self.modelo, self.idx, 5, context_size=8)
        self.assertEqual(tuple(saida.shape), (1, 6))

    def test_segue_a_previsao_do_modelo(self):
        saida = gerar_texto_greedy(self.modelo, self.idx, 4, context_size=8)
        self.assertEqual(saida[0].tolist(), [0, 1, 2, 3, 4])

    def test_e_deterministico(self):
        a = gerar_texto_greedy(self.modelo, self.idx, 6, 8)
        b = gerar_texto_greedy(self.modelo, self.idx, 6, 8)
        self.assertEqual(a.tolist(), b.tolist())

    def test_temperatura_zero_equivale_a_greedy(self):
        a = gerar_texto_greedy(self.modelo, self.idx, 5, 8)
        b = gerar(self.modelo, self.idx, 5, 8, temperatura=0.0)
        self.assertEqual(a.tolist(), b.tolist())

    def test_zero_tokens_nao_altera_a_entrada(self):
        saida = gerar(self.modelo, self.idx, 0, 8)
        self.assertEqual(saida.tolist(), self.idx.tolist())

    def test_recorta_o_contexto(self):
        modelo = ModeloSucessor()
        entrada = torch.tensor([[0, 1, 2, 3, 4, 5]])
        gerar(modelo, entrada, 4, context_size=3)
        self.assertTrue(all(n <= 3 for n in modelo.contextos_vistos))

    def test_preserva_o_prompt(self):
        entrada = torch.tensor([[7, 8]])
        saida = gerar_texto_greedy(self.modelo, entrada, 3, 8)
        self.assertEqual(saida[0, :2].tolist(), [7, 8])

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            gerar(self.modelo, self.idx, -1, 8)
        with self.assertRaises(ValueError):
            gerar(self.modelo, self.idx, 3, 0)

    def test_para_no_token_de_fim(self):
        saida = gerar(self.modelo, self.idx, 9, 8, eos_id=3)
        self.assertEqual(saida[0].tolist(), [0, 1, 2, 3])


class TestAmostragem(unittest.TestCase):
    def setUp(self):
        self.modelo = ModeloFixo()
        self.idx = torch.tensor([[0]])

    def test_top_k_1_e_deterministico(self):
        a = gerar(self.modelo, self.idx, 5, 8, temperatura=1.0, top_k=1, seed=1)
        b = gerar(self.modelo, self.idx, 5, 8, temperatura=1.0, top_k=1, seed=2)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertTrue(all(t == 0 for t in a[0, 1:].tolist()))

    def test_amostra_apenas_dentro_do_top_k(self):
        saida = gerar(self.modelo, self.idx, 40, 8, temperatura=2.0, top_k=3, seed=5)
        self.assertTrue(set(saida[0, 1:].tolist()) <= {0, 1, 2})

    def test_mesma_seed_mesmo_resultado(self):
        a = gerar(self.modelo, self.idx, 15, 8, temperatura=1.5, seed=42)
        b = gerar(self.modelo, self.idx, 15, 8, temperatura=1.5, seed=42)
        self.assertEqual(a.tolist(), b.tolist())

    def test_seeds_diferentes_dao_resultados_diferentes(self):
        a = gerar(self.modelo, self.idx, 25, 8, temperatura=2.0, seed=1)
        b = gerar(self.modelo, self.idx, 25, 8, temperatura=2.0, seed=99)
        self.assertNotEqual(a.tolist(), b.tolist())

    def test_temperatura_alta_gera_mais_variedade(self):
        fria = gerar(self.modelo, self.idx, 40, 8, temperatura=0.2, seed=3)
        quente = gerar(self.modelo, self.idx, 40, 8, temperatura=3.0, seed=3)
        self.assertGreater(len(set(quente[0].tolist())), len(set(fria[0].tolist())))


class TestMedidas(unittest.TestCase):
    def test_entropia_uniforme_e_maxima(self):
        uniforme = torch.full((1, 8), 1 / 8)
        self.assertAlmostEqual(entropia_da_distribuicao(uniforme), 3.0, places=5)

    def test_entropia_certeza_e_zero(self):
        certa = torch.zeros(1, 8)
        certa[0, 2] = 1.0
        self.assertAlmostEqual(entropia_da_distribuicao(certa), 0.0, places=5)

    def test_tokens_para_cobrir(self):
        certa = torch.zeros(10)
        certa[4] = 1.0
        self.assertEqual(tokens_para_cobrir(certa, 0.9), 1)
        self.assertEqual(tokens_para_cobrir(torch.full((10,), 0.1), 0.9), 9)

    def test_taxa_de_repeticao(self):
        self.assertAlmostEqual(taxa_de_repeticao([1, 2, 3, 4]), 0.0)
        self.assertAlmostEqual(taxa_de_repeticao([7, 7, 7, 7]), 0.75)
        self.assertAlmostEqual(taxa_de_repeticao([]), 0.0)


class TestTextoEPassoAPasso(unittest.TestCase):
    def setUp(self):
        self.tokenizador = TokenizadorFalso()
        self.modelo = ModeloSucessor(vocab_size=self.tokenizador.n_vocab)

    def test_texto_para_ids_e_volta(self):
        idx = texto_para_ids("abc", self.tokenizador)
        self.assertEqual(tuple(idx.shape), (1, 3))
        self.assertEqual(ids_para_texto(idx, self.tokenizador), "abc")

    def test_gerar_texto_ponta_a_ponta(self):
        saida = gerar_texto(self.modelo, self.tokenizador, "abc", 3, context_size=8)
        self.assertIsInstance(saida, str)
        self.assertTrue(saida.startswith("abc"))
        self.assertEqual(saida, "abcdef")

    def test_registro_passo_a_passo(self):
        idx = texto_para_ids("a", self.tokenizador)
        saida, registro = gerar_passo_a_passo(self.modelo, idx, 3, context_size=8)
        self.assertEqual(len(registro), 3)
        self.assertEqual(tuple(saida.shape), (1, 4))
        for r in registro:
            self.assertIn("token_escolhido", r)
            self.assertGreaterEqual(r["probabilidade"], 0.0)
            self.assertLessEqual(len(r["top5_ids"]), 5)


if __name__ == "__main__":
    unittest.main()
