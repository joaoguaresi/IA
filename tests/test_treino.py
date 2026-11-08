"""Testes de avaliação e treinamento do modelo GPT (Sprint 5, Entregáveis 2 e 3)."""

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.bpe import obter_tokenizador_bpe  # noqa: E402
from src.transformer import GPTModel  # noqa: E402
from src.treino import (  # noqa: E402
    avaliar_modelo,
    calcular_perda_lote,
    calcular_perda_loader,
    gerar_e_imprimir_amostra,
    perplexidade,
    treinar_modelo,
)


class ModeloLogitsFixos(nn.Module):
    """Sempre devolve os mesmos logits (determinístico), para checar a perda na mão."""

    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # favorece fortemente o token 0: cross-entropy baixa quando o alvo é 0
        self.logits_base = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, n = idx.shape
        return self.logits_base.repeat(b, n, 1)


def _dataloader_trivial(vocab_size: int, tamanho: int, repeticoes: int, batch_size: int) -> DataLoader:
    """Dataset sintético: a sequência [0, 1, ..., tamanho-1] repetida, fácil de memorizar."""
    seq = torch.arange(tamanho) % vocab_size
    entradas = seq[:-1].unsqueeze(0).repeat(repeticoes, 1)
    alvos = seq[1:].unsqueeze(0).repeat(repeticoes, 1)
    return DataLoader(TensorDataset(entradas, alvos), batch_size=batch_size, shuffle=True)


class TestCalcularPerda(unittest.TestCase):
    def setUp(self):
        self.modelo = ModeloLogitsFixos(vocab_size=5)

    def test_perda_bate_com_cross_entropy_manual(self):
        entradas = torch.zeros(2, 3, dtype=torch.long)
        alvos = torch.zeros(2, 3, dtype=torch.long)  # alvo sempre 0 -> perda baixa
        perda = calcular_perda_lote(entradas, alvos, self.modelo)
        logits = self.modelo(entradas)
        esperado = torch.nn.functional.cross_entropy(logits.flatten(0, 1), alvos.flatten())
        self.assertAlmostEqual(float(perda), float(esperado), places=5)

    def test_perda_maior_quando_alvo_e_improvavel(self):
        entradas = torch.zeros(1, 3, dtype=torch.long)
        alvo_provavel = torch.zeros(1, 3, dtype=torch.long)
        alvo_improvavel = torch.full((1, 3), 1, dtype=torch.long)
        perda_facil = calcular_perda_lote(entradas, alvo_provavel, self.modelo)
        perda_dificil = calcular_perda_lote(entradas, alvo_improvavel, self.modelo)
        self.assertLess(float(perda_facil), float(perda_dificil))

    def test_calcular_perda_loader_e_nao_negativa(self):
        loader = _dataloader_trivial(vocab_size=5, tamanho=4, repeticoes=4, batch_size=2)
        perda = calcular_perda_loader(loader, self.modelo)
        self.assertGreaterEqual(perda, 0.0)

    def test_loader_vazio_devolve_nan(self):
        vazio = DataLoader(TensorDataset(torch.empty(0, 3, dtype=torch.long), torch.empty(0, 3, dtype=torch.long)))
        self.assertTrue(math.isnan(calcular_perda_loader(vazio, self.modelo)))

    def test_num_lotes_limita_a_media(self):
        # com este modelo determinístico, a perda é igual em qualquer lote ou sublote
        loader = _dataloader_trivial(vocab_size=5, tamanho=4, repeticoes=10, batch_size=1)
        perda_parcial = calcular_perda_loader(loader, self.modelo, num_lotes=2)
        perda_total = calcular_perda_loader(loader, self.modelo)
        self.assertAlmostEqual(perda_parcial, perda_total, places=5)


class TestPerplexidade(unittest.TestCase):
    def test_perda_zero_da_perplexidade_um(self):
        self.assertAlmostEqual(perplexidade(0.0), 1.0, places=5)

    def test_cresce_com_a_perda(self):
        self.assertLess(perplexidade(1.0), perplexidade(5.0))

    def test_bate_com_exponencial_da_perda(self):
        self.assertAlmostEqual(perplexidade(2.0), math.exp(2.0), places=4)


class TestAvaliarModelo(unittest.TestCase):
    def test_devolve_par_de_perdas_e_restaura_modo_treino(self):
        modelo = ModeloLogitsFixos(vocab_size=5)
        modelo.train()
        loader = _dataloader_trivial(vocab_size=5, tamanho=4, repeticoes=4, batch_size=2)
        p_treino, p_val = avaliar_modelo(modelo, loader, loader)
        self.assertIsInstance(p_treino, float)
        self.assertIsInstance(p_val, float)
        self.assertTrue(modelo.training)  # avaliar_modelo deve devolver o modelo a train()


class TestGerarEImprimirAmostra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizador = obter_tokenizador_bpe()

    def test_devolve_texto_e_restaura_modo_treino(self):
        torch.manual_seed(0)
        modelo = GPTModel(
            vocab_size=self.tokenizador.n_vocab, context_length=8,
            emb_dim=16, n_layers=1, num_heads=2, drop_rate=0.0,
        )
        modelo.train()
        texto = gerar_e_imprimir_amostra(modelo, self.tokenizador, device=None, contexto_inicial="o velho")
        self.assertIsInstance(texto, str)
        self.assertGreater(len(texto), 0)
        self.assertTrue(modelo.training)
        self.assertNotIn("\n", texto)  # gerar_e_imprimir_amostra troca quebras de linha por espaço


class TestTreinarModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizador = obter_tokenizador_bpe()

    def test_perda_de_treino_diminui_apos_algumas_epocas(self):
        torch.manual_seed(0)
        texto = "o velho consertou o relogio quebrado da vovo " * 30
        ids = self.tokenizador.encode(texto)
        janela = 8
        pares = [
            (ids[i:i + janela], ids[i + 1:i + 1 + janela])
            for i in range(0, len(ids) - janela - 1, janela)
        ]
        entradas = torch.tensor([p[0] for p in pares], dtype=torch.long)
        alvos = torch.tensor([p[1] for p in pares], dtype=torch.long)
        loader = DataLoader(TensorDataset(entradas, alvos), batch_size=4, shuffle=True)

        modelo = GPTModel(
            vocab_size=self.tokenizador.n_vocab, context_length=janela,
            emb_dim=32, n_layers=2, num_heads=2, drop_rate=0.0,
        )
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3)

        perda_inicial = calcular_perda_loader(loader, modelo)
        perdas_treino, perdas_val, tokens_vistos, amostras = treinar_modelo(
            modelo, loader, loader, otimizador, device=None,
            num_epocas=5, freq_avaliacao=1, lotes_avaliacao=2,
            contexto_inicial="o velho", tokenizador=self.tokenizador, seed=0,
        )
        perda_final = calcular_perda_loader(loader, modelo)

        self.assertLess(perda_final, perda_inicial)
        self.assertEqual(len(amostras), 5)  # uma amostra por época
        self.assertTrue(all(isinstance(a, str) for a in amostras))
        self.assertEqual(len(perdas_treino), len(perdas_val))
        self.assertEqual(len(perdas_treino), len(tokens_vistos))
        # tokens_vistos deve ser estritamente crescente
        self.assertTrue(all(b > a for a, b in zip(tokens_vistos, tokens_vistos[1:])))


if __name__ == "__main__":
    unittest.main()
