"""Testes das camadas de embedding e do pipeline completo."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.bpe import obter_tokenizador_bpe  # noqa: E402
from src.dados import carregar_texto  # noqa: E402
from src.dataset import create_dataloader_v1  # noqa: E402
from src.embeddings import (  # noqa: E402
    GPTEmbedding,
    comparar_posicoes,
    embedding_de_token,
    tamanho_em_mb,
)
from src.pipeline import executar_pipeline  # noqa: E402


class TestEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texto = carregar_texto()
        cls.tokenizador = obter_tokenizador_bpe()
        cls.vocab_size = cls.tokenizador.n_vocab

    def test_forma_da_saida(self):
        camada = GPTEmbedding(self.vocab_size, output_dim=32, context_length=8, seed=1)
        dl = create_dataloader_v1(
            self.texto,
            tokenizer=self.tokenizador,
            batch_size=4,
            max_length=8,
            stride=8,
            shuffle=False,
        )
        entradas, _ = next(iter(dl))
        saida = camada(entradas)
        self.assertEqual(tuple(saida.shape), (4, 8, 32))

    def test_matriz_de_lookup(self):
        camada = GPTEmbedding(self.vocab_size, output_dim=16, context_length=4, seed=1)
        self.assertEqual(tuple(camada.tok_emb.weight.shape), (self.vocab_size, 16))
        self.assertEqual(tuple(camada.pos_emb.weight.shape), (4, 16))

    def test_token_embedding_independe_da_posicao(self):
        camada = GPTEmbedding(self.vocab_size, output_dim=16, context_length=8, seed=1)
        v1 = embedding_de_token(camada, 5)
        v2 = embedding_de_token(camada, 5)
        self.assertEqual([float(a) for a in v1], [float(b) for b in v2])

    def test_positional_diferencia_posicoes(self):
        camada = GPTEmbedding(self.vocab_size, output_dim=16, context_length=8, seed=1)
        _, distancias = comparar_posicoes(camada, token_id=5, posicoes=[0, 1, 7])
        for distancia in distancias.values():
            self.assertGreater(distancia, 0.0)

    def test_memoria_cresce_linearmente(self):
        self.assertAlmostEqual(
            tamanho_em_mb(50257, 512), 2 * tamanho_em_mb(50257, 256), places=6
        )


class TestPipeline(unittest.TestCase):
    def test_pipeline_completo(self):
        texto = carregar_texto()
        r = executar_pipeline(
            texto, context_length=8, stride=8, batch_size=4, output_dim=32
        )
        self.assertEqual(r.forma_lote_entradas, (4, 8))
        self.assertEqual(r.forma_input_embeddings, (4, 8, 32))
        self.assertEqual(r.forma_token_embeddings, r.forma_input_embeddings)
        self.assertGreater(r.n_amostras, 0)
        self.assertGreater(r.n_tokens, r.n_amostras)


if __name__ == "__main__":
    unittest.main()
