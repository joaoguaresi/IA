"""Testes do Transformer Block e do modelo GPT completo (Sprint 4, Entregáveis 2 e 3)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.transformer import (  # noqa: E402
    GELU,
    FeedForward,
    GPTModel,
    LayerNorm,
    TransformerBlock,
    memoria_em_mb,
    parametros_por_componente,
)


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.randn(2, 5, 8) * 17 + 3  # média e variância bem diferentes de 0/1

    def test_forma_preservada(self):
        norm = LayerNorm(8)
        self.assertEqual(tuple(norm(self.x).shape), (2, 5, 8))

    def test_media_proxima_de_zero_e_variancia_proxima_de_um(self):
        # antes do scale/shift aprendido, a normalização por si só já deve
        # centralizar e padronizar cada token individualmente
        norm = LayerNorm(8)
        with torch.no_grad():
            norm.scale.fill_(1.0)
            norm.shift.fill_(0.0)
        saida = norm(self.x)
        media = saida.mean(dim=-1)
        variancia = saida.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(media, torch.zeros_like(media), atol=1e-5))
        self.assertTrue(torch.allclose(variancia, torch.ones_like(variancia), atol=1e-3))

    def test_normaliza_por_token_nao_pelo_lote(self):
        # cada posição (batch, token) é normalizada independentemente das outras
        norm = LayerNorm(8)
        with torch.no_grad():
            saida = norm(self.x)
        for b in range(2):
            for t in range(5):
                self.assertAlmostEqual(float(saida[b, t].mean()), 0.0, delta=1e-4)

    def test_scale_e_shift_sao_identidade_na_inicializacao(self):
        norm = LayerNorm(8)
        self.assertTrue(torch.equal(norm.scale, torch.ones(8)))
        self.assertTrue(torch.equal(norm.shift, torch.zeros(8)))


class TestGELU(unittest.TestCase):
    def test_forma_preservada(self):
        gelu = GELU()
        x = torch.randn(3, 4)
        self.assertEqual(tuple(gelu(x).shape), (3, 4))

    def test_valores_muito_negativos_ficam_proximos_de_zero(self):
        gelu = GELU()
        saida = gelu(torch.tensor([-10.0, -5.0]))
        self.assertTrue(torch.all(saida.abs() < 0.01))

    def test_valores_muito_positivos_tendem_a_identidade(self):
        gelu = GELU()
        x = torch.tensor([10.0, 5.0])
        self.assertTrue(torch.allclose(gelu(x), x, atol=0.01))

    def test_zero_mapeia_para_zero(self):
        gelu = GELU()
        self.assertAlmostEqual(float(gelu(torch.tensor(0.0))), 0.0, places=5)

    def test_nao_e_monotona_como_relu_mas_e_suave(self):
        # a GELU deixa passar uma fração pequena de valores negativos, ao
        # contrário da ReLU, que os zeraria todos
        gelu = GELU()
        saida = gelu(torch.tensor([-1.0]))
        self.assertLess(float(saida), 0.0)
        self.assertGreater(float(saida), -1.0)


class TestFeedForward(unittest.TestCase):
    def test_forma_preservada_e_expansao_interna(self):
        ff = FeedForward(emb_dim=8)
        x = torch.rand(2, 5, 8)
        saida = ff(x)
        self.assertEqual(tuple(saida.shape), (2, 5, 8))
        # a camada intermediária deve expandir para 4x a dimensão
        primeira_camada = ff.camadas[0]
        self.assertEqual(primeira_camada.out_features, 32)


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.emb_dim, self.n_tokens = 8, 6
        self.x = torch.rand(2, self.n_tokens, self.emb_dim)

    def test_forma_de_entrada_igual_a_forma_de_saida(self):
        bloco = TransformerBlock(self.emb_dim, self.n_tokens, num_heads=2)
        saida = bloco(self.x)
        self.assertEqual(tuple(saida.shape), tuple(self.x.shape))

    def test_saida_difere_da_entrada(self):
        bloco = TransformerBlock(self.emb_dim, self.n_tokens, num_heads=2)
        saida = bloco(self.x)
        self.assertFalse(torch.allclose(saida, self.x))

    def test_conexao_residual_preserva_sinal_mesmo_com_sub_camadas_nulas(self):
        # zerando as transformações internas, a saída deve voltar a ser
        # exatamente a entrada — é a conexão residual (shortcut) agindo sozinha
        bloco = TransformerBlock(self.emb_dim, self.n_tokens, num_heads=2)
        with torch.no_grad():
            for p in bloco.atencao.parameters():
                p.zero_()
            for p in bloco.feed_forward.parameters():
                p.zero_()
            bloco.norm1.scale.zero_()
            bloco.norm2.scale.zero_()
        saida = bloco(self.x)
        self.assertTrue(torch.allclose(saida, self.x, atol=1e-6))

    def test_expoe_pesos_de_atencao_da_ultima_chamada(self):
        bloco = TransformerBlock(self.emb_dim, self.n_tokens, num_heads=2)
        bloco(self.x)
        self.assertEqual(
            tuple(bloco.ultimo_attn_weights.shape), (2, 2, self.n_tokens, self.n_tokens)
        )

    def test_dropout_rate_zero_e_determinístico(self):
        bloco = TransformerBlock(self.emb_dim, self.n_tokens, num_heads=2, drop_rate=0.0)
        bloco.eval()
        self.assertTrue(torch.allclose(bloco(self.x), bloco(self.x)))


class TestGPTModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.vocab_size, self.context_length = 50, 6
        self.idx = torch.randint(0, self.vocab_size, (2, self.context_length))

    def _modelo_pequeno(self, **kwargs):
        config = dict(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            emb_dim=16,
            n_layers=2,
            num_heads=2,
            drop_rate=0.0,
        )
        config.update(kwargs)
        return GPTModel(**config)

    def test_forma_da_saida_e_batch_tokens_vocab(self):
        modelo = self._modelo_pequeno()
        logits = modelo(self.idx)
        self.assertEqual(tuple(logits.shape), (2, self.context_length, self.vocab_size))

    def test_aceita_sequencia_mais_curta_que_o_contexto(self):
        modelo = self._modelo_pequeno()
        idx_curto = self.idx[:, :3]
        logits = modelo(idx_curto)
        self.assertEqual(tuple(logits.shape), (2, 3, self.vocab_size))

    def test_mesma_entrada_produz_mesma_saida_em_modo_avaliacao(self):
        modelo = self._modelo_pequeno()
        modelo.eval()
        with torch.no_grad():
            a = modelo(self.idx)
            b = modelo(self.idx)
        self.assertTrue(torch.allclose(a, b))

    def test_numero_de_blocos_bate_com_n_layers(self):
        modelo = self._modelo_pequeno(n_layers=4)
        self.assertEqual(len(modelo.blocos), 4)

    def test_d_out_nao_divisivel_por_num_heads_leva_a_valueerror(self):
        with self.assertRaises(ValueError):
            self._modelo_pequeno(emb_dim=10, num_heads=3)

    def test_ultimo_attn_weights_tem_um_item_por_bloco(self):
        modelo = self._modelo_pequeno(n_layers=3)
        modelo(self.idx)
        pesos = modelo.ultimo_attn_weights
        self.assertEqual(len(pesos), 3)
        self.assertEqual(tuple(pesos[0].shape), (2, 2, self.context_length, self.context_length))

    def test_out_head_nao_tem_vies(self):
        modelo = self._modelo_pequeno()
        self.assertIsNone(modelo.out_head.bias)


class TestUtilidadesDeAnalise(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.modelo = GPTModel(
            vocab_size=100, context_length=8, emb_dim=16, n_layers=2, num_heads=2,
        )

    def test_memoria_em_mb_e_positiva(self):
        self.assertGreater(memoria_em_mb(self.modelo), 0.0)

    def test_parametros_por_componente_soma_o_total(self):
        partes = parametros_por_componente(self.modelo)
        soma = (
            partes["token_embeddings"]
            + partes["positional_embeddings"]
            + partes["transformer_blocks"]
            + partes["layernorm_final"]
            + partes["out_head"]
        )
        self.assertEqual(soma, partes["total"])

    def test_mais_camadas_aumenta_os_parametros(self):
        menor = GPTModel(vocab_size=100, context_length=8, emb_dim=16, n_layers=1, num_heads=2)
        maior = GPTModel(vocab_size=100, context_length=8, emb_dim=16, n_layers=4, num_heads=2)
        partes_menor = parametros_por_componente(menor)
        partes_maior = parametros_por_componente(maior)
        self.assertLess(partes_menor["total"], partes_maior["total"])
        # só os blocos Transformer devem mudar; embeddings e out_head não dependem de n_layers
        self.assertEqual(
            partes_menor["token_embeddings"] + partes_menor["out_head"],
            partes_maior["token_embeddings"] + partes_maior["out_head"],
        )


if __name__ == "__main__":
    unittest.main()
