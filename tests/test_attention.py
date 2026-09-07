"""Testes dos mecanismos de atenção (Sprint 3)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.attention import (  # noqa: E402
    CausalAttention,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    SelfAttention_v1,
    SelfAttention_v2,
    context_vector_passo_a_passo,
    entropia_dos_pesos,
    numero_de_parametros,
    scaled_dot_product_attention,
    self_attention_simples,
)


class TestSelfAttentionSimples(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.inputs = torch.rand(6, 3)

    def test_formas(self):
        contexto, pesos = self_attention_simples(self.inputs)
        self.assertEqual(tuple(contexto.shape), (6, 3))
        self.assertEqual(tuple(pesos.shape), (6, 6))

    def test_linhas_somam_um(self):
        _, pesos = self_attention_simples(self.inputs)
        self.assertTrue(torch.allclose(pesos.sum(dim=-1), torch.ones(6), atol=1e-6))

    def test_pesos_nao_negativos(self):
        _, pesos = self_attention_simples(self.inputs)
        self.assertGreaterEqual(float(pesos.min()), 0.0)

    def test_passo_a_passo_bate_com_matricial(self):
        contexto_todos, _ = self_attention_simples(self.inputs)
        contexto_um, _, _ = context_vector_passo_a_passo(self.inputs, 2)
        self.assertTrue(torch.allclose(contexto_todos[2], contexto_um, atol=1e-5))

    def test_matriz_e_simetrica_sem_pesos_treinaveis(self):
        # sem Wq/Wk, os scores são inputs @ inputs.T, que é simétrica
        scores = self.inputs @ self.inputs.T
        self.assertTrue(torch.allclose(scores, scores.T, atol=1e-6))


class TestScaledDotProduct(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.q = torch.randn(5, 16)
        self.k = torch.randn(5, 16)
        self.v = torch.randn(5, 16)

    def test_formas(self):
        contexto, pesos = scaled_dot_product_attention(self.q, self.k, self.v)
        self.assertEqual(tuple(contexto.shape), (5, 16))
        self.assertEqual(tuple(pesos.shape), (5, 5))

    def test_escala_reduz_concentracao(self):
        _, com = scaled_dot_product_attention(self.q, self.k, self.v, escalar=True)
        _, sem = scaled_dot_product_attention(self.q, self.k, self.v, escalar=False)
        self.assertGreater(entropia_dos_pesos(com), entropia_dos_pesos(sem))

    def test_mascara_zera_posicoes(self):
        mascara = torch.triu(torch.ones(5, 5), diagonal=1).bool()
        _, pesos = scaled_dot_product_attention(self.q, self.k, self.v, mascara=mascara)
        self.assertAlmostEqual(float(pesos.triu(diagonal=1).max()), 0.0, places=9)
        self.assertTrue(torch.allclose(pesos.sum(dim=-1), torch.ones(5), atol=1e-6))


class TestSelfAttentionTreinavel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.rand(6, 3)

    def test_v1_forma_e_parametros(self):
        camada = SelfAttention_v1(3, 2)
        self.assertEqual(tuple(camada(self.x).shape), (6, 2))
        self.assertEqual(numero_de_parametros(camada), 3 * 2 * 3)

    def test_v2_forma_e_parametros(self):
        camada = SelfAttention_v2(3, 2)
        self.assertEqual(tuple(camada(self.x).shape), (6, 2))
        self.assertEqual(numero_de_parametros(camada), 3 * 2 * 3)

    def test_guarda_pesos_para_visualizacao(self):
        camada = SelfAttention_v2(3, 2)
        camada(self.x)
        self.assertEqual(tuple(camada.ultimo_attn_weights.shape), (6, 6))
        self.assertTrue(
            torch.allclose(camada.ultimo_attn_weights.sum(dim=-1), torch.ones(6), atol=1e-6)
        )

    def test_semente_torna_deterministico(self):
        a = SelfAttention_v2(3, 2, seed=99)(self.x)
        b = SelfAttention_v2(3, 2, seed=99)(self.x)
        self.assertTrue(torch.allclose(a, b))


class TestCausalAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.n = 6
        self.lote = torch.rand(2, self.n, 3)
        self.camada = CausalAttention(3, 2, context_length=self.n, dropout=0.0)

    def test_forma_com_batch(self):
        self.assertEqual(tuple(self.camada(self.lote).shape), (2, self.n, 2))

    def test_futuro_tem_peso_zero(self):
        self.camada(self.lote)
        pesos = self.camada.ultimo_attn_weights
        self.assertAlmostEqual(float(pesos.triu(diagonal=1).max()), 0.0, places=9)

    def test_linhas_continuam_somando_um(self):
        self.camada(self.lote)
        somas = self.camada.ultimo_attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(somas, torch.ones_like(somas), atol=1e-6))

    def test_primeiro_token_so_ve_a_si_mesmo(self):
        self.camada(self.lote)
        self.assertAlmostEqual(float(self.camada.ultimo_attn_weights[0, 0, 0]), 1.0, places=6)

    def test_aceita_sequencia_menor_que_o_contexto(self):
        camada = CausalAttention(3, 2, context_length=16, dropout=0.0)
        saida = camada(torch.rand(1, 4, 3))
        self.assertEqual(tuple(saida.shape), (1, 4, 2))
        self.assertEqual(tuple(camada.ultimo_attn_weights.shape), (1, 4, 4))

    def test_mascara_e_buffer_nao_parametro(self):
        nomes = [n for n, _ in self.camada.named_buffers()]
        self.assertIn("mask", nomes)
        self.assertNotIn("mask", [n for n, _ in self.camada.named_parameters()])

    def test_dropout_desligado_em_eval(self):
        camada = CausalAttention(3, 2, self.n, dropout=0.5)
        camada.eval()
        a = camada(self.lote)
        b = camada(self.lote)
        self.assertTrue(torch.allclose(a, b))


class TestMultiHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.n = 6
        self.lote = torch.rand(2, self.n, 8)

    def test_wrapper_concatena_cabecas(self):
        camada = MultiHeadAttentionWrapper(8, 4, self.n, num_heads=3)
        self.assertEqual(tuple(camada(self.lote).shape), (2, self.n, 12))

    def test_eficiente_mantem_d_out(self):
        camada = MultiHeadAttention(8, 8, self.n, num_heads=4)
        self.assertEqual(tuple(camada(self.lote).shape), (2, self.n, 8))

    def test_forma_dos_pesos_por_cabeca(self):
        camada = MultiHeadAttention(8, 8, self.n, num_heads=4)
        camada(self.lote)
        self.assertEqual(tuple(camada.ultimo_attn_weights.shape), (2, 4, self.n, self.n))

    def test_todas_as_cabecas_respeitam_a_mascara(self):
        camada = MultiHeadAttention(8, 8, self.n, num_heads=4)
        camada(self.lote)
        self.assertAlmostEqual(
            float(camada.ultimo_attn_weights.triu(diagonal=1).max()), 0.0, places=9
        )

    def test_cabecas_aprendem_padroes_diferentes(self):
        camada = MultiHeadAttention(16, 16, self.n, num_heads=4)
        camada(torch.rand(1, self.n, 16))
        pesos = camada.ultimo_attn_weights[0]
        self.assertFalse(torch.allclose(pesos[0], pesos[1], atol=1e-3))

    def test_d_out_precisa_ser_divisivel(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(8, 10, self.n, num_heads=4)

    def test_numero_de_cabecas_nao_muda_parametros_qkv(self):
        a = MultiHeadAttention(96, 96, self.n, num_heads=2)
        b = MultiHeadAttention(96, 96, self.n, num_heads=12)
        self.assertEqual(numero_de_parametros(a), numero_de_parametros(b))

    def test_saida_tem_a_forma_da_entrada(self):
        # é o que permite empilhar blocos na Sprint 4
        camada = MultiHeadAttention(8, 8, self.n, num_heads=2)
        saida = camada(self.lote)
        self.assertEqual(tuple(saida.shape), tuple(self.lote.shape))


class TestEntropia(unittest.TestCase):
    def test_uniforme_tem_entropia_maxima(self):
        uniforme = torch.full((4, 4), 0.25)
        self.assertAlmostEqual(entropia_dos_pesos(uniforme), 2.0, places=5)

    def test_concentrada_tem_entropia_zero(self):
        concentrada = torch.eye(4)
        self.assertAlmostEqual(entropia_dos_pesos(concentrada), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
