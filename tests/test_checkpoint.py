"""Testes do salvamento e carregamento dos pesos (Sprint 5, Entregável 6)."""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.checkpoint import (  # noqa: E402
    carregar_checkpoint,
    carregar_modelo,
    config_do_modelo,
    mesmos_pesos,
    numero_de_parametros,
    resumir_checkpoint,
    restaurar_otimizador,
    salvar_checkpoint,
    tamanho_do_arquivo_mb,
)
from src.geracao import gerar_texto_greedy  # noqa: E402
from src.transformer import GPTModel  # noqa: E402

CONFIG = dict(
    vocab_size=40, context_length=8, emb_dim=16,
    n_layers=1, num_heads=2, drop_rate=0.0, qkv_bias=False,
)


def modelo_pequeno(seed: int = 123) -> GPTModel:
    return GPTModel(seed=seed, **CONFIG)


class TestConfigDoModelo(unittest.TestCase):
    def test_extrai_os_campos_da_arquitetura(self):
        config = config_do_modelo(modelo_pequeno())
        for campo in ("vocab_size", "context_length", "emb_dim", "n_layers", "num_heads"):
            self.assertEqual(config[campo], CONFIG[campo])

    def test_deduz_dropout_e_qkv_bias(self):
        config = config_do_modelo(modelo_pequeno())
        self.assertAlmostEqual(config["drop_rate"], CONFIG["drop_rate"])
        self.assertEqual(config["qkv_bias"], CONFIG["qkv_bias"])

    def test_config_permite_reconstruir(self):
        original = modelo_pequeno()
        copia = GPTModel(**config_do_modelo(original))
        self.assertEqual(numero_de_parametros(original), numero_de_parametros(copia))


class TestSalvarCarregar(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.caminho = pathlib.Path(self.pasta.name) / "modelo.pt"
        self.modelo = modelo_pequeno()
        self.modelo.eval()

    def tearDown(self):
        self.pasta.cleanup()

    def test_cria_o_arquivo(self):
        salvar_checkpoint(self.caminho, self.modelo)
        self.assertTrue(self.caminho.exists())
        self.assertGreater(tamanho_do_arquivo_mb(self.caminho), 0.0)

    def test_cria_a_pasta_se_nao_existir(self):
        destino = pathlib.Path(self.pasta.name) / "nova" / "sub" / "m.pt"
        salvar_checkpoint(destino, self.modelo)
        self.assertTrue(destino.exists())

    def test_pesos_identicos_apos_recarregar(self):
        salvar_checkpoint(self.caminho, self.modelo)
        carregado, _ = carregar_modelo(self.caminho)
        self.assertTrue(mesmos_pesos(self.modelo, carregado))

    def test_geracao_identica_apos_recarregar(self):
        salvar_checkpoint(self.caminho, self.modelo)
        carregado, _ = carregar_modelo(self.caminho)
        idx = torch.tensor([[1, 2, 3]])
        a = gerar_texto_greedy(self.modelo, idx, 5, CONFIG["context_length"])
        b = gerar_texto_greedy(carregado, idx, 5, CONFIG["context_length"])
        self.assertEqual(a.tolist(), b.tolist())

    def test_modelo_volta_em_modo_avaliacao(self):
        salvar_checkpoint(self.caminho, self.modelo)
        carregado, _ = carregar_modelo(self.caminho)
        self.assertFalse(carregado.training)

    def test_carrega_sobre_instancia_existente(self):
        salvar_checkpoint(self.caminho, self.modelo)
        destino = modelo_pequeno(seed=999)
        self.assertFalse(mesmos_pesos(self.modelo, destino))
        carregado, _ = carregar_modelo(self.caminho, modelo=destino)
        self.assertTrue(mesmos_pesos(self.modelo, carregado))

    def test_guarda_a_configuracao(self):
        salvar_checkpoint(self.caminho, self.modelo)
        conteudo = carregar_checkpoint(self.caminho)
        self.assertEqual(conteudo["config"]["emb_dim"], CONFIG["emb_dim"])
        self.assertEqual(conteudo["config"]["n_layers"], CONFIG["n_layers"])

    def test_config_explicita_tem_prioridade(self):
        salvar_checkpoint(self.caminho, self.modelo, config={"anotacao": "teste"})
        self.assertEqual(carregar_checkpoint(self.caminho)["config"], {"anotacao": "teste"})

    def test_metadados_do_usuario_e_automaticos(self):
        salvar_checkpoint(self.caminho, self.modelo, metadados={"perda": 1.23})
        meta = carregar_checkpoint(self.caminho)["metadados"]
        self.assertEqual(meta["perda"], 1.23)
        self.assertIn("salvo_em", meta)
        self.assertEqual(meta["n_parametros"], numero_de_parametros(self.modelo))

    def test_resumo_do_checkpoint(self):
        salvar_checkpoint(self.caminho, self.modelo)
        resumo = resumir_checkpoint(self.caminho)
        self.assertGreater(resumo["n_tensores"], 0)
        self.assertFalse(resumo["tem_otimizador"])
        self.assertEqual(resumo["metadados"]["n_parametros"],
                         numero_de_parametros(self.modelo))

    def test_arquivo_guarda_parametros_e_buffers(self):
        # o state_dict inclui a máscara causal de cada bloco, registrada como
        # buffer: o arquivo guarda mais valores do que o modelo tem parâmetros
        salvar_checkpoint(self.caminho, self.modelo)
        resumo = resumir_checkpoint(self.caminho)
        parametros = numero_de_parametros(self.modelo)
        mascaras = CONFIG["n_layers"] * CONFIG["context_length"] ** 2
        self.assertEqual(resumo["n_valores"], parametros + mascaras)


class TestOtimizador(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.caminho = pathlib.Path(self.pasta.name) / "m.pt"
        self.modelo = modelo_pequeno()
        self.otimizador = torch.optim.AdamW(self.modelo.parameters(), lr=1e-3)

    def tearDown(self):
        self.pasta.cleanup()

    def test_salva_o_estado_do_otimizador(self):
        salvar_checkpoint(self.caminho, self.modelo, otimizador=self.otimizador)
        self.assertTrue(resumir_checkpoint(self.caminho)["tem_otimizador"])

    def test_restaura_o_estado(self):
        salvar_checkpoint(self.caminho, self.modelo, otimizador=self.otimizador)
        conteudo = carregar_checkpoint(self.caminho)
        novo = torch.optim.AdamW(self.modelo.parameters(), lr=1e-3)
        restaurar_otimizador(novo, conteudo)
        self.assertEqual(
            len(novo.state_dict()["param_groups"]),
            len(self.otimizador.state_dict()["param_groups"]),
        )

    def test_erro_quando_nao_ha_estado_salvo(self):
        salvar_checkpoint(self.caminho, self.modelo)
        conteudo = carregar_checkpoint(self.caminho)
        with self.assertRaises(ValueError):
            restaurar_otimizador(self.otimizador, conteudo)


class TestErros(unittest.TestCase):
    def test_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            carregar_checkpoint("/caminho/que/nao/existe.pt")

    def test_arquivo_sem_as_chaves_esperadas(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = pathlib.Path(pasta) / "qualquer.pt"
            torch.save({"algo": 1}, caminho)
            with self.assertRaises(ValueError):
                carregar_checkpoint(caminho)


class TestMesmosPesos(unittest.TestCase):
    def test_detecta_modelos_diferentes(self):
        self.assertFalse(mesmos_pesos(modelo_pequeno(123), modelo_pequeno(999)))

    def test_reconhece_modelos_iguais(self):
        self.assertTrue(mesmos_pesos(modelo_pequeno(7), modelo_pequeno(7)))


if __name__ == "__main__":
    unittest.main()
