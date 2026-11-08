"""
Sprint 5 — Salvamento e carregamento dos pesos (Entregável 6).

Corresponde à seção 5.4 do Capítulo 5. Treinar um modelo custa tempo; sem
persistência, todo o trabalho se perde quando o processo termina. Este módulo
guarda e recupera três coisas:

    1. os PESOS do modelo (state_dict) — o que foi aprendido;
    2. a CONFIGURAÇÃO da arquitetura — sem ela não é possível reconstruir o
       objeto para receber os pesos de volta;
    3. o ESTADO DO OTIMIZADOR e metadados — o que permite retomar o treino
       exatamente de onde parou, em vez de recomeçar.

O ponto 2 costuma ser esquecido: um arquivo só com os pesos é inútil para quem
não souber de cor quantas camadas, quantas cabeças e que dimensão o modelo tinha.
Por isso `salvar_checkpoint` deduz a configuração a partir do próprio modelo e a
grava junto.
"""

from __future__ import annotations

import datetime
import pathlib
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from .attention import numero_de_parametros

Caminho = Union[str, pathlib.Path]

VERSAO_FORMATO = 1
CHAVES_OBRIGATORIAS = ("versao_formato", "config", "pesos_modelo")


# --------------------------------------------------------------------------- #
# Configuração da arquitetura
# --------------------------------------------------------------------------- #
def config_do_modelo(modelo: nn.Module) -> Dict[str, Any]:
    """
    Extrai do próprio modelo os hiperparâmetros necessários para reconstruí-lo.

    `vocab_size`, `context_length`, `emb_dim`, `n_layers` e `num_heads` são
    atributos do GPTModel. Os outros dois são deduzidos das camadas: a taxa de
    dropout vem da camada aplicada aos embeddings, e `qkv_bias` é verificado
    olhando se a projeção de query do primeiro bloco tem termo de viés.
    """
    config = {
        campo: getattr(modelo, campo)
        for campo in ("vocab_size", "context_length", "emb_dim", "n_layers", "num_heads")
        if hasattr(modelo, campo)
    }
    drop = getattr(modelo, "drop_emb", None)
    if drop is not None and hasattr(drop, "p"):
        config["drop_rate"] = float(drop.p)
    blocos = getattr(modelo, "blocos", None)
    if blocos is not None and len(blocos) > 0:
        atencao = getattr(blocos[0], "atencao", None)
        consulta = getattr(atencao, "W_query", None) if atencao is not None else None
        if consulta is not None:
            config["qkv_bias"] = getattr(consulta, "bias", None) is not None
    return config


# --------------------------------------------------------------------------- #
# Salvar
# --------------------------------------------------------------------------- #
def salvar_checkpoint(
    caminho: Caminho,
    modelo: nn.Module,
    otimizador: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    metadados: Optional[Dict[str, Any]] = None,
) -> pathlib.Path:
    """
    Grava pesos, configuração e (opcionalmente) estado do otimizador.

    Salvamos o `state_dict` e não o objeto do modelo inteiro. O state_dict é um
    dicionário de tensores, independente do código que o criou; serializar o
    objeto amarraria o arquivo à estrutura exata das classes, e qualquer
    renomeação futura de módulo o tornaria impossível de abrir.

    Informe o `otimizador` quando a intenção for RETOMAR o treino: otimizadores
    como o AdamW mantêm médias móveis dos gradientes, e descartá-las faz o
    treino recomeçar com passos mal calibrados nas primeiras iterações.
    """
    caminho = pathlib.Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    conteudo: Dict[str, Any] = {
        "versao_formato": VERSAO_FORMATO,
        "config": dict(config) if config else config_do_modelo(modelo),
        "pesos_modelo": modelo.state_dict(),
        "metadados": dict(metadados) if metadados else {},
    }
    conteudo["metadados"].setdefault(
        "salvo_em", datetime.datetime.now().isoformat(timespec="seconds")
    )
    conteudo["metadados"].setdefault("n_parametros", numero_de_parametros(modelo))
    if otimizador is not None:
        conteudo["estado_otimizador"] = otimizador.state_dict()
        conteudo["metadados"].setdefault("otimizador", type(otimizador).__name__)

    torch.save(conteudo, caminho)
    return caminho


# --------------------------------------------------------------------------- #
# Carregar
# --------------------------------------------------------------------------- #
def carregar_checkpoint(caminho: Caminho, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Lê o arquivo e devolve o dicionário salvo, sem construir modelo nenhum.

    `map_location` garante que um checkpoint gravado em GPU possa ser aberto em
    uma máquina que só tem CPU — sem isso, o carregamento falha procurando um
    dispositivo que não existe.
    """
    caminho = pathlib.Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {caminho}")

    conteudo = torch.load(caminho, map_location=device or "cpu", weights_only=False)
    faltando = [c for c in CHAVES_OBRIGATORIAS if c not in conteudo]
    if faltando:
        raise ValueError(
            f"Arquivo não parece um checkpoint deste projeto — faltam as chaves {faltando}"
        )
    return conteudo


def carregar_modelo(
    caminho: Caminho,
    device: Optional[torch.device] = None,
    modelo: Optional[nn.Module] = None,
    strict: bool = True,
):
    """
    Reconstrói o modelo a partir do checkpoint e devolve `(modelo, checkpoint)`.

    Sem o argumento `modelo`, a arquitetura é recriada do zero usando a
    configuração salva — é isto que torna o arquivo autossuficiente. Informe um
    modelo já construído apenas quando quiser carregar os pesos sobre uma
    instância existente.

    O modelo volta em modo de avaliação (`eval`), que é o estado correto para
    inferência: sem dropout e, portanto, determinístico.
    """
    conteudo = carregar_checkpoint(caminho, device)

    if modelo is None:
        from .transformer import GPTModel

        config = dict(conteudo["config"])
        config.pop("seed", None)
        modelo = GPTModel(**config)

    modelo.load_state_dict(conteudo["pesos_modelo"], strict=strict)
    if device is not None:
        modelo = modelo.to(device)
    modelo.eval()
    return modelo, conteudo


def restaurar_otimizador(
    otimizador: torch.optim.Optimizer, checkpoint: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Devolve ao otimizador o estado salvo, para retomar o treino."""
    estado = checkpoint.get("estado_otimizador")
    if estado is None:
        raise ValueError(
            "Este checkpoint não guardou o estado do otimizador — salve com "
            "`salvar_checkpoint(..., otimizador=otim)` se quiser retomar o treino."
        )
    otimizador.load_state_dict(estado)
    return otimizador


# --------------------------------------------------------------------------- #
# Inspeção
# --------------------------------------------------------------------------- #
def tamanho_do_arquivo_mb(caminho: Caminho) -> float:
    """
    Tamanho do arquivo de checkpoint, em MB.

    Nome distinto de `src.embeddings.tamanho_em_mb`, que estima a memória da
    matriz de embeddings — são medidas diferentes e não podem se confundir no
    pacote.
    """
    return pathlib.Path(caminho).stat().st_size / (1024**2)


def resumir_checkpoint(caminho: Caminho) -> Dict[str, Any]:
    """
    Resumo legível do que há dentro do arquivo, sem reconstruir o modelo.

    É o que permite identificar a configuração de um checkpoint antigo meses
    depois, requisito do enunciado para reproduzir os experimentos.

    `n_valores` conta TUDO o que está gravado, não só os parâmetros treináveis:
    o `state_dict` carrega também os buffers registrados com `register_buffer`,
    e neste projeto isso inclui a máscara causal de cada bloco. Por isso esse
    número é maior que o de parâmetros — a diferença é de context_length² por
    bloco, irrelevante em modelos pequenos e considerável em contextos longos.
    """
    conteudo = carregar_checkpoint(caminho)
    pesos = conteudo["pesos_modelo"]
    return {
        "arquivo": str(caminho),
        "tamanho_mb": round(tamanho_do_arquivo_mb(caminho), 2),
        "versao_formato": conteudo["versao_formato"],
        "config": conteudo["config"],
        "n_tensores": len(pesos),
        "n_valores": int(sum(int(t.numel()) for t in pesos.values())),
        "tem_otimizador": "estado_otimizador" in conteudo,
        "metadados": conteudo.get("metadados", {}),
    }


def mesmos_pesos(modelo_a: nn.Module, modelo_b: nn.Module, tolerancia: float = 0.0) -> bool:
    """
    Compara os pesos de dois modelos tensor a tensor.

    Usado nos testes e no experimento 21 para demonstrar que o modelo carregado
    é idêntico ao que foi salvo — e não apenas parecido.
    """
    a, b = modelo_a.state_dict(), modelo_b.state_dict()
    if a.keys() != b.keys():
        return False
    return all(bool(torch.allclose(a[k], b[k], atol=tolerancia, rtol=0.0)) for k in a)
