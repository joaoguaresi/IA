"""
Preparação comum aos experimentos 21 e 22: um GPT pequeno treinado no corpus.

Os dois experimentos precisam de um modelo que tenha aprendido alguma coisa —
sem isso não há o que salvar no 21 nem o que comparar no 22. Este módulo
concentra a configuração e o treino curto, para que os dois usem exatamente o
mesmo modelo e os resultados sejam comparáveis entre si.
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import torch

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto
from src.dataset import create_dataloader_v1
from src.transformer import GPTModel
from src.treino import treinar_modelo

RAIZ = pathlib.Path(__file__).resolve().parents[1]
PASTA_CHECKPOINTS = RAIZ / "checkpoints"
CHECKPOINT_DEMO = PASTA_CHECKPOINTS / "gpt_demo_sprint5.pt"

# Configuração reduzida: o modelo do livro (124M parâmetros) não treina em tempo
# razoável numa CPU. As proporções da arquitetura são as mesmas; o que muda é a
# escala.
CONFIG_DEMO = dict(
    vocab_size=50257,
    context_length=32,
    emb_dim=96,
    n_layers=2,
    num_heads=4,
    drop_rate=0.1,
    qkv_bias=False,
)

TREINO = dict(
    num_epocas=3,
    batch_size=2,
    taxa_aprendizado=5e-4,
    weight_decay=0.1,
    fracao_treino=0.9,
    freq_avaliacao=5,
    lotes_avaliacao=2,
    seed=123,
)

CONTEXTO_INICIAL = "O velho consertou"


def preparar_loaders(texto: str, tokenizador, config=CONFIG_DEMO, treino=TREINO):
    """Divide o corpus em treino e validação e devolve os dois DataLoaders."""
    corte = int(len(texto) * treino["fracao_treino"])
    train_loader = create_dataloader_v1(
        texto[:corte], tokenizer=tokenizador, batch_size=treino["batch_size"],
        max_length=config["context_length"], stride=config["context_length"],
        shuffle=True, drop_last=True,
    )
    val_loader = create_dataloader_v1(
        texto[corte:], tokenizer=tokenizador, batch_size=treino["batch_size"],
        max_length=config["context_length"], stride=config["context_length"],
        shuffle=False, drop_last=False,
    )
    return train_loader, val_loader


def modelo_novo(config=CONFIG_DEMO, seed: int = 123) -> GPTModel:
    """GPT com pesos aleatórios, na configuração reduzida."""
    return GPTModel(seed=seed, **config)


def treinar_demo(verbose: bool = True) -> Tuple[GPTModel, torch.optim.Optimizer, dict]:
    """
    Treina o modelo reduzido e devolve `(modelo, otimizador, historico)`.

    O histórico traz as perdas de treino e validação registradas durante a
    execução, mais a amostra de texto gerada ao fim de cada época.
    """
    tokenizador = obter_tokenizador_bpe()
    texto = carregar_texto()
    train_loader, val_loader = preparar_loaders(texto, tokenizador)

    torch.manual_seed(TREINO["seed"])
    modelo = modelo_novo()
    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=TREINO["taxa_aprendizado"],
        weight_decay=TREINO["weight_decay"],
    )

    perdas_treino, perdas_val, tokens_vistos, amostras = treinar_modelo(
        modelo, train_loader, val_loader, otimizador, None,
        TREINO["num_epocas"], TREINO["freq_avaliacao"], TREINO["lotes_avaliacao"],
        CONTEXTO_INICIAL, tokenizador, seed=TREINO["seed"],
    )

    historico = {
        "perdas_treino": perdas_treino,
        "perdas_val": perdas_val,
        "tokens_vistos": tokens_vistos,
        "amostras": amostras,
        "tokenizador": tokenizador,
        "texto": texto,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }
    if verbose:
        print(f"treino concluído: {len(perdas_treino)} avaliações, "
              f"{tokens_vistos[-1] if tokens_vistos else 0} tokens vistos")
    return modelo, otimizador, historico
