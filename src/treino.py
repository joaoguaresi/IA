"""
Sprint 5 — Avaliação e treinamento do modelo GPT (Entregáveis 2 e 3).

Corresponde às seções 5.1 (Evaluating generative text models) e 5.2
(Training an LLM) do Capítulo 5: como medir, numericamente, a qualidade do
texto gerado por um GPTModel (cross-entropy loss e perplexity) e como usar
essa medida para efetivamente treinar o modelo (loop de épocas/lotes,
retropropagação, otimizador).

Fluxo implementado:

    GPTModel (pesos aleatórios, Sprint 4)
      -> calcular_perda_lote / calcular_perda_loader   (seção 5.1: avaliação)
      -> treinar_modelo                                (seção 5.2: treino)
      -> GPTModel (pesos ajustados) + histórico de perdas

A perda usada é a mesma cross-entropy de qualquer classificador multi-classe:
o GPTModel produz logits (batch, tokens, vocab_size); comparando-os com os
alvos (batch, tokens) — que já vêm deslocados uma posição desde a Sprint 2 —,
a cross-entropy mede o quão longe a probabilidade atribuída ao token certo
está de 1. Minimizar essa perda É o treinamento: não há nenhuma função de
perda "especial" para LLMs, só a aplicação repetida da mesma ideia de uma
classificação comum, token a token.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .geracao import gerar_texto_greedy, texto_para_ids


# --------------------------------------------------------------------------- #
# 1. Avaliação: cross-entropy loss e perplexity (seção 5.1)
# --------------------------------------------------------------------------- #
def calcular_perda_lote(
    entradas: torch.Tensor,
    alvos: torch.Tensor,
    modelo: nn.Module,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Cross-entropy loss de um único lote (seção 5.1.2, `calc_loss_batch`).

    O modelo produz logits de forma (batch, tokens, vocab_size); achatamos a
    dimensão de lote e a de tokens em uma só, (batch*tokens, vocab_size),
    para comparar, posição a posição, com os alvos achatados (batch*tokens,)
    — exatamente a interface que `torch.nn.functional.cross_entropy` espera
    de uma classificação multi-classe comum, só que aplicada T vezes por
    amostra em vez de uma vez só.
    """
    if device is not None:
        entradas = entradas.to(device)
        alvos = alvos.to(device)
    logits = modelo(entradas)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), alvos.flatten())


def calcular_perda_loader(
    loader: DataLoader,
    modelo: nn.Module,
    device: Optional[torch.device] = None,
    num_lotes: Optional[int] = None,
) -> float:
    """
    Perda média sobre um DataLoader inteiro, ou só sobre os `num_lotes` primeiros
    (seção 5.1.3, `calc_loss_loader`).

    `num_lotes` existe para acelerar a avaliação durante o treino (seção 5.2):
    calcular a perda sobre TODO o conjunto de validação a cada poucos passos
    seria caro; uma amostra de alguns lotes já dá uma estimativa razoável.
    """
    if len(loader) == 0:
        return float("nan")
    total = min(num_lotes, len(loader)) if num_lotes is not None else len(loader)
    perda_acumulada = 0.0
    for i, (entradas, alvos) in enumerate(loader):
        if i >= total:
            break
        perda_acumulada += calcular_perda_lote(entradas, alvos, modelo, device).item()
    return perda_acumulada / total


def perplexidade(perda: float) -> float:
    """
    Perplexity = e^perda (seção 5.1.2, quadro "Perplexity").

    Mais interpretável que a perda crua: representa o tamanho efetivo do
    vocabulário sobre o qual o modelo está "em dúvida" a cada posição. Uma
    perplexity de 48.725, por exemplo (o valor do próprio livro para o
    modelo ainda não treinado), significa que o modelo está tão incerto
    quanto se tivesse que escolher ao acaso entre 48.725 tokens igualmente
    prováveis — próximo do vocabulário inteiro do GPT-2 (50.257), ou seja,
    praticamente sem nenhuma noção de qual token vem a seguir.
    """
    return float(torch.exp(torch.tensor(perda)))


@torch.no_grad()
def avaliar_modelo(
    modelo: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    num_lotes: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Perda de treino e de validação, com o modelo em modo de avaliação (seção 5.2,
    `evaluate_model`).

    `modelo.eval()` desliga o dropout — sem isso, a perda de validação
    oscilaria de uma chamada para outra só por causa do dropout, mascarando
    o progresso real do treino. `modelo.train()` é restaurado ao final para
    que o loop de treinamento continue normalmente.
    """
    modelo.eval()
    perda_treino = calcular_perda_loader(train_loader, modelo, device, num_lotes)
    perda_val = calcular_perda_loader(val_loader, modelo, device, num_lotes)
    modelo.train()
    return perda_treino, perda_val


@torch.no_grad()
def gerar_e_imprimir_amostra(
    modelo: nn.Module,
    tokenizador,
    device: Optional[torch.device],
    contexto_inicial: str,
    max_novos_tokens: int = 50,
) -> str:
    """
    Gera uma amostra de texto (greedy) para acompanhar o treino visualmente
    (seção 5.2, `generate_and_print_sample`).

    Reaproveita `gerar_texto_greedy` da Sprint 4 (`src/geracao.py`) em vez de
    reimplementar o laço de geração — o único código novo aqui é alternar o
    modelo para `eval()` antes e `train()` depois, para que a amostragem não
    seja afetada pelo dropout nem deixe o modelo fora do modo de treino.
    """
    modelo.eval()
    idx = texto_para_ids(contexto_inicial, tokenizador, device=device)
    ids_gerados = gerar_texto_greedy(modelo, idx, max_novos_tokens, modelo.context_length)
    texto = tokenizador.decode(ids_gerados.squeeze(0).tolist())
    modelo.train()
    return texto.replace("\n", " ")


# --------------------------------------------------------------------------- #
# 2. Treinamento (seção 5.2)
# --------------------------------------------------------------------------- #
def treinar_modelo(
    modelo: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    otimizador: torch.optim.Optimizer,
    device: Optional[torch.device],
    num_epocas: int,
    freq_avaliacao: int,
    lotes_avaliacao: int,
    contexto_inicial: str,
    tokenizador,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[float], List[int], List[str]]:
    """
    Loop de treinamento (seção 5.2, `train_model_simple` / Figura 5.11).

    Para cada época, para cada lote: zera os gradientes do passo anterior
    (`zero_grad`), calcula a perda do lote atual, retropropaga
    (`perda.backward()`) e atualiza os pesos (`otimizador.step()`) — os
    quatro passos centrais de qualquer treino supervisionado em PyTorch,
    aqui aplicados à tarefa de prever o próximo token.

    A cada `freq_avaliacao` passos, mede a perda de treino/validação sobre
    `lotes_avaliacao` lotes (não o conjunto inteiro, para não pesar o
    treino) e registra o ponto. Ao final de cada época, gera uma amostra de
    texto a partir de `contexto_inicial`, permitindo acompanhar
    qualitativamente a evolução do modelo além do número da perda.

    Devolve `(perdas_treino, perdas_val, tokens_vistos, amostras)`: as três
    primeiras listas alinhadas por índice (um ponto por avaliação), a última
    com uma amostra de texto por época. Nenhuma impressão acontece aqui —
    cabe ao script de experimento decidir o que e como imprimir.
    """
    if seed is not None:
        torch.manual_seed(seed)

    perdas_treino: List[float] = []
    perdas_val: List[float] = []
    tokens_vistos_hist: List[int] = []
    amostras: List[str] = []
    tokens_vistos, passo_global = 0, -1

    for _ in range(num_epocas):
        modelo.train()
        for entradas, alvos in train_loader:
            otimizador.zero_grad()
            perda = calcular_perda_lote(entradas, alvos, modelo, device)
            perda.backward()
            otimizador.step()
            tokens_vistos += entradas.numel()
            passo_global += 1

            if passo_global % freq_avaliacao == 0:
                p_treino, p_val = avaliar_modelo(
                    modelo, train_loader, val_loader, device, lotes_avaliacao
                )
                perdas_treino.append(p_treino)
                perdas_val.append(p_val)
                tokens_vistos_hist.append(tokens_vistos)

        amostras.append(
            gerar_e_imprimir_amostra(modelo, tokenizador, device, contexto_inicial)
        )

    return perdas_treino, perdas_val, tokens_vistos_hist, amostras


# --------------------------------------------------------------------------- #
# Utilidades de análise
# --------------------------------------------------------------------------- #
def plotar_perdas(
    epocas_vistas: Sequence[float],
    tokens_vistos: Sequence[int],
    perdas_treino: Sequence[float],
    perdas_val: Sequence[float],
):
    """
    Gráfico de perda de treino/validação por época, com um segundo eixo
    horizontal de tokens vistos (seção 5.2, Figura 5.12) — a forma mais
    direta de visualizar convergência e overfitting: as duas curvas descem
    juntas no início; se a de validação estaciona ou volta a subir enquanto
    a de treino continua caindo, é sinal de que o modelo está decorando o
    conjunto de treino em vez de generalizar.

    Devolve a `Figure` do matplotlib (não chama `plt.show()`), para que o
    script de experimento decida se salva, mostra ou embute em um painel —
    mesma convenção de `src/visualizacao.py` na Sprint 3.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epocas_vistas, perdas_treino, color="#1C5D86", linewidth=2, label="Perda de treino")
    ax1.plot(epocas_vistas, perdas_val, color="#9A5312", linewidth=2, linestyle="-.",
              label="Perda de validação")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Perda (cross-entropy)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="upper right", frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(alpha=0.25, linewidth=0.7)

    ax2 = ax1.twiny()
    ax2.plot(tokens_vistos, perdas_treino, alpha=0)  # só para alinhar os ticks do eixo
    ax2.set_xlabel("Tokens vistos")

    fig.tight_layout()
    return fig
