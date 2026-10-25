"""
Sprint 4 — Inferência e geração autoregressiva de texto (Entregável 4).

Corresponde à seção 4.7 do Capítulo 4 (geração gulosa) e acrescenta as
estratégias de amostragem pedidas pela Sprint: temperatura e top-k.

O ciclo autoregressivo, que é o coração deste módulo:

    1. o modelo recebe a sequência atual e devolve logits
    2. só os logits da ÚLTIMA posição interessam — são a previsão do próximo token
    3. os logits viram probabilidades (softmax)
    4. escolhe-se um token (o mais provável, ou um sorteado)
    5. o token escolhido entra no fim da sequência e tudo recomeça

Nada aqui depende de uma arquitetura específica: qualquer objeto chamável que
receba um tensor de Token IDs (batch, num_tokens) e devolva logits
(batch, num_tokens, vocab_size) serve como `modelo` — inclusive o GPTModel do
Entregável 3, quando estiver pronto.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

Modelo = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# Conversão entre texto e tensor de Token IDs
# --------------------------------------------------------------------------- #
def texto_para_ids(texto: str, tokenizador, device=None) -> torch.Tensor:
    """
    Texto -> tensor (1, num_tokens) pronto para o modelo.

    O `unsqueeze(0)` acrescenta a dimensão de lote: o modelo sempre espera
    (batch, num_tokens), mesmo quando há uma única sequência.
    """
    ids = tokenizador.encode(texto)
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return tensor if device is None else tensor.to(device)


def ids_para_texto(ids: torch.Tensor, tokenizador) -> str:
    """Tensor (1, num_tokens) ou (num_tokens,) -> texto."""
    lista = ids.squeeze(0).tolist() if ids.dim() > 1 else ids.tolist()
    return tokenizador.decode([int(i) for i in lista])


# --------------------------------------------------------------------------- #
# Transformações sobre os logits
# --------------------------------------------------------------------------- #
def aplicar_temperatura(logits: torch.Tensor, temperatura: float) -> torch.Tensor:
    """
    Divide os logits pela temperatura antes do softmax.

    temperatura < 1  -> distribuição mais afiada (modelo mais conservador)
    temperatura = 1  -> distribuição original do modelo
    temperatura > 1  -> distribuição mais achatada (modelo mais "criativo")

    A divisão não muda a ORDEM dos candidatos, só a distância entre eles — por
    isso a temperatura altera a variedade do texto, nunca qual token é o favorito.
    """
    if temperatura <= 0:
        raise ValueError("temperatura deve ser > 0 (use temperatura=0 em `gerar` para greedy)")
    return logits / temperatura


def aplicar_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mantém apenas os `k` maiores logits e manda o resto para -infinito.

    Depois do softmax, os descartados ficam com probabilidade exatamente zero,
    o que impede o modelo de sortear, por azar, um token muito improvável —
    o principal causador de texto incoerente quando a temperatura é alta.

    O corte é feito pelo VALOR do k-ésimo maior logit. Se houver empate nesse
    valor, todos os empatados sobrevivem, e o número de candidatos pode passar
    de k. Num modelo treinado isso praticamente não ocorre (os logits são
    números reais distintos), mas acontece em modelos de contagem com
    suavização uniforme, em que todos os tokens não observados têm o mesmo logit.
    """
    if k <= 0:
        raise ValueError("top_k deve ser >= 1")
    k = min(k, logits.shape[-1])
    maiores, _ = torch.topk(logits, k, dim=-1)
    corte = maiores[..., -1].unsqueeze(-1)          # o menor logit que sobrevive
    return torch.where(logits < corte, torch.full_like(logits, -torch.inf), logits)


def distribuicao_do_proximo_token(
    logits: torch.Tensor,
    temperatura: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Logits da última posição -> probabilidades, aplicando top-k e temperatura.

    A ordem importa: o top-k corta primeiro, sobre os logits originais, e só
    então a temperatura reescala o que sobrou. Cortar depois faria o `k` incidir
    sobre valores já distorcidos.
    """
    if top_k is not None:
        logits = aplicar_top_k(logits, top_k)
    if temperatura != 1.0:
        logits = aplicar_temperatura(logits, temperatura)
    return torch.softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# Geração autoregressiva
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gerar(
    modelo: Modelo,
    idx: torch.Tensor,
    max_novos_tokens: int,
    context_size: int,
    temperatura: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Gera `max_novos_tokens` tokens a partir de `idx`, um token por vez.

    Parâmetros
    ----------
    modelo : qualquer chamável (batch, tokens) -> (batch, tokens, vocab_size).
    idx : tensor (batch, num_tokens) com a sequência inicial (o prompt).
    max_novos_tokens : quantos tokens acrescentar.
    context_size : janela máxima que o modelo aceita; a sequência é recortada
        para os últimos `context_size` tokens a cada passo.
    temperatura : 0 (padrão) = greedy, sempre o token mais provável.
        Maior que 0 ativa a amostragem.
    top_k : se informado, restringe o sorteio aos k tokens mais prováveis.
    eos_id : se informado, a geração para assim que este token é produzido.
    seed : fixa o gerador aleatório, tornando a amostragem reproduzível.

    Retorna o tensor (batch, num_tokens + n) com o prompt mais o texto gerado.

    O modelo nunca é treinado aqui: `@torch.no_grad()` desliga o cálculo de
    gradientes, que não faz sentido durante a inferência e só gastaria memória.
    """
    if max_novos_tokens < 0:
        raise ValueError("max_novos_tokens deve ser >= 0")
    if context_size < 1:
        raise ValueError("context_size deve ser >= 1")
    if seed is not None:
        torch.manual_seed(seed)

    for _ in range(max_novos_tokens):
        # 1. recorta o contexto: o modelo só enxerga os últimos context_size tokens
        idx_cond = idx[:, -context_size:]

        # 2. logits de todas as posições; só a última é a previsão do próximo token
        logits = modelo(idx_cond)
        logits = logits[:, -1, :]                       # (batch, vocab_size)

        # 3. escolhe o próximo token
        if temperatura == 0.0 and top_k is None:
            proximo = torch.argmax(logits, dim=-1, keepdim=True)          # greedy
        elif temperatura == 0.0:
            proximo = torch.argmax(aplicar_top_k(logits, top_k), dim=-1, keepdim=True)
        else:
            probabilidades = distribuicao_do_proximo_token(logits, temperatura, top_k)
            proximo = torch.multinomial(probabilidades, num_samples=1)    # amostragem

        # 4. o token escolhido passa a fazer parte do contexto da próxima rodada
        idx = torch.cat((idx, proximo), dim=1)

        if eos_id is not None and bool((proximo == eos_id).all()):
            break

    return idx


def gerar_texto_greedy(
    modelo: Modelo, idx: torch.Tensor, max_novos_tokens: int, context_size: int
) -> torch.Tensor:
    """
    Geração gulosa (greedy decoding) — a versão da seção 4.7 do livro.

    A cada passo escolhe o token de maior probabilidade. É determinística: a
    mesma entrada produz sempre a mesma saída. Atalho para
    `gerar(..., temperatura=0)`.
    """
    return gerar(modelo, idx, max_novos_tokens, context_size, temperatura=0.0)


def gerar_texto(
    modelo: Modelo,
    tokenizador,
    prompt: str,
    max_novos_tokens: int = 20,
    context_size: int = 256,
    temperatura: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    device=None,
) -> str:
    """Conveniência: texto -> texto, escondendo a conversão para tensores."""
    idx = texto_para_ids(prompt, tokenizador, device=device)
    saida = gerar(
        modelo, idx, max_novos_tokens, context_size,
        temperatura=temperatura, top_k=top_k, eos_id=eos_id, seed=seed,
    )
    return ids_para_texto(saida, tokenizador)


@torch.no_grad()
def gerar_passo_a_passo(
    modelo: Modelo,
    idx: torch.Tensor,
    max_novos_tokens: int,
    context_size: int,
    temperatura: float = 0.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Igual a `gerar`, mas devolve também o registro de cada iteração.

    Cada item do registro traz: o contexto visto pelo modelo, a forma dos logits,
    o token escolhido, a probabilidade dele e os cinco candidatos mais prováveis.
    É o que permite ao experimento 16 mostrar a autorregressão acontecendo.
    """
    if seed is not None:
        torch.manual_seed(seed)
    registro: List[Dict] = []

    for passo in range(max_novos_tokens):
        idx_cond = idx[:, -context_size:]
        logits_todos = modelo(idx_cond)
        logits = logits_todos[:, -1, :]
        # distribuição do próprio modelo, para o relatório (sem temperatura/top-k)
        probabilidades_modelo = torch.softmax(logits, dim=-1)
        # distribuição efetivamente usada na escolha
        probabilidades = (
            probabilidades_modelo
            if temperatura == 0.0
            else distribuicao_do_proximo_token(logits, temperatura, top_k)
        )

        if temperatura == 0.0:
            escolhido = torch.argmax(
                logits if top_k is None else aplicar_top_k(logits, top_k), dim=-1, keepdim=True
            )
        else:
            escolhido = torch.multinomial(probabilidades, num_samples=1)

        topo_probs, topo_ids = torch.topk(probabilidades_modelo, min(5, logits.shape[-1]), dim=-1)
        registro.append({
            "passo": passo + 1,
            "contexto": idx_cond[0].tolist(),
            "forma_logits": tuple(logits_todos.shape),
            "token_escolhido": int(escolhido[0, 0]),
            # probabilidade e entropia segundo o modelo, não segundo a amostragem
            "probabilidade": float(probabilidades_modelo[0, int(escolhido[0, 0])]),
            "probabilidade_amostragem": float(probabilidades[0, int(escolhido[0, 0])]),
            "top5_ids": [int(i) for i in topo_ids[0]],
            "top5_probs": [float(p) for p in topo_probs[0]],
            "entropia": entropia_da_distribuicao(probabilidades_modelo),
        })
        idx = torch.cat((idx, escolhido), dim=1)

    return idx, registro


# --------------------------------------------------------------------------- #
# Medidas usadas nos experimentos
# --------------------------------------------------------------------------- #
def entropia_da_distribuicao(probabilidades: torch.Tensor) -> float:
    """
    Entropia média da distribuição do próximo token, em bits.

    Mede o quanto o modelo está indeciso: 0 bit = certeza absoluta em um token;
    log2(vocab_size) = todos os tokens igualmente prováveis. É a medida usada
    para quantificar o efeito da temperatura e do top-k.
    """
    p = probabilidades.reshape(-1, probabilidades.shape[-1]).clamp_min(1e-12)
    return float(-(p * torch.log2(p)).sum(dim=-1).mean())


def tokens_para_cobrir(
    probabilidades: torch.Tensor, massa: float = 0.9, tolerancia: float = 1e-9
) -> int:
    """
    Quantos tokens são necessários para acumular `massa` de probabilidade.

    Um número baixo significa que a decisão está concentrada em poucos
    candidatos; um número alto, que a escolha está espalhada pelo vocabulário.

    A `tolerancia` existe por causa do ponto flutuante: somar dez vezes 0,1 dá
    0,8999999999999999 na nona parcela, e sem a folga o décimo token seria
    contado sem necessidade.
    """
    p = probabilidades.reshape(-1)
    ordenadas, _ = torch.sort(p, descending=True)
    acumulada = torch.cumsum(ordenadas, dim=0)
    return int((acumulada < massa - tolerancia).sum()) + 1


def taxa_de_repeticao(ids: Sequence[int]) -> float:
    """
    Fração de tokens repetidos na sequência gerada (0 = todos distintos).

    Serve para mostrar numericamente a tendência da geração gulosa a entrar em
    laço, repetindo sempre os mesmos tokens.
    """
    lista = list(ids)
    if not lista:
        return 0.0
    return 1.0 - len(set(lista)) / len(lista)
