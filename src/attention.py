"""
Sprint 3 — Mecanismos de atenção (Capítulo 3).

Implementa as quatro variantes do capítulo, na ordem em que o livro as constrói:

    1. Self-attention simplificado  ......... sem pesos treináveis (seção 3.3)
    2. Scaled dot-product attention ......... com Wq, Wk, Wv treináveis (seção 3.4)
    3. Causal attention ..................... máscara + dropout (seção 3.5)
    4. Multi-head attention ................. várias cabeças em paralelo (seção 3.6)

Entrada esperada (produzida pela Sprint 2):

    input_embeddings  ->  (batch, num_tokens, d_in)

Saída de todas as variantes:

    context_vectors   ->  (batch, num_tokens, d_out)

Todos os módulos guardam os pesos de atenção da última chamada em
`ultimo_attn_weights`, para permitir a visualização das matrizes sem recalcular.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1. Self-attention simplificado (seção 3.3) — sem pesos treináveis
# --------------------------------------------------------------------------- #
def attention_scores_simples(inputs: torch.Tensor) -> torch.Tensor:
    """
    Scores de atenção via produto escalar de cada token com todos os outros.

    `inputs` tem forma (num_tokens, d_in) e a saída (num_tokens, num_tokens).
    O elemento [i, j] mede o quanto o token i "combina" com o token j: quanto
    maior o produto escalar, maior a similaridade entre os dois embeddings.
    """
    return inputs @ inputs.T


def self_attention_simples(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Self-attention sem pesos treináveis (versão didática da seção 3.3).

    Três passos:
      1. scores  = inputs @ inputs.T          (similaridade entre tokens)
      2. weights = softmax(scores, dim=-1)    (cada linha soma 1)
      3. context = weights @ inputs           (média ponderada dos tokens)

    Devolve (context_vectors, attention_weights), ambos calculados sem nenhum
    parâmetro aprendido — é a intuição do mecanismo, não a versão usada no GPT.
    """
    scores = attention_scores_simples(inputs)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ inputs
    return context, weights


def context_vector_passo_a_passo(
    inputs: torch.Tensor, indice_query: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    O mesmo cálculo para UM token de consulta, explicitando cada etapa.

    Serve para o experimento 7: mostrar os números intermediários de um único
    context vector, como a Figura 3.3 do capítulo.
    """
    query = inputs[indice_query]                     # (d_in,)
    scores = torch.empty(inputs.shape[0])            # (num_tokens,)
    for i, entrada in enumerate(inputs):
        scores[i] = torch.dot(query, entrada)        # produto escalar
    weights = torch.softmax(scores, dim=0)           # normaliza: soma 1
    context = (weights.unsqueeze(-1) * inputs).sum(dim=0)
    return context, weights, scores


# --------------------------------------------------------------------------- #
# 2. Scaled dot-product attention (seção 3.4) — com pesos treináveis
# --------------------------------------------------------------------------- #
class SelfAttention_v1(nn.Module):
    """
    Self-attention com pesos treináveis usando `nn.Parameter` (seção 3.4.2).

    Wq, Wk e Wv projetam cada embedding em query, key e value. Os scores são
    divididos por raiz de d_k — daí o nome *scaled* dot-product attention.
    """

    def __init__(self, d_in: int, d_out: int, seed: Optional[int] = 123) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        self.d_out = d_out
        self.ultimo_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        self.ultimo_attn_weights = attn_weights.detach()
        return attn_weights @ values


class SelfAttention_v2(nn.Module):
    """
    Mesma matemática da V1, porém com `nn.Linear` (seção 3.4.2).

    O livro prefere esta versão: `nn.Linear` usa uma inicialização de pesos
    melhor (Kaiming uniforme) que `torch.rand`, o que estabiliza o treinamento.
    Com `qkv_bias=False`, a camada faz exatamente a mesma multiplicação da V1.
    """

    def __init__(
        self, d_in: int, d_out: int, qkv_bias: bool = False, seed: Optional[int] = 123
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        self.ultimo_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        self.ultimo_attn_weights = attn_weights.detach()
        return attn_weights @ values


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mascara: Optional[torch.Tensor] = None,
    escalar: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A operação de atenção em forma de função, para os experimentos.

        attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V

    `escalar=False` remove a divisão por sqrt(d_k), permitindo medir o efeito
    da escala sobre a distribuição dos pesos (experimento 8).
    `mascara` é um tensor booleano True nas posições que devem ser bloqueadas.
    """
    d_k = keys.shape[-1]
    scores = queries @ keys.transpose(-2, -1)
    if escalar:
        scores = scores / d_k**0.5
    if mascara is not None:
        scores = scores.masked_fill(mascara, -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    return weights @ values, weights


# --------------------------------------------------------------------------- #
# 3. Causal attention (seção 3.5)
# --------------------------------------------------------------------------- #
class CausalAttention(nn.Module):
    """
    Atenção causal: cada token só enxerga a si mesmo e aos tokens anteriores.

    Os tokens futuros recebem -infinito ANTES do softmax. Como e^(-inf) = 0,
    eles saem do softmax com peso exatamente zero e a linha continua somando 1,
    sem precisar de renormalização.

    A máscara é registrada com `register_buffer`: ela acompanha o modelo quando
    ele vai para a GPU, mas não é um parâmetro treinável.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self.ultimo_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        self.ultimo_attn_weights = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values


# --------------------------------------------------------------------------- #
# 4. Multi-head attention (seção 3.6)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionWrapper(nn.Module):
    """
    Versão ingênua (seção 3.6.1): empilha várias `CausalAttention` independentes
    e concatena os context vectors na última dimensão.

    Simples de entender, mas ineficiente: as cabeças são processadas em
    sequência, uma multiplicação de matriz por cabeça.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        num_heads: int = 2,
        qkv_bias: bool = False,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in, d_out, context_length, dropout, qkv_bias,
                    seed=None if seed is None else seed + i,
                )
                for i in range(num_heads)
            ]
        )
        self.num_heads = num_heads
        self.d_out_total = d_out * num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)

    @property
    def ultimo_attn_weights(self) -> torch.Tensor:
        """Empilha os pesos das cabeças: (batch, num_heads, tokens, tokens)."""
        return torch.stack([h.ultimo_attn_weights for h in self.heads], dim=1)


class MultiHeadAttention(nn.Module):
    """
    Implementação eficiente (seção 3.6.2) — a que segue para a Sprint 4.

    Em vez de uma instância por cabeça, faz UMA multiplicação de matriz para
    Q, K e V e divide o resultado em cabeças por reshape:

        (b, tokens, d_out) -> (b, tokens, num_heads, head_dim) -> (b, num_heads, tokens, head_dim)

    Depois do cálculo, as cabeças são reconsolidadas e passam por uma camada
    linear de saída (`out_proj`), que mistura o que cada cabeça aprendeu.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        num_heads: int = 2,
        qkv_bias: bool = False,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(
                f"d_out ({d_out}) precisa ser divisível por num_heads ({num_heads}): "
                "cada cabeça fica com d_out / num_heads dimensões."
            )
        if seed is not None:
            torch.manual_seed(seed)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self.ultimo_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)                      # (b, tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        # divide a última dimensão em (num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, tokens, heads, head_dim) -> (b, heads, tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)   # (b, heads, tokens, tokens)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        self.ultimo_attn_weights = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ values                # (b, heads, tokens, head_dim)
        context = context.transpose(1, 2)              # (b, tokens, heads, head_dim)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)


# --------------------------------------------------------------------------- #
# Utilidades de análise
# --------------------------------------------------------------------------- #
def entropia_dos_pesos(attn_weights: torch.Tensor) -> float:
    """
    Entropia média das linhas da matriz de atenção, em bits.

    Entropia alta = atenção espalhada por muitos tokens.
    Entropia baixa = atenção concentrada em poucos tokens (softmax saturado).
    É a medida usada no experimento da escala por raiz de d_k.
    """
    p = attn_weights.reshape(-1, attn_weights.shape[-1]).clamp_min(1e-12)
    return float(-(p * torch.log2(p)).sum(dim=-1).mean())


def numero_de_parametros(modulo: nn.Module) -> int:
    """Total de parâmetros treináveis de um módulo de atenção."""
    return sum(p.numel() for p in modulo.parameters() if p.requires_grad)
