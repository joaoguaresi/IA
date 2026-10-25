"""
Sprint 4 — Transformer Block e arquitetura GPT completa (Entregáveis 2 e 3).

Corresponde às seções 4.2 a 4.6 do Capítulo 4: Layer Normalization, GELU e a
rede FeedForward, conexões residuais (shortcut) e o Transformer Block que
integra tudo isso ao redor do Multi-Head Attention da Sprint 3 — e, por fim,
o empilhamento desses blocos na arquitetura GPT completa.

Fluxo implementado:

    Token IDs (batch, tokens)
      -> GPTEmbedding (token + positional embeddings)          (Sprint 2)
      -> dropout
      -> N x TransformerBlock                                  (Entregável 2)
      -> LayerNorm final
      -> out_head (Linear sem bias)
      -> logits (batch, tokens, vocab_size)                    (Entregável 3)

Cada TransformerBlock, por sua vez:

    x -> LayerNorm -> MultiHeadAttention -> dropout -> + x        (shortcut 1)
      -> LayerNorm -> FeedForward(GELU)  -> dropout -> + resultado (shortcut 2)

As conexões residuais somam a entrada de cada sub-camada à sua própria saída,
criando um "atalho" para o gradiente fluir direto até camadas anteriores
durante a retropropagação — sem elas, redes profundas sofrem do problema do
gradiente desaparecendo (vanishing gradient) e ficam difíceis de treinar.

Entrada e saída de um TransformerBlock (e, portanto, de um GPTModel inteiro)
têm sempre a mesma forma (batch, tokens, emb_dim): é essa consistência de
forma que permite empilhar quantos blocos quiser sem nenhum ajuste de
dimensão entre eles.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn

from .attention import MultiHeadAttention, numero_de_parametros
from .embeddings import GPTEmbedding


# --------------------------------------------------------------------------- #
# 1. Layer Normalization (seção 4.2)
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    """
    Normaliza cada token (não o lote) para média 0 e variância 1, seguida de
    uma escala e um deslocamento aprendidos (`scale` e `shift`).

    Diferente do Batch Normalization, que normaliza entre as amostras de um
    lote, o Layer Normalization normaliza ao longo da última dimensão
    (`emb_dim`) — por isso se comporta igual em treino e em inferência e não
    depende do tamanho do lote, o que o torna adequado para sequências de
    comprimento variável como as de um LLM.

    `scale` e `shift` são inicializados como identidade (1 e 0): no início do
    treinamento a camada não altera nada além da normalização; é o
    treinamento que aprende o quanto cada dimensão deve ser reescalada.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        media = x.mean(dim=-1, keepdim=True)
        # unbiased=False: divide por n, não por n-1 — a mesma convenção do GPT-2
        variancia = x.var(dim=-1, keepdim=True, unbiased=False)
        normalizado = (x - media) / torch.sqrt(variancia + self.eps)
        return self.scale * normalizado + self.shift


# --------------------------------------------------------------------------- #
# 2. GELU e a rede FeedForward (seção 4.3)
# --------------------------------------------------------------------------- #
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit — a função de ativação usada pelos modelos GPT
    no lugar da ReLU.

    A ReLU zera tudo que é negativo em um corte abrupto (derivada zero para
    sempre abaixo de zero, o problema dos "neurônios mortos"). A GELU é suave:
    pondera cada entrada pela probabilidade de ela "passar", deixando fluir
    uma fração pequena, mas não nula, dos valores negativos.

    Implementa a aproximação usada no GPT-2 original e no livro (baseada em
    tanh), mais barata de calcular que a GELU exata, que depende da função
    erro (erf).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """
    Rede de duas camadas lineares com expansão interna de 4x (seção 4.3).

        emb_dim -> 4*emb_dim -> GELU -> emb_dim

    A expansão temporária para um espaço de maior dimensão dá à rede mais
    "espaço" para combinar, token a token, as informações que a atenção
    trouxe de outras posições, antes de ser comprimida de volta à dimensão
    original — exigida pela conexão residual logo em seguida.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.camadas(x)


# --------------------------------------------------------------------------- #
# 3. Transformer Block (seção 4.5) — Entregável 2
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """
    Integra Multi-Head Attention (Sprint 3), Layer Normalization, FeedForward
    e conexões residuais em um único bloco reutilizável.

    Duas sub-camadas, cada uma seguindo o padrão
    "pré-normalização -> transformação -> dropout -> soma com a entrada":

        1. x        -> LayerNorm -> MultiHeadAttention -> dropout -> + x
        2. resultado -> LayerNorm -> FeedForward        -> dropout -> + resultado

    A atenção mistura informação ENTRE tokens (cada posição passa a depender
    das demais); a FeedForward processa cada token individualmente, já com
    essa informação incorporada. Entrada e saída têm exatamente a mesma forma
    (batch, tokens, emb_dim) — é o que permite empilhar vários blocos em
    sequência sem nenhum ajuste de dimensão entre eles.
    """

    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        num_heads: int,
        drop_rate: float = 0.0,
        qkv_bias: bool = False,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        self.atencao = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            dropout=drop_rate,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            seed=seed,
        )
        self.feed_forward = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atalho = x
        x = self.norm1(x)
        x = self.atencao(x)
        x = self.drop_shortcut(x)
        x = x + atalho  # 1ª conexão residual: ao redor da atenção

        atalho = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        return x + atalho  # 2ª conexão residual: ao redor da FeedForward

    @property
    def ultimo_attn_weights(self) -> Optional[torch.Tensor]:
        """Repassa os pesos de atenção da última chamada (para visualização)."""
        return self.atencao.ultimo_attn_weights


# --------------------------------------------------------------------------- #
# 4. Modelo GPT completo (seção 4.6) — Entregável 3
# --------------------------------------------------------------------------- #
class GPTModel(nn.Module):
    """
    Arquitetura GPT simplificada: embeddings -> N TransformerBlocks ->
    normalização final -> camada de saída (logits).

        Token IDs (batch, tokens)
          -> GPTEmbedding (token + positional embeddings, Sprint 2)
          -> dropout
          -> N x TransformerBlock
          -> LayerNorm final
          -> out_head (Linear sem bias, emb_dim -> vocab_size)
          -> logits (batch, tokens, vocab_size)

    `vocab_size` e `context_length` formam o "contrato" externo do modelo
    (precisam bater com o tokenizador e com a janela de contexto usada no
    treino); os demais parâmetros controlam a capacidade interna. Os valores
    padrão reproduzem a configuração GPT_CONFIG_124M do livro (GPT-2 small):
    12 camadas, 12 cabeças, 768 dimensões de embedding.

    `out_head` não tem viés (`bias=False`), seguindo a convenção do GPT-2: o
    viés nessa camada ajudaria pouco (o vocabulário já é bem representado
    pelos pesos) e economiza `vocab_size` parâmetros.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int = 768,
        n_layers: int = 12,
        num_heads: int = 12,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.num_heads = num_heads

        self.embeddings = GPTEmbedding(vocab_size, emb_dim, context_length, seed=seed)
        self.drop_emb = nn.Dropout(drop_rate)
        self.blocos = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim,
                    context_length,
                    num_heads,
                    drop_rate,
                    qkv_bias,
                    seed=None if seed is None else seed + i,
                )
                for i in range(n_layers)
            ]
        )
        self.norm_final = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(in_idx)
        x = self.drop_emb(x)
        for bloco in self.blocos:
            x = bloco(x)
        x = self.norm_final(x)
        return self.out_head(x)

    @property
    def ultimo_attn_weights(self) -> List[Optional[torch.Tensor]]:
        """Pesos de atenção da última chamada, um tensor por bloco."""
        return [bloco.ultimo_attn_weights for bloco in self.blocos]

    @property
    def nome(self) -> str:
        return (
            f"GPTModel(emb={self.emb_dim}, heads={self.num_heads}, "
            f"camadas={self.n_layers}, contexto={self.context_length})"
        )


# --------------------------------------------------------------------------- #
# Utilidades de análise
# --------------------------------------------------------------------------- #
def memoria_em_mb(modulo: nn.Module, bytes_por_parametro: int = 4) -> float:
    """
    Memória aproximada ocupada pelos parâmetros treináveis do módulo, em MB.

    Usa `numero_de_parametros` (Sprint 3) e assume float32 (4 bytes) por
    padrão — a mesma convenção de `tamanho_em_mb` em `embeddings.py`, agora
    aplicada ao modelo inteiro, não só à matriz de embeddings.
    """
    return numero_de_parametros(modulo) * bytes_por_parametro / (1024**2)


def parametros_por_componente(modelo: GPTModel) -> dict:
    """
    Quebra o total de parâmetros do modelo por componente estrutural.

    Útil para observar, por exemplo, que a matriz de token embeddings e a
    camada de saída dominam o total em modelos com vocabulário grande e
    poucas camadas, enquanto os blocos Transformer dominam em modelos com
    muitas camadas — o mesmo tipo de análise já feito para a embedding layer
    na Sprint 2, agora estendido ao modelo completo.
    """
    return {
        "token_embeddings": numero_de_parametros(modelo.embeddings.tok_emb),
        "positional_embeddings": numero_de_parametros(modelo.embeddings.pos_emb),
        "transformer_blocks": sum(numero_de_parametros(b) for b in modelo.blocos),
        "layernorm_final": numero_de_parametros(modelo.norm_final),
        "out_head": numero_de_parametros(modelo.out_head),
        "total": numero_de_parametros(modelo),
    }
