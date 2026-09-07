"""
Experimento 7 — Self-attention simplificado (seção 3.3).

Objetivos:
  * ver o context vector sendo construído passo a passo para um token;
  * confirmar que cada linha da matriz de atenção soma 1;
  * observar como diferentes sequências de entrada mudam a matriz.
"""

import torch
from _comum import subtitulo, tabela, titulo

from src.attention import (
    attention_scores_simples,
    context_vector_passo_a_passo,
    self_attention_simples,
)

# Seis tokens de exemplo, com embeddings pequenos (3 dimensões) para caber na tela.
TOKENS = ["o", "velho", "consertou", "o", "relógio", "quebrado"]


def embeddings_exemplo(seed: int = 123) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(len(TOKENS), 3)


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 7 — SELF-ATTENTION SIMPLIFICADO (SEM PESOS TREINÁVEIS)")
    inputs = embeddings_exemplo()
    print(f"sequência: {' '.join(TOKENS)}")
    print(f"embeddings de entrada: {tuple(inputs.shape)}  (tokens, d_in)")

    subtitulo("7.1 Passo a passo para um único token de consulta")
    indice = 2  # "consertou"
    context, weights, scores = context_vector_passo_a_passo(inputs, indice)
    print(f"query = token {indice} ({TOKENS[indice]!r})\n")
    tabela(
        ["token", "score (produto escalar)", "peso após softmax"],
        [
            [t, f"{float(s):.4f}", f"{float(w):.4f}"]
            for t, s, w in zip(TOKENS, scores, weights)
        ],
    )
    print(f"\nsoma dos pesos: {float(weights.sum()):.6f}  (o softmax garante 1)")
    print(f"context vector: {[round(float(v), 4) for v in context]}")

    subtitulo("7.2 Todos os context vectors de uma vez (multiplicação de matrizes)")
    contexts, matriz = self_attention_simples(inputs)
    print(f"scores  = inputs @ inputs.T -> {tuple(attention_scores_simples(inputs).shape)}")
    print(f"weights = softmax(scores)   -> {tuple(matriz.shape)}")
    print(f"context = weights @ inputs  -> {tuple(contexts.shape)}")
    print(f"\nlinha {indice} da versão matricial confere com o passo a passo: "
          f"{bool(torch.allclose(contexts[indice], context, atol=1e-6))}")

    subtitulo("7.3 A matriz de atenção")
    print("     " + "  ".join(f"{t[:7]:>7}" for t in TOKENS))
    for i, linha in enumerate(matriz):
        print(f"{TOKENS[i][:4]:>4} " + "  ".join(f"{float(v):7.3f}" for v in linha))
    somas = matriz.sum(dim=-1)
    print(f"\nsoma de cada linha: {[round(float(s), 4) for s in somas]}")

    subtitulo("7.4 Sequências de entrada diferentes")
    linhas = []
    for n in (4, 6, 10, 32):
        torch.manual_seed(123)
        entrada = torch.rand(n, 3)
        _, pesos = self_attention_simples(entrada)
        maior = float(pesos.max())
        diagonal = float(pesos.diagonal().mean())
        linhas.append([n, str(tuple(pesos.shape)), f"{maior:.3f}", f"{diagonal:.3f}",
                       f"{1.0 / n:.3f}"])
    tabela(
        ["nº de tokens", "forma da matriz", "maior peso", "peso médio na diagonal",
         "peso se fosse uniforme"],
        linhas,
    )


if __name__ == "__main__":
    main()
