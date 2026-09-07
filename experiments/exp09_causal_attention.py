"""
Experimento 9 — Causal attention (seção 3.5).

Objetivos:
  * provar que a máscara impede cada token de ver o futuro;
  * comparar as duas formas de mascarar: -infinito antes do softmax
    versus zerar depois e renormalizar;
  * observar o efeito do dropout sobre os pesos de atenção;
  * verificar que a máscara acompanha sequências mais curtas que o contexto.
"""

import torch
from _comum import subtitulo, tabela, titulo

from src.attention import CausalAttention, scaled_dot_product_attention

TOKENS = ["o", "velho", "consertou", "o", "relógio", "quebrado"]


def imprimir_matriz(matriz, titulo_matriz: str) -> None:
    print(f"\n{titulo_matriz}")
    print("       " + " ".join(f"{t[:6]:>7}" for t in TOKENS))
    for i, linha in enumerate(matriz):
        celulas = " ".join(
            f"{float(v):7.3f}" if float(v) != 0.0 else f"{'·':>7}" for v in linha
        )
        print(f"{TOKENS[i][:6]:>6} {celulas}")


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 9 — CAUSAL ATTENTION (MÁSCARA)")
    torch.manual_seed(123)
    n = len(TOKENS)
    lote = torch.rand(1, n, 3)          # (batch, tokens, d_in)

    subtitulo("9.1 A máscara triangular")
    mascara = torch.triu(torch.ones(n, n), diagonal=1)
    print("1 = posição bloqueada (futuro), 0 = posição permitida (passado e presente)\n")
    for i, linha in enumerate(mascara):
        print(f"{TOKENS[i][:9]:>9} " + " ".join(str(int(v)) for v in linha))

    subtitulo("9.2 Atenção sem máscara vs. com máscara")
    camada = CausalAttention(d_in=3, d_out=2, context_length=n, dropout=0.0)
    saida = camada(lote)
    com_mascara = camada.ultimo_attn_weights[0]

    q = camada.W_query(lote)[0]
    k = camada.W_key(lote)[0]
    v = camada.W_value(lote)[0]
    _, sem_mascara = scaled_dot_product_attention(q, k, v, escalar=True)

    imprimir_matriz(sem_mascara, "SEM máscara — todo token vê todo mundo:")
    imprimir_matriz(com_mascara, "COM máscara causal — o ponto marca peso zero:")
    print(f"\nsaída da camada: {tuple(saida.shape)}  (batch, tokens, d_out)")

    subtitulo("9.3 Prova numérica")
    triangulo_superior = com_mascara.triu(diagonal=1)
    somas = com_mascara.sum(dim=-1)
    print(f"maior peso acima da diagonal: {float(triangulo_superior.max()):.10f}")
    print(f"soma de cada linha: {[round(float(s), 6) for s in somas]}")
    print(f"todas as linhas somam 1: {bool(torch.allclose(somas, torch.ones(n), atol=1e-6))}")

    subtitulo("9.4 Por que não zerar depois do softmax?")
    _, pesos_livres = scaled_dot_product_attention(q, k, v, escalar=True)
    zerado = pesos_livres * (1 - mascara)
    print(f"soma das linhas ao zerar depois: {[round(float(s), 4) for s in zerado.sum(dim=-1)]}")
    renormalizado = zerado / zerado.sum(dim=-1, keepdim=True)
    print(f"soma após renormalizar ......... {[round(float(s), 4) for s in renormalizado.sum(dim=-1)]}")
    print(f"o resultado bate com o -infinito: "
          f"{bool(torch.allclose(renormalizado, com_mascara, atol=1e-5))}")

    subtitulo("9.5 Efeito do dropout sobre os pesos")
    linhas = []
    for taxa in (0.0, 0.1, 0.2, 0.5):
        camada_d = CausalAttention(3, 2, n, dropout=taxa)
        camada_d.train()                      # dropout só age em modo de treinamento
        torch.manual_seed(42)
        camada_d(lote)
        pesos_treino = camada_d.ultimo_attn_weights[0]
        torch.manual_seed(42)
        aplicado = camada_d.dropout(pesos_treino)
        zerados = int((aplicado == 0).sum()) - int((pesos_treino == 0).sum())
        linhas.append([
            f"{taxa:.0%}", f"{zerados}", f"{float(aplicado.max()):.3f}",
            f"{float(aplicado.sum(dim=-1).mean()):.3f}",
        ])
    tabela(["taxa de dropout", "pesos zerados a mais", "maior peso", "soma média por linha"], linhas)

    subtitulo("9.6 Sequência mais curta que o contexto")
    camada_longa = CausalAttention(3, 2, context_length=16, dropout=0.0)
    for tamanho in (2, 4, 6):
        entrada = torch.rand(1, tamanho, 3)
        camada_longa(entrada)
        pesos = camada_longa.ultimo_attn_weights[0]
        print(f"tokens={tamanho:>2}  matriz={tuple(pesos.shape)}  "
              f"maior peso no futuro={float(pesos.triu(1).max()):.1e}")


if __name__ == "__main__":
    main()
