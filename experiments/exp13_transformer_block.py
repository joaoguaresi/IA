"""
Experimento 13 — Transformer Block (Entregável 2).

Objetivos:
  * observar o efeito do LayerNorm e da GELU isoladamente;
  * acompanhar as dimensões dos tensores em cada etapa do bloco;
  * mostrar que a conexão residual (shortcut) preserva o sinal quando as
    sub-camadas internas não contribuem com nada;
  * medir o efeito do dropout no shortcut sobre a variância da saída.
"""

import torch
from _comum import subtitulo, tabela, titulo

from src.attention import numero_de_parametros
from src.transformer import GELU, FeedForward, LayerNorm, TransformerBlock

EMB_DIM, N_TOKENS, N_HEADS = 8, 6, 2


@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 13 — TRANSFORMER BLOCK")
    torch.manual_seed(123)

    subtitulo("13.1 Layer Normalization")
    # escala e deslocamento propositalmente longe de média 0 / variância 1
    x = torch.randn(2, N_TOKENS, EMB_DIM) * 17.0 + 3.0
    norm = LayerNorm(EMB_DIM)
    saida = norm(x)
    print(f"entrada  -> média por token: {[round(float(v), 2) for v in x[0].mean(dim=-1)]}")
    print(f"           variância por token: {[round(float(v), 2) for v in x[0].var(dim=-1, unbiased=False)]}")
    print(f"depois   -> média por token: {[round(float(v), 4) for v in saida[0].mean(dim=-1)]}")
    print(f"           variância por token: {[round(float(v), 4) for v in saida[0].var(dim=-1, unbiased=False)]}")
    print("(cada token é normalizado individualmente, ao longo de emb_dim — "
          "não existe mistura entre tokens nem entre amostras do lote)")

    subtitulo("13.2 GELU vs. ReLU")
    gelu = GELU()
    relu = torch.nn.ReLU()
    valores = torch.linspace(-3, 3, 7)
    tabela(
        ["x", "ReLU(x)", "GELU(x)"],
        [[f"{float(v):.2f}", f"{float(relu(v)):.4f}", f"{float(gelu(v)):.4f}"] for v in valores],
    )
    print("a ReLU zera tudo abaixo de 0 (corte abrupto); a GELU deixa passar uma "
          "fração pequena dos valores negativos, numa transição suave")

    subtitulo("13.3 FeedForward: expansão de 4x")
    ff = FeedForward(EMB_DIM)
    x_ff = torch.rand(2, N_TOKENS, EMB_DIM)
    saida_ff = ff(x_ff)
    print(f"entrada ................. {tuple(x_ff.shape)}")
    print(f"após 1ª camada (4x) ...... ({2}, {N_TOKENS}, {4 * EMB_DIM})  [intermediário, não retornado]")
    print(f"saída (de volta a emb_dim) {tuple(saida_ff.shape)}")
    print(f"parâmetros ............... {numero_de_parametros(ff):,}  "
          f"= 8·emb_dim² + 5·emb_dim  (pesos + viés das duas camadas lineares)")

    subtitulo("13.4 O Transformer Block completo")
    bloco = TransformerBlock(EMB_DIM, N_TOKENS, N_HEADS, drop_rate=0.0)
    x_bloco = torch.rand(2, N_TOKENS, EMB_DIM)
    saida_bloco = bloco(x_bloco)
    print(f"entrada .................. {tuple(x_bloco.shape)}  (batch, tokens, emb_dim)")
    print(f"saída .................... {tuple(saida_bloco.shape)}  (idêntica à entrada)")
    print(f"parâmetros totais ........ {numero_de_parametros(bloco):,}")
    print(f"  - atenção .............. {numero_de_parametros(bloco.atencao):,}")
    print(f"  - feed-forward ......... {numero_de_parametros(bloco.feed_forward):,}")
    print(f"  - 2× layernorm ......... {numero_de_parametros(bloco.norm1) + numero_de_parametros(bloco.norm2):,}")

    subtitulo("13.5 A conexão residual preserva o sinal")
    bloco_zerado = TransformerBlock(EMB_DIM, N_TOKENS, N_HEADS, drop_rate=0.0)
    for p in bloco_zerado.atencao.parameters():
        p.zero_()
    for p in bloco_zerado.feed_forward.parameters():
        p.zero_()
    bloco_zerado.norm1.scale.zero_()
    bloco_zerado.norm2.scale.zero_()
    saida_zerada = bloco_zerado(x_bloco)
    print("com todos os pesos das sub-camadas zerados (atenção, feed-forward e o "
          "'scale' do LayerNorm), cada sub-camada contribui exatamente 0.")
    print(f"saída == entrada: {bool(torch.allclose(saida_zerada, x_bloco, atol=1e-6))}")
    print("(é exatamente isso que a conexão residual garante: mesmo que uma "
          "camada aprenda a não fazer nada útil, o sinal ainda atravessa o bloco)")

    subtitulo("13.6 Pesos de atenção expostos pelo bloco")
    bloco(x_bloco)
    pesos = bloco.ultimo_attn_weights
    print(f"ultimo_attn_weights: {tuple(pesos.shape)}  (batch, heads, tokens, tokens)")
    print("(o TransformerBlock repassa os pesos da sua MultiHeadAttention interna, "
          "para permitir visualização sem recalcular — mesmo contrato da Sprint 3)")

    subtitulo("13.7 Dropout no shortcut: variância entre chamadas")
    linhas = []
    for taxa in (0.0, 0.1, 0.3, 0.5):
        bloco_d = TransformerBlock(EMB_DIM, N_TOKENS, N_HEADS, drop_rate=taxa)
        bloco_d.train()  # dropout só age em modo de treinamento
        torch.manual_seed(7)
        saidas = torch.stack([bloco_d(x_bloco) for _ in range(20)])
        linhas.append([f"{taxa:.0%}", f"{float(saidas.var(dim=0).mean()):.5f}"])
    tabela(["drop_rate", "variância média entre 20 chamadas"], linhas)
    print("com drop_rate=0, o bloco é determinístico (variância 0); taxas maiores "
          "tornam a saída mais ruidosa de uma chamada para a outra, em modo de "
          "treinamento — exatamente o comportamento esperado do dropout.")


if __name__ == "__main__":
    main()
