"""
Experimento 12 — Integração com a Sprint 2 (texto → atenção).

Encadeia o pipeline inteiro construído até aqui:

    texto -> tokenização -> Token IDs -> lotes -> embeddings + posição
          -> MULTI-HEAD ATTENTION -> context vectors

É esta a saída que a Sprint 4 vai empilhar dentro dos blocos Transformer.
"""

import torch
from _comum import subtitulo, tabela, titulo

from src.attention import MultiHeadAttention, numero_de_parametros
from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto, nome_do_corpus
from src.dataset import create_dataloader_v1
from src.embeddings import GPTEmbedding


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 12 — DA SPRINT 2 PARA A ATENÇÃO")

    texto = carregar_texto()
    tokenizador = obter_tokenizador_bpe()
    context_length, output_dim, batch_size, num_heads = 8, 64, 4, 4

    subtitulo("12.1 Etapas da Sprint 2")
    dataloader = create_dataloader_v1(
        texto, tokenizer=tokenizador, batch_size=batch_size,
        max_length=context_length, stride=context_length, shuffle=False,
    )
    entradas, alvos = next(iter(dataloader))
    camada_emb = GPTEmbedding(
        vocab_size=tokenizador.n_vocab, output_dim=output_dim,
        context_length=context_length, seed=123,
    )
    input_embeddings = camada_emb(entradas)

    print(f"corpus ................... {nome_do_corpus()}")
    print(f"tokenizador .............. {tokenizador.nome}")
    print(f"lote de Token IDs ........ {tuple(entradas.shape)}")
    print(f"alvos .................... {tuple(alvos.shape)}")
    print(f"input embeddings ......... {tuple(input_embeddings.shape)}")

    subtitulo("12.2 Passando pela atenção")
    atencao = MultiHeadAttention(
        d_in=output_dim, d_out=output_dim, context_length=context_length,
        dropout=0.0, num_heads=num_heads,
    )
    context_vectors = atencao(input_embeddings)
    pesos = atencao.ultimo_attn_weights

    print(f"entrada da atenção ....... {tuple(input_embeddings.shape)}")
    print(f"matrizes de atenção ...... {tuple(pesos.shape)}  (batch, heads, tokens, tokens)")
    print(f"context vectors .......... {tuple(context_vectors.shape)}")
    print(f"parâmetros da atenção .... {numero_de_parametros(atencao):,}")

    subtitulo("12.3 O que mudou dentro dos vetores")
    antes = input_embeddings[0, 0]
    depois = context_vectors[0, 0]
    print(f"token 0 antes da atenção (5 primeiras dims): {[round(float(v), 3) for v in antes[:5]]}")
    print(f"token 0 depois da atenção .................: {[round(float(v), 3) for v in depois[:5]]}")
    print(f"distância entre os dois ...................: "
          f"{float(torch.norm(depois - antes)):.4f}")

    subtitulo("12.4 O primeiro token é um caso especial")
    primeira_linha = pesos[0, 0, 0]
    print(f"pesos da linha 0 (cabeça 1): {[round(float(v), 3) for v in primeira_linha]}")

    subtitulo("12.5 Varredura de configurações")
    linhas = []
    for ctx, dim, heads, lote in ((4, 32, 2, 2), (8, 64, 4, 4), (16, 128, 8, 4), (32, 256, 8, 2)):
        dl = create_dataloader_v1(
            texto, tokenizer=tokenizador, batch_size=lote,
            max_length=ctx, stride=ctx, shuffle=False,
        )
        x, _ = next(iter(dl))
        emb = GPTEmbedding(tokenizador.n_vocab, dim, ctx, seed=123)
        att = MultiHeadAttention(dim, dim, ctx, dropout=0.0, num_heads=heads)
        saida = att(emb(x))
        celulas = lote * heads * ctx * ctx
        linhas.append([
            ctx, dim, heads, dim // heads, lote, str(tuple(saida.shape)),
            f"{numero_de_parametros(att):,}", f"{celulas:,}",
        ])
    tabela(
        ["contexto", "d_out", "cabeças", "head_dim", "batch", "context vectors",
         "params atenção", "células de atenção"],
        linhas,
    )


if __name__ == "__main__":
    main()
