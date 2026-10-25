"""
Experimento 16 — Geração autoregressiva (Entregável 4 / parte do 5).

Observa o laço de geração token a token: logits -> probabilidades -> escolha ->
realimentação do contexto. Investiga sequências iniciais diferentes, recorte de
contexto, comprimentos de geração e o comportamento determinístico do greedy.
"""

import time

import torch
from _comum import subtitulo, tabela, titulo
from _modelos_para_geracao import ModeloBigrama, obter_modelo_gpt

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto
from src.geracao import (
    gerar,
    gerar_passo_a_passo,
    gerar_texto,
    gerar_texto_greedy,
    ids_para_texto,
    taxa_de_repeticao,
    texto_para_ids,
)

PROMPT = "O velho consertou"


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 16 — GERAÇÃO AUTOREGRESSIVA")

    tokenizador = obter_tokenizador_bpe()
    texto = carregar_texto()
    bigrama = ModeloBigrama(tokenizador.encode(texto), tokenizador.n_vocab)
    print(f"tokenizador: {tokenizador.nome}  |  vocabulário: {tokenizador.n_vocab:,}")
    print(f"modelo de referência: {bigrama.nome}")

    subtitulo("16.1 O laço passo a passo")
    idx = texto_para_ids(PROMPT, tokenizador)
    print(f"prompt: {PROMPT!r}  ->  ids {idx[0].tolist()}  forma {tuple(idx.shape)}")
    _, registro = gerar_passo_a_passo(bigrama, idx, 6, context_size=16)
    tabela(
        ["passo", "contexto (ids)", "logits do modelo", "token escolhido", "texto",
         "probabilidade", "entropia (bits)"],
        [
            [r["passo"], str(r["contexto"][-3:]), str(r["forma_logits"]),
             r["token_escolhido"], repr(tokenizador.decode([r["token_escolhido"]])),
             f"{r['probabilidade']:.4f}", f"{r['entropia']:.3f}"]
            for r in registro
        ],
    )

    subtitulo("16.2 Os cinco candidatos de cada passo")
    for r in registro[:4]:
        candidatos = ", ".join(
            f"{tokenizador.decode([i])!r}={p:.3f}"
            for i, p in zip(r["top5_ids"], r["top5_probs"])
        )
        print(f"passo {r['passo']}: {candidatos}")

    subtitulo("16.3 De logits a probabilidades")
    logits = bigrama(idx)
    ultimo = logits[:, -1, :]
    probabilidades = torch.softmax(ultimo, dim=-1)
    print(f"logits de todas as posições ... {tuple(logits.shape)}  (batch, tokens, vocab)")
    print(f"logits da última posição ...... {tuple(ultimo.shape)}  (batch, vocab)")
    print(f"soma das probabilidades ....... {float(probabilidades.sum()):.6f}")
    print(f"maior probabilidade ........... {float(probabilidades.max()):.4f}")
    print(f"menor probabilidade ........... {float(probabilidades.min()):.3e}")
    print(f"argmax(logits) == argmax(probs): "
          f"{int(torch.argmax(ultimo)) == int(torch.argmax(probabilidades))}")

    subtitulo("16.4 Greedy é determinístico")
    saidas = [
        ids_para_texto(gerar_texto_greedy(bigrama, idx, 15, 16), tokenizador)
        for _ in range(3)
    ]
    for i, s in enumerate(saidas, 1):
        print(f"execução {i}: {s!r}")
    print(f"as três são idênticas: {len(set(saidas)) == 1}")

    subtitulo("16.5 Sequências iniciais diferentes")
    linhas = []
    for prompt in ("O velho", "A biblioteca", "O relógio parou", "Na rua das"):
        entrada = texto_para_ids(prompt, tokenizador)
        saida = gerar_texto_greedy(bigrama, entrada, 12, 16)
        gerados = saida[0, entrada.shape[1]:].tolist()
        linhas.append([
            repr(prompt), entrada.shape[1], tuple(saida.shape),
            repr(ids_para_texto(saida, tokenizador)[:52]),
            f"{taxa_de_repeticao(gerados):.2f}",
        ])
    tabela(["prompt", "tokens do prompt", "forma final", "texto gerado",
            "repetição"], linhas)

    subtitulo("16.6 Comprimento da geração")
    linhas = []
    for n in (5, 10, 25, 50, 100):
        inicio = time.perf_counter()
        saida = gerar_texto_greedy(bigrama, idx, n, 16)
        ms = (time.perf_counter() - inicio) * 1000
        gerados = saida[0, idx.shape[1]:].tolist()
        linhas.append([
            n, tuple(saida.shape), f"{ms:7.1f} ms", f"{ms / n:5.2f} ms",
            len(set(gerados)), f"{taxa_de_repeticao(gerados):.2f}",
        ])
    tabela(["tokens gerados", "forma final", "tempo total", "por token",
            "tokens distintos", "repetição"], linhas)

    subtitulo("16.7 Recorte do contexto")
    context_size = 8
    modelo = obter_modelo_gpt(tokenizador.n_vocab, context_size, emb_dim=64, num_heads=4)
    print(f"modelo: {getattr(modelo, 'nome', type(modelo).__name__)}")
    print(f"context_size = {context_size}")
    prompt_longo = texto_para_ids(
        "O velho consertava relógios havia quarenta anos naquela rua estreita", tokenizador
    )
    print(f"prompt com {prompt_longo.shape[1]} tokens (maior que o contexto)")
    saida = gerar(modelo, prompt_longo, 5, context_size)
    print(f"forma final: {tuple(saida.shape)}  "
          f"(prompt {prompt_longo.shape[1]} + 5 gerados)")
    print("a sequência cresce, mas o modelo só recebe os últimos "
          f"{context_size} tokens a cada passo")

    subtitulo("16.8 Modelo sem treinamento vs. modelo de contagem")
    linhas = []
    for rotulo, m, ctx in (
        ("ModeloProvisorio (pesos aleatórios)", modelo, context_size),
        ("ModeloBigrama (contagens do corpus)", bigrama, 16),
    ):
        texto_gerado = gerar_texto(m, tokenizador, PROMPT, 12, ctx)
        linhas.append([rotulo, repr(texto_gerado[:58])])
    tabela(["modelo", "saída com greedy"], linhas)


if __name__ == "__main__":
    main()
