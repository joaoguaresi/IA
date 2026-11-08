"""
Experimento 18 — Avaliação do modelo: cross-entropy loss e perplexity (Entregável 2).

Objetivos:
  * reproduzir, passo a passo, o caminho logits -> probabilidades -> perda
    descrito na seção 5.1 do livro, e confirmar que bate com
    `torch.nn.functional.cross_entropy`;
  * calcular a perplexity e interpretar o que ela significa;
  * aplicar `calcular_perda_loader` aos conjuntos de treino e validação do
    corpus do projeto, com o modelo ainda não treinado (ponto de partida).
"""

import torch
from _comum import subtitulo, tabela, titulo

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto, nome_do_corpus
from src.dataset import create_dataloader_v1
from src.transformer import GPTModel
from src.treino import calcular_perda_loader, perplexidade

EMB_DIM, N_LAYERS, N_HEADS, CONTEXTO = 32, 2, 2, 8


# os experimentos apenas medem: nenhum gradiente é necessário
@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 18 — AVALIAÇÃO: CROSS-ENTROPY LOSS E PERPLEXITY")
    torch.manual_seed(123)

    tokenizador = obter_tokenizador_bpe()
    modelo = GPTModel(
        vocab_size=tokenizador.n_vocab, context_length=CONTEXTO,
        emb_dim=EMB_DIM, n_layers=N_LAYERS, num_heads=N_HEADS, drop_rate=0.0,
    )
    modelo.eval()

    subtitulo("18.1 Dois exemplos de entrada e seus alvos")
    textos = ["o velho consertou", "a chuva caiu"]
    entradas = torch.stack([torch.tensor(tokenizador.encode(t)[:3]) for t in textos])
    # alvo = entrada deslocada uma posição (mesma convenção da Sprint 2); aqui,
    # para ilustrar, usamos o token seguinte de cada frase como 4º alvo
    alvos = torch.stack([
        torch.tensor(tokenizador.encode(t)[1:4]) for t in textos
    ])
    for t, e, a in zip(textos, entradas, alvos):
        print(f"{t!r:<22} entrada={e.tolist()}  alvo={a.tolist()}")

    subtitulo("18.2 De logits a probabilidades (softmax)")
    logits = modelo(entradas)
    probas = torch.softmax(logits, dim=-1)
    print(f"logits ................. {tuple(logits.shape)}  (batch, tokens, vocab)")
    print(f"probabilidades .......... {tuple(probas.shape)}")
    print(f"soma de uma linha de probas: {float(probas[0, 0].sum()):.6f}")

    subtitulo("18.3 Probabilidade atribuída ao token-alvo, em cada posição")
    linhas = []
    for i, texto in enumerate(textos):
        prob_alvo = probas[i, torch.arange(3), alvos[i]]
        linhas.append([texto, str([f"{p:.2e}" for p in prob_alvo.tolist()])])
    tabela(["texto", "probabilidade do token-alvo em cada posição"], linhas)
    print("(modelo ainda não treinado: as probabilidades do token certo ficam "
          f"perto de 1/vocab_size ≈ {1 / tokenizador.n_vocab:.2e}, ou seja, "
          "próximas do acaso)")

    subtitulo("18.4 Log -> média -> sinal trocado = cross-entropy, manualmente")
    todas_probs_alvo = torch.cat([
        probas[i, torch.arange(3), alvos[i]] for i in range(len(textos))
    ])
    log_probs = torch.log(todas_probs_alvo)
    media_log_probs = log_probs.mean()
    perda_manual = -media_log_probs
    print(f"probabilidades do alvo ... {[f'{p:.3e}' for p in todas_probs_alvo.tolist()]}")
    print(f"log das probabilidades ... {[f'{p:.4f}' for p in log_probs.tolist()]}")
    print(f"média dos logs ........... {float(media_log_probs):.4f}")
    print(f"perda (-média) ........... {float(perda_manual):.4f}")

    perda_pytorch = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), alvos.flatten()
    )
    print(f"cross_entropy do PyTorch . {float(perda_pytorch):.4f}")
    print(f"as duas batem: {bool(torch.allclose(perda_manual, perda_pytorch, atol=1e-4))}")

    subtitulo("18.5 Perplexity")
    ppl = perplexidade(float(perda_pytorch))
    print(f"perda ............... {float(perda_pytorch):.4f}")
    print(f"perplexity .......... {ppl:,.1f}")
    print(f"vocabulário completo  {tokenizador.n_vocab:,}")
    if ppl <= tokenizador.n_vocab:
        print(f"(o modelo está tão incerto quanto se escolhesse ao acaso entre "
              f"{ppl:,.0f} dos {tokenizador.n_vocab:,} tokens do vocabulário)")
    else:
        print(f"(a perplexity ({ppl:,.0f}) chega a superar o próprio vocabulário "
              f"({tokenizador.n_vocab:,}) — com pesos aleatórios, a distribuição do "
              "modelo não é exatamente uniforme, e pode, por puro acaso de "
              "inicialização, atribuir ao token certo uma probabilidade ainda menor "
              "do que a de um sorteio uniforme entre todos os tokens)")

    subtitulo("18.6 Perda sobre o corpus inteiro: treino vs. validação")
    texto_completo = carregar_texto()
    split_idx = int(0.90 * len(texto_completo))
    texto_treino, texto_val = texto_completo[:split_idx], texto_completo[split_idx:]

    train_loader = create_dataloader_v1(
        texto_treino, tokenizer=tokenizador, batch_size=2,
        max_length=CONTEXTO, stride=CONTEXTO, drop_last=True, shuffle=True,
    )
    val_loader = create_dataloader_v1(
        texto_val, tokenizer=tokenizador, batch_size=2,
        max_length=CONTEXTO, stride=CONTEXTO, drop_last=False, shuffle=False,
    )
    perda_treino = calcular_perda_loader(train_loader, modelo)
    perda_val = calcular_perda_loader(val_loader, modelo)
    print(f"corpus .................. {nome_do_corpus()} ({len(texto_completo):,} caracteres)")
    print(f"lotes de treino ......... {len(train_loader)}")
    print(f"lotes de validação ...... {len(val_loader)}")
    tabela(
        ["conjunto", "perda (cross-entropy)", "perplexity"],
        [
            ["treino", f"{perda_treino:.4f}", f"{perplexidade(perda_treino):,.0f}"],
            ["validação", f"{perda_val:.4f}", f"{perplexidade(perda_val):,.0f}"],
        ],
    )
    print("sem nenhum treino, as duas perdas ficam próximas e altas — é exatamente "
          "esse par de números que o exp19 vai observar diminuir a cada época.")


if __name__ == "__main__":
    main()
