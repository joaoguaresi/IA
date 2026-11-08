"""
Experimento 19 — Loop de treinamento (Entregável 3).

Objetivos:
  * treinar de fato um GPTModel pequeno sobre o corpus do projeto, usando o
    loop de `treinar_modelo` (zero_grad -> perda -> backward -> step);
  * acompanhar a perda de treino/validação a cada avaliação e uma amostra de
    texto gerada ao final de cada época;
  * registrar o número de parâmetros, o tempo de treino e salvar o gráfico
    de perdas (Figura 5.12 do livro).
"""

import time

import torch
from _comum import salvar_figura, subtitulo, tabela, titulo

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto, nome_do_corpus
from src.dataset import create_dataloader_v1
from src.transformer import GPTModel, numero_de_parametros
from src.treino import calcular_perda_loader, perplexidade, plotar_perdas, treinar_modelo

EMB_DIM, N_LAYERS, N_HEADS, CONTEXTO = 64, 4, 4, 32
NUM_EPOCAS, TAXA_APRENDIZADO, DECAIMENTO = 10, 4e-4, 0.1
CONTEXTO_INICIAL = "O velho consertou"


def main() -> None:
    titulo("EXPERIMENTO 19 — LOOP DE TREINAMENTO")
    torch.manual_seed(123)

    subtitulo("19.1 Dados: divisão treino/validação (90/10)")
    tokenizador = obter_tokenizador_bpe()
    texto = carregar_texto()
    split_idx = int(0.90 * len(texto))
    texto_treino, texto_val = texto[:split_idx], texto[split_idx:]

    train_loader = create_dataloader_v1(
        texto_treino, tokenizer=tokenizador, batch_size=2,
        max_length=CONTEXTO, stride=CONTEXTO, drop_last=True, shuffle=True,
    )
    val_loader = create_dataloader_v1(
        texto_val, tokenizer=tokenizador, batch_size=2,
        max_length=CONTEXTO, stride=CONTEXTO, drop_last=False, shuffle=False,
    )
    print(f"corpus ................. {nome_do_corpus()} ({len(texto):,} caracteres)")
    print(f"tokens (treino+val) .... {len(tokenizador.encode(texto)):,}")
    print(f"lotes de treino ........ {len(train_loader)}")
    print(f"lotes de validação ..... {len(val_loader)}")

    subtitulo("19.2 O modelo")
    modelo = GPTModel(
        vocab_size=tokenizador.n_vocab, context_length=CONTEXTO,
        emb_dim=EMB_DIM, n_layers=N_LAYERS, num_heads=N_HEADS, drop_rate=0.0,
    )
    print(f"configuração ............ emb_dim={EMB_DIM}, n_layers={N_LAYERS}, "
          f"num_heads={N_HEADS}, context_length={CONTEXTO}")
    print(f"parâmetros .............. {numero_de_parametros(modelo):,}")

    perda_inicial_treino = calcular_perda_loader(train_loader, modelo)
    perda_inicial_val = calcular_perda_loader(val_loader, modelo)
    print(f"perda inicial (treino) .. {perda_inicial_treino:.3f}  "
          f"(perplexity {perplexidade(perda_inicial_treino):,.0f})")
    print(f"perda inicial (val) ..... {perda_inicial_val:.3f}  "
          f"(perplexity {perplexidade(perda_inicial_val):,.0f})")

    subtitulo("19.3 Treinamento")
    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=TAXA_APRENDIZADO, weight_decay=DECAIMENTO
    )
    inicio = time.perf_counter()
    perdas_treino, perdas_val, tokens_vistos, amostras = treinar_modelo(
        modelo, train_loader, val_loader, otimizador, device=None,
        num_epocas=NUM_EPOCAS, freq_avaliacao=5, lotes_avaliacao=5,
        contexto_inicial=CONTEXTO_INICIAL, tokenizador=tokenizador, seed=123,
    )
    duracao = time.perf_counter() - inicio
    print(f"tempo total ............. {duracao:.1f} s  "
          f"({duracao / NUM_EPOCAS:.1f} s/época)")
    print(f"pontos de avaliação ..... {len(perdas_treino)}")

    tabela(
        ["avaliação", "tokens vistos", "perda treino", "perda val"],
        [
            [i + 1, f"{t:,}", f"{pt:.3f}", f"{pv:.3f}"]
            for i, (t, pt, pv) in enumerate(zip(tokens_vistos, perdas_treino, perdas_val))
        ],
    )

    subtitulo("19.4 Amostras de texto geradas ao final de cada época")
    for i, amostra in enumerate(amostras, 1):
        print(f"época {i:>2}: {amostra[:90]!r}")

    subtitulo("19.5 Antes e depois")
    tabela(
        ["", "perda treino", "perda val", "perplexity treino", "perplexity val"],
        [
            ["antes do treino", f"{perda_inicial_treino:.3f}", f"{perda_inicial_val:.3f}",
             f"{perplexidade(perda_inicial_treino):,.0f}", f"{perplexidade(perda_inicial_val):,.0f}"],
            ["depois do treino", f"{perdas_treino[-1]:.3f}", f"{perdas_val[-1]:.3f}",
             f"{perplexidade(perdas_treino[-1]):,.0f}", f"{perplexidade(perdas_val[-1]):,.0f}"],
        ],
    )

    subtitulo("19.6 Figura")
    try:
        import matplotlib
        matplotlib.use("Agg")

        epocas_vistas = torch.linspace(0, NUM_EPOCAS, len(perdas_treino)).tolist()
        fig = plotar_perdas(epocas_vistas, tokens_vistos, perdas_treino, perdas_val)
        salvar_figura(fig, "exp19_perdas_treino_val.png")
    except ImportError:
        print("[aviso] matplotlib não instalado — gráfico não gerado.")


if __name__ == "__main__":
    main()
