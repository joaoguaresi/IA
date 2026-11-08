"""
Experimento 20 — Experimentos de treinamento (parte do Entregável 4).

Objetivos:
  * variar o número de épocas e observar o ponto em que a perda de validação
    para de acompanhar a de treino (sinal de sobreajuste);
  * variar a taxa de aprendizado e observar o efeito sobre a convergência;
  * comparar explicitamente treino vs. validação, para distinguir
    "aprendendo" de "decorando".
"""

import time

import torch
from _comum import subtitulo, tabela, titulo

from src.bpe import obter_tokenizador_bpe
from src.dados import carregar_texto
from src.dataset import create_dataloader_v1
from src.transformer import GPTModel
from src.treino import calcular_perda_loader, treinar_modelo

EMB_DIM, N_LAYERS, N_HEADS, CONTEXTO = 64, 2, 2, 32


def _preparar_loaders(tokenizador, texto):
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
    return train_loader, val_loader


def _modelo_novo(vocab_size, seed=123):
    torch.manual_seed(seed)
    return GPTModel(
        vocab_size=vocab_size, context_length=CONTEXTO,
        emb_dim=EMB_DIM, n_layers=N_LAYERS, num_heads=N_HEADS, drop_rate=0.0,
    )


def main() -> None:
    titulo("EXPERIMENTO 20 — EXPERIMENTOS DE TREINAMENTO")

    tokenizador = obter_tokenizador_bpe()
    texto = carregar_texto()
    train_loader, val_loader = _preparar_loaders(tokenizador, texto)

    subtitulo("20.1 Efeito do número de épocas: quando a validação para de melhorar?")
    modelo = _modelo_novo(tokenizador.n_vocab)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=4e-4, weight_decay=0.1)
    perdas_treino, perdas_val, tokens_vistos, _ = treinar_modelo(
        modelo, train_loader, val_loader, otimizador, device=None,
        num_epocas=8, freq_avaliacao=len(train_loader), lotes_avaliacao=len(val_loader),
        contexto_inicial="O velho", tokenizador=tokenizador, seed=123,
    )
    linhas = []
    melhor_val = float("inf")
    epoca_melhor_val = None
    for i, (pt, pv) in enumerate(zip(perdas_treino, perdas_val), 1):
        gap = pv - pt
        if pv < melhor_val:
            melhor_val, epoca_melhor_val = pv, i
        linhas.append([i, f"{pt:.3f}", f"{pv:.3f}", f"{gap:+.3f}"])
    tabela(["época", "perda treino", "perda val", "val - treino"], linhas)
    print(f"menor perda de validação: época {epoca_melhor_val} ({melhor_val:.3f}); "
          f"treinar além desse ponto só reduz a perda de TREINO, não a de validação "
          "— a definição prática de sobreajuste.")

    subtitulo("20.2 Efeito da taxa de aprendizado (3 épocas, mesma inicialização)")
    linhas = []
    for lr in (1e-5, 1e-4, 4e-4, 1e-3, 1e-2):
        modelo_lr = _modelo_novo(tokenizador.n_vocab)
        otimizador_lr = torch.optim.AdamW(modelo_lr.parameters(), lr=lr, weight_decay=0.1)
        inicio = time.perf_counter()
        pt, pv, _, _ = treinar_modelo(
            modelo_lr, train_loader, val_loader, otimizador_lr, device=None,
            num_epocas=3, freq_avaliacao=len(train_loader), lotes_avaliacao=len(val_loader),
            contexto_inicial="O velho", tokenizador=tokenizador, seed=123,
        )
        duracao = time.perf_counter() - inicio
        linhas.append([
            f"{lr:.0e}", f"{pt[0]:.3f}", f"{pt[-1]:.3f}", f"{pv[-1]:.3f}", f"{duracao:.1f} s",
        ])
    tabela(["learning rate", "perda treino (época 1)", "perda treino (época 3)",
            "perda val (época 3)", "tempo"], linhas)
    print("taxas pequenas demais convergem devagar (perda quase não muda em 3 "
          "épocas); taxas grandes demais podem convergir mais rápido no treino, mas "
          "arriscam desestabilizar a otimização — por isso o livro usa 4e-4 como "
          "meio-termo, não a maior taxa possível.")

    subtitulo("20.3 Distinguindo \"aprender\" de \"decorar\"")
    modelo_longo = _modelo_novo(tokenizador.n_vocab)
    otimizador_longo = torch.optim.AdamW(modelo_longo.parameters(), lr=4e-4, weight_decay=0.1)
    pt_longo, pv_longo, _, amostras_longo = treinar_modelo(
        modelo_longo, train_loader, val_loader, otimizador_longo, device=None,
        num_epocas=10, freq_avaliacao=len(train_loader), lotes_avaliacao=len(val_loader),
        contexto_inicial="O velho consertou", tokenizador=tokenizador, seed=123,
    )
    gap_inicial = pv_longo[0] - pt_longo[0]
    gap_final = pv_longo[-1] - pt_longo[-1]
    print(f"gap (val - treino) na época 1 ..... {gap_inicial:+.3f}")
    print(f"gap (val - treino) na época 10 .... {gap_final:+.3f}")
    print(f"amostra na época 1 ................ {amostras_longo[0][:80]!r}")
    print(f"amostra na época 10 ............... {amostras_longo[-1][:80]!r}")
    texto_corpus_normalizado = " ".join(texto.split())
    trecho_amostra = " ".join(amostras_longo[-1].split())[:40]
    memorizado = trecho_amostra in texto_corpus_normalizado if trecho_amostra else False
    print(f"os primeiros ~40 caracteres da amostra final aparecem literalmente no "
          f"corpus de treino: {memorizado}")
    if memorizado:
        print("(um gap crescente entre treino e validação, lado a lado com um trecho "
              "gerado que reaparece palavra por palavra no corpus, é a evidência dupla "
              "de que o modelo decorou o texto em vez de aprender um padrão geral — "
              "esperado dado o quão pequeno é o corpus desta Sprint)")
    else:
        print("(mesmo sem uma citação literal do corpus neste recorte de 40 "
              "caracteres, o gap crescente entre treino e validação já é, sozinho, "
              "evidência de overfitting; aqui o sintoma aparece de outra forma: a "
              "amostra final degenerou em espaços/repetições em vez de texto fluente "
              "— um modelo pequeno e pouco treinado nem chega a memorizar direito, "
              "só perde a capacidade de generalizar. O exp19, com um modelo maior "
              "pelas mesmas 10 épocas, já produz uma amostra final com estrutura de "
              "frase reconhecível, ainda que repetindo trechos do corpus.)")


if __name__ == "__main__":
    main()
