"""
Experimento 22 — Estratégias de geração de texto (Entregável 5).

Compara, a partir do MESMO contexto inicial, a seleção gulosa com a amostragem
probabilística, variando temperatura e top-k. Todas as medidas são repetidas com
o modelo antes e depois do treino, para separar o que vem da estratégia de
decodificação do que vem dos pesos aprendidos.

Medidas usadas:
  * tokens distintos e taxa de repetição  -> diversidade
  * cobertura de bigramas do corpus       -> coerência local
  * perplexidade do texto gerado          -> confiança do modelo no que produziu
"""

import pathlib
from collections import Counter
from typing import Sequence

import torch
from _comum import salvar_figura, subtitulo, tabela, titulo
from _treino_demo import (
    CHECKPOINT_DEMO,
    CONFIG_DEMO,
    CONTEXTO_INICIAL,
    modelo_novo,
    treinar_demo,
)

from src.bpe import obter_tokenizador_bpe
from src.checkpoint import carregar_modelo, mesmos_pesos, salvar_checkpoint
from src.dados import carregar_texto
from src.geracao import gerar, ids_para_texto, taxa_de_repeticao, texto_para_ids
from src.treino import calcular_perda_lote, perplexidade

CORES = ("#1C5D86", "#B45309")
TEMPERATURAS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0)


def bigramas_do_corpus(token_ids: Sequence[int]) -> set:
    return set(zip(token_ids, list(token_ids)[1:]))


def cobertura_de_bigramas(gerados: Sequence[int], conhecidos: set) -> float:
    """Fração dos pares consecutivos gerados que existem no corpus."""
    pares = list(zip(gerados, list(gerados)[1:]))
    if not pares:
        return 0.0
    return sum(1 for p in pares if p in conhecidos) / len(pares)


@torch.no_grad()
def perplexidade_do_texto(modelo, ids: torch.Tensor, context_length: int) -> float:
    """Perplexidade que o próprio modelo atribui ao texto que produziu."""
    recorte = ids[:, : context_length + 1]
    if recorte.shape[1] < 2:
        return float("nan")
    entradas, alvos = recorte[:, :-1], recorte[:, 1:]
    return perplexidade(float(calcular_perda_lote(entradas, alvos, modelo)))


def medir(modelo, tokenizador, conhecidos, n_tokens=24, **cfg):
    """Gera um texto com a estratégia pedida e devolve as medidas."""
    idx = texto_para_ids(CONTEXTO_INICIAL, tokenizador)
    saida = gerar(modelo, idx, n_tokens, CONFIG_DEMO["context_length"], **cfg)
    gerados = saida[0, idx.shape[1]:].tolist()
    return {
        "texto": ids_para_texto(saida, tokenizador),
        "distintos": len(set(gerados)),
        "repeticao": taxa_de_repeticao(gerados),
        "cobertura": cobertura_de_bigramas(gerados, conhecidos),
        "perplexidade": perplexidade_do_texto(modelo, saida, CONFIG_DEMO["context_length"]),
    }


def main() -> None:
    titulo("EXPERIMENTO 22 — ESTRATÉGIAS DE GERAÇÃO APÓS O TREINO")

    tokenizador = obter_tokenizador_bpe()
    texto = carregar_texto()
    conhecidos = bigramas_do_corpus(tokenizador.encode(texto))

    subtitulo("22.1 Modelos comparados")
    nao_treinado = modelo_novo(seed=123)
    nao_treinado.eval()
    if pathlib.Path(CHECKPOINT_DEMO).exists():
        treinado, conteudo = carregar_modelo(CHECKPOINT_DEMO)
        origem = f"checkpoint {pathlib.Path(CHECKPOINT_DEMO).name}"
        metadados = conteudo.get("metadados", {})
    else:
        treinado, otimizador, historico = treinar_demo(verbose=False)
        treinado.eval()
        salvar_checkpoint(CHECKPOINT_DEMO, treinado, otimizador=otimizador)
        origem = "treinado agora (exp21 ainda não havia sido executado)"
        metadados = {}
    print(f"contexto inicial ......... {CONTEXTO_INICIAL!r}")
    print(f"modelo treinado .......... {origem}")
    for chave in ("perda_treino", "perda_validacao", "epocas"):
        if chave in metadados:
            print(f"  {chave} = {metadados[chave]}")
    print(f"bigramas distintos no corpus: {len(conhecidos):,}")
    # verificação: sem pesos diferentes, toda a comparação desta seção seria vazia
    print(f"os dois modelos têm pesos diferentes: "
          f"{not mesmos_pesos(nao_treinado, treinado)}")

    subtitulo("22.2 Mesma estratégia, modelos diferentes (o efeito do treino)")
    linhas = []
    for rotulo_modelo, m in (("não treinado", nao_treinado), ("treinado", treinado)):
        for rotulo_estrategia, cfg in (
            ("greedy", dict(temperatura=0.0)),
            ("amostragem T=1.0", dict(temperatura=1.0, seed=42)),
        ):
            r = medir(m, tokenizador, conhecidos, **cfg)
            linhas.append([
                rotulo_modelo, rotulo_estrategia, r["distintos"],
                f"{r['repeticao']:.2f}", f"{r['cobertura']:.2f}",
                f"{r['perplexidade']:9.1f}",
            ])
    tabela(["modelo", "estratégia", "tokens distintos", "repetição",
            "cobertura de bigramas", "perplexidade"], linhas)

    subtitulo("22.3 Textos produzidos")
    for rotulo_modelo, m in (("não treinado", nao_treinado), ("treinado", treinado)):
        for rotulo_estrategia, cfg in (
            ("greedy", dict(temperatura=0.0)),
            ("amostragem T=1.0", dict(temperatura=1.0, seed=42)),
        ):
            r = medir(m, tokenizador, conhecidos, **cfg)
            print(f"[{rotulo_modelo} · {rotulo_estrategia}] {r['texto'][:76]!r}")

    subtitulo("22.4 Temperatura no modelo treinado")
    linhas = []
    for t in TEMPERATURAS:
        r = medir(treinado, tokenizador, conhecidos, temperatura=t, seed=42)
        linhas.append([f"{t:.1f}", r["distintos"], f"{r['repeticao']:.2f}",
                       f"{r['cobertura']:.2f}", f"{r['perplexidade']:9.1f}",
                       repr(r["texto"][:40])])
    r0 = medir(treinado, tokenizador, conhecidos, temperatura=0.0)
    linhas.insert(0, ["0 (greedy)", r0["distintos"], f"{r0['repeticao']:.2f}",
                      f"{r0['cobertura']:.2f}", f"{r0['perplexidade']:9.1f}",
                      repr(r0["texto"][:40])])
    tabela(["temperatura", "distintos", "repetição", "cobertura", "perplexidade",
            "texto"], linhas)

    subtitulo("22.5 Top-k no modelo treinado (T = 1.0)")
    linhas = []
    for k in (1, 5, 20, 50, 200):
        r = medir(treinado, tokenizador, conhecidos, temperatura=1.0, top_k=k, seed=42)
        linhas.append([k, r["distintos"], f"{r['repeticao']:.2f}",
                       f"{r['cobertura']:.2f}", f"{r['perplexidade']:9.1f}",
                       repr(r["texto"][:40])])
    tabela(["top_k", "distintos", "repetição", "cobertura", "perplexidade",
            "texto"], linhas)

    subtitulo("22.6 Temperatura e top-k combinados")
    linhas = []
    for t, k in ((0.7, None), (0.7, 20), (1.0, 20), (1.5, 20), (1.5, None)):
        r = medir(treinado, tokenizador, conhecidos, temperatura=t, top_k=k, seed=42)
        linhas.append([f"{t:.1f}", k if k else "—", r["distintos"],
                       f"{r['repeticao']:.2f}", f"{r['cobertura']:.2f}",
                       f"{r['perplexidade']:9.1f}"])
    tabela(["temperatura", "top_k", "distintos", "repetição", "cobertura",
            "perplexidade"], linhas)

    subtitulo("22.7 Variação entre execuções com a mesma estratégia")
    for seed in (1, 2, 3):
        r = medir(treinado, tokenizador, conhecidos, temperatura=1.0, top_k=20, seed=seed)
        print(f"seed={seed}: {r['texto'][:66]!r}")
    r_a = medir(treinado, tokenizador, conhecidos, temperatura=0.0)
    r_b = medir(treinado, tokenizador, conhecidos, temperatura=0.0)
    print(f"greedy repetido duas vezes produz o mesmo texto: {r_a['texto'] == r_b['texto']}")

    subtitulo("22.8 Figura")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for (rotulo, m), cor in zip((("não treinado", nao_treinado),
                                     ("treinado", treinado)), CORES):
            distintos, cobertura = [], []
            for t in TEMPERATURAS:
                r = medir(m, tokenizador, conhecidos, temperatura=t, seed=42)
                distintos.append(r["distintos"])
                cobertura.append(r["cobertura"])
            ax1.plot(TEMPERATURAS, distintos, "o-", color=cor, linewidth=2, label=rotulo)
            ax2.plot(TEMPERATURAS, cobertura, "o-", color=cor, linewidth=2, label=rotulo)
        ax1.set_xlabel("temperatura"); ax1.set_ylabel("tokens distintos gerados")
        ax1.set_title("Diversidade")
        ax2.set_xlabel("temperatura"); ax2.set_ylabel("fração de bigramas do corpus")
        ax2.set_title("Coerência local")
        for ax in (ax1, ax2):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, linewidth=0.7)
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle("Estratégia de decodificação × treino dos pesos", fontsize=12.5)
        salvar_figura(fig, "exp22_estrategias_treino.png")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib não instalado — figura não gerada.")


if __name__ == "__main__":
    main()
