"""
Experimento 21 — Salvamento e carregamento dos pesos (Entregável 6).

Treina um GPT reduzido, grava o checkpoint, recarrega em uma instância nova e
verifica que o modelo recuperado é idêntico ao salvo — nos pesos e no texto que
produz. Também demonstra a retomada do treino a partir do estado do otimizador.
"""

import time

import torch
from _comum import subtitulo, tabela, titulo
from _treino_demo import (
    CHECKPOINT_DEMO,
    CONFIG_DEMO,
    CONTEXTO_INICIAL,
    TREINO,
    modelo_novo,
    treinar_demo,
)

from src.checkpoint import (
    carregar_checkpoint,
    carregar_modelo,
    config_do_modelo,
    mesmos_pesos,
    numero_de_parametros,
    resumir_checkpoint,
    restaurar_otimizador,
    salvar_checkpoint,
    tamanho_do_arquivo_mb,
)
from src.geracao import gerar_texto
from src.treino import avaliar_modelo


def main() -> None:
    titulo("EXPERIMENTO 21 — SALVAMENTO E CARREGAMENTO DOS PESOS")

    subtitulo("21.1 Configuração")
    tabela(
        ["parâmetro", "valor"],
        [[k, v] for k, v in CONFIG_DEMO.items()]
        + [[k, v] for k, v in TREINO.items()],
    )

    subtitulo("21.2 Treino")
    inicio = time.perf_counter()
    modelo, otimizador, historico = treinar_demo(verbose=False)
    segundos = time.perf_counter() - inicio
    tokenizador = historico["tokenizador"]
    perda_treino, perda_val = avaliar_modelo(
        modelo, historico["train_loader"], historico["val_loader"], None, 2
    )
    print(f"parâmetros do modelo ..... {numero_de_parametros(modelo):,}")
    print(f"tempo de treino .......... {segundos:.1f} s")
    print(f"perda final (treino) ..... {perda_treino:.4f}")
    print(f"perda final (validação) .. {perda_val:.4f}")

    subtitulo("21.3 Salvamento")
    caminho = salvar_checkpoint(
        CHECKPOINT_DEMO, modelo, otimizador=otimizador,
        metadados={
            "perda_treino": round(perda_treino, 4),
            "perda_validacao": round(perda_val, 4),
            "epocas": TREINO["num_epocas"],
            "taxa_aprendizado": TREINO["taxa_aprendizado"],
            "tokenizador": tokenizador.nome,
            "contexto_inicial": CONTEXTO_INICIAL,
        },
    )
    resumo = resumir_checkpoint(caminho)
    print(f"arquivo .................. {caminho}")
    parametros = numero_de_parametros(modelo)
    mascaras = CONFIG_DEMO["n_layers"] * CONFIG_DEMO["context_length"] ** 2
    tabela(
        ["campo", "valor"],
        [["tamanho", f"{resumo['tamanho_mb']:.2f} MB"],
         ["versão do formato", resumo["versao_formato"]],
         ["tensores no state_dict", resumo["n_tensores"]],
         ["valores guardados", f"{resumo['n_valores']:,}"],
         ["  dos quais parâmetros", f"{parametros:,}"],
         ["  dos quais buffers (máscaras causais)", f"{resumo['n_valores'] - parametros:,}"],
         ["máscaras esperadas (n_layers × contexto²)", f"{mascaras:,}"],
         ["estado do otimizador", "sim" if resumo["tem_otimizador"] else "não"]],
    )
    print("configuração gravada:")
    for chave, valor in resumo["config"].items():
        print(f"  {chave} = {valor}")
    print("metadados:")
    for chave, valor in resumo["metadados"].items():
        print(f"  {chave} = {valor}")

    subtitulo("21.4 Carregamento em uma instância nova")
    carregado, conteudo = carregar_modelo(caminho)
    print(f"arquitetura reconstruída a partir do arquivo: {type(carregado).__name__}")
    print(f"config lida do checkpoint confere com a do modelo original: "
          f"{config_do_modelo(modelo) == conteudo['config']}")
    print(f"pesos idênticos aos salvos: {mesmos_pesos(modelo, carregado)}")
    print(f"parâmetros: original {numero_de_parametros(modelo):,} | "
          f"carregado {numero_de_parametros(carregado):,}")

    subtitulo("21.5 O modelo carregado gera o mesmo texto")
    modelo.eval()
    linhas = []
    for rotulo, m in (("original", modelo), ("carregado", carregado)):
        texto = gerar_texto(m, tokenizador, CONTEXTO_INICIAL, 20,
                            CONFIG_DEMO["context_length"])
        linhas.append([rotulo, repr(texto[:62])])
    tabela(["modelo", "geração greedy"], linhas)
    print(f"as duas saídas são idênticas: {linhas[0][1] == linhas[1][1]}")

    subtitulo("21.6 Pesos aleatórios vs. pesos carregados")
    aleatorio = modelo_novo(seed=999)
    linhas = []
    for rotulo, m in (("pesos aleatórios", aleatorio), ("pesos treinados", carregado)):
        texto = gerar_texto(m, tokenizador, CONTEXTO_INICIAL, 20,
                            CONFIG_DEMO["context_length"])
        linhas.append([rotulo, repr(texto[:62])])
    tabela(["modelo", "geração greedy"], linhas)
    print(f"mesmos pesos? {mesmos_pesos(aleatorio, carregado)}")

    subtitulo("21.7 Retomada do treino")
    novo_otimizador = torch.optim.AdamW(
        carregado.parameters(), lr=TREINO["taxa_aprendizado"],
        weight_decay=TREINO["weight_decay"],
    )
    estado_antes = len(novo_otimizador.state_dict().get("state", {}))
    restaurar_otimizador(novo_otimizador, conteudo)
    estado_depois = len(novo_otimizador.state_dict().get("state", {}))
    print(f"tensores com estado no otimizador: antes {estado_antes}, "
          f"depois de restaurar {estado_depois}")

    subtitulo("21.8 Tamanho: só os pesos vs. pesos + otimizador")
    so_pesos = CHECKPOINT_DEMO.with_name("gpt_demo_so_pesos.pt")
    salvar_checkpoint(so_pesos, modelo)
    tabela(
        ["conteúdo", "tamanho", "serve para"],
        [["pesos + config", f"{tamanho_do_arquivo_mb(so_pesos):.2f} MB", "inferência"],
         ["pesos + config + otimizador", f"{tamanho_do_arquivo_mb(caminho):.2f} MB",
          "inferência e retomada do treino"]],
    )

    subtitulo("21.9 Leitura do checkpoint sem reconstruir o modelo")
    conteudo_cru = carregar_checkpoint(caminho)
    print(f"chaves do arquivo: {sorted(conteudo_cru.keys())}")
    primeiros = list(conteudo_cru["pesos_modelo"].items())[:5]
    tabela(
        ["tensor", "forma", "valores"],
        [[nome, str(tuple(t.shape)), f"{int(t.numel()):,}"] for nome, t in primeiros],
    )


if __name__ == "__main__":
    main()
