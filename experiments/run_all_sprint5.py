"""Executa os experimentos da Sprint 5 em sequência.

Detecta dinamicamente quais experimentos já existem (mesmo padrão do
run_all_sprint4.py), então roda mesmo que só uma parte da dupla já tenha
enviado o código: exp18-exp20 (avaliação, treinamento e experimentos de
treinamento) e exp21-exp22 (persistência de pesos e geração pós-treino).

    python experiments/run_all_sprint5.py
"""

import importlib
import pathlib
import sys

PASTA = pathlib.Path(__file__).parent
PREFIXOS = ("exp18", "exp19", "exp20", "exp21", "exp22")


def modulos_existentes():
    nomes = []
    for prefixo in PREFIXOS:
        achados = sorted(PASTA.glob(f"{prefixo}_*.py"))
        nomes.extend(a.stem for a in achados)
    return nomes


def main() -> int:
    import _comum  # noqa: F401  (ajusta o sys.path)

    modulos = modulos_existentes()
    ausentes = [p for p in PREFIXOS if not any(n.startswith(p) for n in modulos)]
    if ausentes:
        print(f"[nota] {', '.join(ausentes)} ainda não estão na pasta — "
              f"rodando os {len(modulos)} experimentos disponíveis.")

    falhas = []
    for nome in modulos:
        try:
            importlib.import_module(nome).main()
        except Exception as erro:  # pragma: no cover
            falhas.append((nome, erro))
            print(f"\n[ERRO] {nome}: {erro}")

    print("\n" + "=" * 78)
    if falhas:
        print(f"{len(falhas)} experimento(s) falharam: {[f[0] for f in falhas]}")
        return 1
    print(f"Todos os {len(modulos)} experimentos da Sprint 5 rodaram com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
