"""Executa os experimentos da Sprint 4 em sequência.

Inclui automaticamente os experimentos de arquitetura (exp13 a exp15) quando
eles existirem, além dos de geração (exp16 e exp17).

    python experiments/run_all_sprint4.py
"""

import importlib
import pathlib
import sys

PASTA = pathlib.Path(__file__).parent
PREFIXOS = ("exp13", "exp14", "exp15", "exp16", "exp17")


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
        print(f"[nota] {', '.join(ausentes)} (experimentos de arquitetura) ainda não estão "
              f"na pasta — rodando os {len(modulos)} experimentos disponíveis.")

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
    print(f"Todos os {len(modulos)} experimentos da Sprint 4 rodaram com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
