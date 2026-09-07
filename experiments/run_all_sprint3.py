"""Executa todos os experimentos da Sprint 3 em sequência.

    python experiments/run_all_sprint3.py
"""

import importlib
import sys

MODULOS = [
    "exp07_self_attention",
    "exp08_scaled_dot_product",
    "exp09_causal_attention",
    "exp10_multihead",
    "exp11_visualizacoes",
    "exp12_integracao_sprint2",
]


def main() -> int:
    import _comum  # noqa: F401  (ajusta o sys.path)

    falhas = []
    for nome in MODULOS:
        try:
            importlib.import_module(nome).main()
        except Exception as erro:  # pragma: no cover
            falhas.append((nome, erro))
            print(f"\n[ERRO] {nome}: {erro}")
    print("\n" + "=" * 78)
    if falhas:
        print(f"{len(falhas)} experimento(s) falharam: {[f[0] for f in falhas]}")
        return 1
    print(f"Todos os {len(MODULOS)} experimentos da Sprint 3 rodaram com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
