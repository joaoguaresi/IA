"""Executa todos os experimentos da Sprint 2 em sequência.

    python experiments/run_all.py
"""

import importlib
import sys

MODULOS = [
    "exp01_tokenizacao",
    "exp02_vocabulario_token_ids",
    "exp03_janela_deslizante",
    "exp04_dataloader",
    "exp05_embeddings",
    "exp06_pipeline_completo",
]


def main() -> int:
    import _comum  # noqa: F401  (ajusta o sys.path)

    falhas = []
    for nome in MODULOS:
        try:
            modulo = importlib.import_module(nome)
            modulo.main()
        except Exception as erro:  # pragma: no cover
            falhas.append((nome, erro))
            print(f"\n[ERRO] {nome}: {erro}")
    print("\n" + "=" * 78)
    if falhas:
        print(f"{len(falhas)} experimento(s) falharam: {[f[0] for f in falhas]}")
        return 1
    print(f"Todos os {len(MODULOS)} experimentos foram executados com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
