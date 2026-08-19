"""Sanity check do ambiente computacional (Sprint 0-1).

Confirma que Python, PyTorch e as demais bibliotecas usadas no projeto
estão instaladas e funcionando corretamente.
"""

import sys


def main() -> None:
    print(f"Python: {sys.version}")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    x = torch.rand(3, 3)
    y = torch.rand(3, 3)
    print(f"Multiplicação de tensores 3x3 OK, shape resultante: {(x @ y).shape}")

    import numpy
    print(f"NumPy: {numpy.__version__}")

    import matplotlib
    print(f"Matplotlib: {matplotlib.__version__}")

    import tiktoken
    print(f"tiktoken: {tiktoken.__version__}")

    print("\nAmbiente configurado corretamente.")


if __name__ == "__main__":
    main()
