"""
Experimento 14 — Modelo GPT completo (Entregável 3).

Objetivos:
  * acompanhar as dimensões dos tensores do início ao fim do modelo;
  * decompor o total de parâmetros por componente estrutural;
  * validar uma fórmula fechada de contagem de parâmetros contra o modelo
    instanciado, e então usá-la (sem instanciar nada grande) para projetar o
    custo das quatro configurações oficiais do GPT-2;
  * mostrar o efeito do *weight tying* sobre o tamanho do GPT-2 small — a
    diferença entre os famosos "163M" e "124M" parâmetros.
"""

import torch
from _comum import subtitulo, tabela, titulo

from src.transformer import GPTModel, memoria_em_mb, parametros_por_componente

VOCAB, CONTEXT, EMB_DIM, N_LAYERS, N_HEADS = 50, 6, 16, 2, 2


def contar_parametros_teorico(vocab_size: int, context_length: int, emb_dim: int, n_layers: int) -> int:
    """
    Fórmula fechada para o total de parâmetros de um GPTModel, sem instanciar
    nada — útil para projetar modelos grandes demais para montar na memória
    só para contar pesos.

    Por componente (ver `parametros_por_componente` em src/transformer.py):
        token_embeddings  = vocab_size × emb_dim
        positional_emb.   = context_length × emb_dim
        cada bloco        = atenção (4·emb_dim² + emb_dim, com viés só no out_proj)
                           + feed-forward (8·emb_dim² + 5·emb_dim)
                           + 2 layernorms (4·emb_dim)
                           = 12·emb_dim² + 10·emb_dim
        layernorm final   = 2·emb_dim
        out_head          = emb_dim × vocab_size  (sem viés)
    """
    token_emb = vocab_size * emb_dim
    pos_emb = context_length * emb_dim
    por_bloco = 12 * emb_dim**2 + 10 * emb_dim
    norm_final = 2 * emb_dim
    out_head = emb_dim * vocab_size
    return token_emb + pos_emb + n_layers * por_bloco + norm_final + out_head


@torch.no_grad()
def main() -> None:
    titulo("EXPERIMENTO 14 — MODELO GPT COMPLETO")
    torch.manual_seed(123)

    subtitulo("14.1 Do Token ID aos logits, passo a passo")
    modelo = GPTModel(
        vocab_size=VOCAB, context_length=CONTEXT, emb_dim=EMB_DIM,
        n_layers=N_LAYERS, num_heads=N_HEADS, drop_rate=0.0,
    )
    modelo.eval()
    idx = torch.randint(0, VOCAB, (2, CONTEXT))
    x = modelo.embeddings(idx)
    print(f"Token IDs ................ {tuple(idx.shape)}  (batch, tokens)")
    print(f"após embeddings .......... {tuple(x.shape)}  (batch, tokens, emb_dim)")
    for i, bloco in enumerate(modelo.blocos):
        x = bloco(x)
        print(f"após bloco {i + 1}/{N_LAYERS} ........... {tuple(x.shape)}  (forma preservada)")
    x = modelo.norm_final(x)
    logits = modelo.out_head(x)
    print(f"após LayerNorm final ..... {tuple(x.shape)}")
    print(f"logits ................... {tuple(logits.shape)}  (batch, tokens, vocab_size)")
    print(f"forward() do modelo bate com o passo a passo: "
          f"{bool(torch.allclose(modelo(idx), logits, atol=1e-6))}")

    subtitulo("14.2 Parâmetros por componente")
    partes = parametros_por_componente(modelo)
    tabela(
        ["componente", "parâmetros", "% do total"],
        [
            [nome, f"{valor:,}", f"{100 * valor / partes['total']:.1f}%"]
            for nome, valor in partes.items()
            if nome != "total"
        ]
        + [["TOTAL", f"{partes['total']:,}", "100.0%"]],
    )
    print(f"memória (float32) ........ {memoria_em_mb(modelo):.3f} MB")

    subtitulo("14.3 Validando a fórmula fechada contra o modelo real")
    teorico = contar_parametros_teorico(VOCAB, CONTEXT, EMB_DIM, N_LAYERS)
    print(f"contagem real (modelo instanciado) .... {partes['total']:,}")
    print(f"contagem pela fórmula fechada ......... {teorico:,}")
    print(f"batem exatamente: {teorico == partes['total']}")

    subtitulo("14.4 Projetando as quatro configurações oficiais do GPT-2")
    # emb_dim, n_layers, n_heads — de Radford et al. (2019); vocab e contexto
    # são os mesmos nas quatro variantes
    configs = {
        "GPT-2 small  (\"124M\")": (768, 12, 12),
        "GPT-2 medium (\"355M\")": (1024, 24, 16),
        "GPT-2 large  (\"774M\")": (1280, 36, 20),
        "GPT-2 XL     (\"1558M\")": (1600, 48, 25),
    }
    vocab_gpt2, ctx_gpt2 = 50257, 1024
    linhas = []
    for nome, (emb, layers, heads) in configs.items():
        total = contar_parametros_teorico(vocab_gpt2, ctx_gpt2, emb, layers)
        mem_mb = total * 4 / (1024**2)
        linhas.append([
            nome, emb, layers, heads, f"{total:,}", f"{mem_mb / 1024:.2f} GB",
        ])
    tabela(["configuração", "emb_dim", "n_layers", "n_heads", "parâmetros (sem tying)",
            "memória (float32)"], linhas)
    print("(nenhum desses modelos foi instanciado — só a fórmula validada acima "
          "foi aplicada; o GPT-2 XL teria ~6,1 GB só nos pesos em float32)")

    subtitulo("14.5 Weight tying: de onde vêm os \"124M\" do GPT-2 small")
    emb, layers, _ = configs["GPT-2 small  (\"124M\")"]
    sem_tying = contar_parametros_teorico(vocab_gpt2, ctx_gpt2, emb, layers)
    duplicado = vocab_gpt2 * emb  # tok_emb e out_head têm exatamente essa forma
    com_tying = sem_tying - duplicado
    print(f"parâmetros sem weight tying ........... {sem_tying:,}")
    print(f"tamanho de tok_emb (== out_head) ...... {duplicado:,}")
    print(f"parâmetros COM weight tying ............ {com_tying:,}")
    print("O GPT-2 original reaproveita a MESMA matriz para tok_emb e out_head "
          "(weight tying): como as duas têm forma (vocab_size, emb_dim), isso "
          "economiza vocab_size × emb_dim parâmetros — a diferença entre os "
          "163M \"de verdade\" desta implementação (sem tying, como no livro) e "
          "os 124M anunciados originalmente para o GPT-2 small.")


if __name__ == "__main__":
    main()
