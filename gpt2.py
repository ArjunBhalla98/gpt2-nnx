from typing import List
from dataclasses import dataclass

import jax
from flax import nnx
import jax.numpy as jnp


@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int  # Number of attention heads
    embed_dim: int  # Embedding dimensions
    block_size: int  # maximum input sequence length, 1024
    vocab_size: int = 50273


class SelfAttention(nnx.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, rngs: nnx.Rngs):
        super().__init__()
        pass


class MLP(nnx.Module):
    """
    Feedforward NN block, sub block of TransformerBlock
    """

    def __init__(self, embed_dim: int, rngs: nnx.Rngs):
        """
        n.b. https://github.com/openai/gpt-2/blob/master/src/model.py#L118-L119 shows this is originally implemented with Conv1D instead of linear,
        but future implementations show using a linear layer as expected -- I will use the linear for clarity's sake.

        ref: https://github.com/prabhudavidsheryl/flax_nnx_gpt2_fine_tuning/blob/main/gpt2.py#L69
        https://github.com/huggingface/transformers/issues/311
        """
        super().__init__()
        SCALING_FACTOR = (
            4  # https://github.com/openai/gpt-2/blob/master/src/model.py#L128
        )
        self.fc = nnx.Linear(embed_dim, SCALING_FACTOR * embed_dim, rngs=rngs)
        self.proj = nnx.Linear(SCALING_FACTOR * embed_dim, embed_dim, rngs=rngs)

    def __call__(self, X):
        X = self.fc(X)
        X = nnx.gelu(X)
        X = self.proj(X)

        return X


class TransformerBlock(nnx.Module):
    """
    Core transformer block with layernorm, attention layer, and MLP layer
    """

    def __init__(self, config: ModelConfig, rngs: nnx.rngs):
        super().__init__()
        self.ln_1 = nnx.LayerNorm(
            config.embed_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs
        )
        self.ln_2 = nnx.LayerNorm(
            config.embed_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs
        )

        self.mlp = MLP(config.embed_dim, rngs)
        self.attn = SelfAttention(
            config.embed_dim, config.n_heads, config.dropout, rngs=rngs
        )

    def __call__(self, X: jnp.array) -> jnp.array:
        X += self.attn(self.ln_1(X))
        X += self.MLP(self.ln_2(X))

        return X


class GPT2(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()

        self.config = config

        self.w_pos_emb = nnx.Embed(config.block_size, config.embed_dim, rngs=rngs)
        self.w_tok_emb = nnx.Embed(config.vocab_size, config.embed_dim, rngs=rngs)
        # self.dropout = nnx.Dropout(rate=config.dropout)
        self.hidden_layers: List[TransformerBlock] = [
            TransformerBlock(config) for _ in range(config.n_layers)
        ]

        self.ln_final = nnx.LayerNorm(
            config.embed_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs
        )
        self.logits_head = nnx.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs
        )

    def __call__(self, X):
        BATCH_SIZE, N_TOKENS = X.shape

        if N_TOKENS > self.config.block_size:
            raise IndexError(
                f"Block size is {self.config.block_size}, Input sequence length is {N_TOKENS}"
            )

        pos = jnp.arange(0, N_TOKENS)
        pos_embeddings = self.w_pos_emb(pos)  # (N_TOKENS, EMBED_DIM)
        tok_embeddings = self.w_tok_emb(X)  # (BATCH_SIZE, N_TOKENS, EMBED_DIM)

        embeddings = pos_embeddings + tok_embeddings

        # X_in = self.dropout(embeddings)

        for block in self.hidden_layers:
            X_in = block(X_in)

        X_in = self.ln_final(X_in)

        # TODO: Don't we need a softmax here?
        logits = self.logits_head(X_in)

        return logits
