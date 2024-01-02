import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional


class EmbeddingBlock(nn.Module):
    def __init__(self, dictionary_size: int, embedding_dim: int):
        super(EmbeddingBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_lookup_table = nn.Embedding(
            num_embeddings=dictionary_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        # From Section 3.4 of the paper: "In the embedding layers, we multiply those weights by sqrt(d_model)"
        return self.embedding_lookup_table(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_tokens: int, scalar: int = 1e4):
        super(PositionalEncoding, self).__init__()
        token_positions = torch.arange(0, max_tokens).unsqueeze(1)
        even_embedding_positions = torch.arange(0, embedding_dim, 2)
        # We encapsulate self.positional_encoding to nn.Parameter() so when moving a Transformer
        # instance to a device (GPU or CPU) self.positional_encoding will be moved too.
        self.positional_encoding = nn.Parameter(torch.zeros(max_tokens, embedding_dim))
        self.positional_encoding.requires_grad = False
        self.positional_encoding[:, 0::2] = torch.sin(
            token_positions / np.power(scalar, even_embedding_positions / embedding_dim))
        self.positional_encoding[:, 1::2] = torch.cos(
            token_positions / np.power(scalar, even_embedding_positions / embedding_dim))

    def forward(self, x):  # x.shape: (n_batch, n_tokens, embedding_dimension)
        assert (len(x.size()) == 3)
        n_tokens = x.size(1)
        return x + self.positional_encoding[: n_tokens, :]


def batched_scaled_dot_product_attention(Q, K, V, mask: Optional[torch.Tensor] = None):
    # Q, K, V size: (num_batches, num_tokens, dim)
    # Note that Q can have a different number of rows (i.e. tokens) with respect to K and V when computing
    # cross-attention!
    # e.g. In a english-to-italian translation model we may have
    # encoder_input: <start> I like you <end>  (5 tokens)
    # decoder_input: <start> Mi piaci <end>    (4 tokens)
    assert (len(Q.shape) == 3)
    assert (len(K.shape) == 3)
    assert (len(V.shape) == 3)
    num_batches_q, num_tokens_q, dim_q = Q.shape
    num_batches_k, num_tokens_k, dim_k = K.shape
    assert num_batches_q == num_batches_k
    assert dim_q == dim_k
    assert K.shape == V.shape
    raw_attention_values = torch.bmm(Q, K.permute(0, 2, 1)) / np.sqrt(dim_k)
    # Note that we normalize by np.sqrt(dim_k)  instead of np.sqrt(dim_q), this is due to the fact that we are
    # going to apply the softmax on each row of raw_attention_values (dq x dk),
    # and each row will be composed of dim_k elements.
    if mask is not None:
        assert mask.shape == (num_batches_q, num_tokens_q, num_tokens_k)
        # Fill values where mask == False with -np.inf
        raw_attention_values = raw_attention_values.masked_fill(~mask, -np.inf)
    soft_attention_values = nn.Softmax(dim=2)(raw_attention_values)
    return torch.bmm(soft_attention_values, V)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        number_of_attention_heads: int,
        embedding_dimension: int
    ):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dimension % number_of_attention_heads == 0
        self.d_k = embedding_dimension // number_of_attention_heads
        self.number_of_attention_heads = number_of_attention_heads

        # fixme: Is a single projection equivalent to number_of_attention_heads separate projections?
        self.Q_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.K_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.V_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.fnn_layer = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, Q, K, V, mask: Optional[torch.Tensor] = None):
        Q_proj = self.Q_projection(Q)
        K_proj = self.K_projection(K)
        V_proj = self.V_projection(V)

        d_k = self.d_k  # I unpack d_k merely for aesthetic reasons.

        x = torch.concat(
            [
                batched_scaled_dot_product_attention(
                    Q=Q_proj[:, :, i * d_k: (i + 1) * d_k],
                    K=K_proj[:, :, i * d_k: (i + 1) * d_k],
                    V=V_proj[:, :, i * d_k: (i + 1) * d_k],
                    mask=mask
                )  # output shape: (num_batches, num_tokens, d_k)
                for i in range(self.number_of_attention_heads)
            ],
            dim=-1  # We concatenate on the features dimension.
        )  # output shape: (num_batches, num_tokens, embedding_dimension)

        return self.fnn_layer(x)  # output shape: (num_batches, num_tokens, embedding_dimension)


class TransformerEncoderBlock(nn.Module):

    def __init__(
            self,
            embedding_dimension: int,
            n_attention_heads: int,
            feedforward_dimension: int,
            dropout_probability: float,
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.multi_head_self_attention = MultiHeadAttention(
            number_of_attention_heads=n_attention_heads,
            embedding_dimension=embedding_dimension
        )
        self.dropout_1 = nn.Dropout(p=dropout_probability)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embedding_dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dimension, feedforward_dimension),
            nn.ReLU(),
            nn.Linear(feedforward_dimension, embedding_dimension)
        )
        self.dropout_2 = nn.Dropout(p=dropout_probability)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embedding_dimension)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = self.layer_norm_1(
            x + self.dropout_1(self.multi_head_self_attention(Q=x, K=x, V=x, mask=mask))
        )
        x = self.layer_norm_2(
            x + self.dropout_2(self.feedforward(x))
        )
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            embedding_dimension: int,
            n_attention_heads: int,
            feedforward_dimension: int,
            dropout_probability: float,
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.masked_multi_head_self_attention = MultiHeadAttention(
            number_of_attention_heads=n_attention_heads,
            embedding_dimension=embedding_dimension
        )
        self.dropout_1 = nn.Dropout(p=dropout_probability)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embedding_dimension)

        self.multi_head_cross_attention = MultiHeadAttention(
            number_of_attention_heads=n_attention_heads,
            embedding_dimension=embedding_dimension
        )
        self.dropout_2 = nn.Dropout(p=dropout_probability)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embedding_dimension)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dimension, feedforward_dimension),
            nn.ReLU(),
            nn.Linear(feedforward_dimension, embedding_dimension)
        )
        self.dropout_3 = nn.Dropout(p=dropout_probability)
        self.layer_norm_3 = nn.LayerNorm(normalized_shape=embedding_dimension)

    def forward(
            self,
            x: torch.Tensor,
            encoder_output: torch.Tensor,
            self_attention_mask: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None
    ):

        x = self.layer_norm_1(
            x + self.dropout_1(self.masked_multi_head_self_attention(Q=x, K=x, V=x, mask=self_attention_mask))
        )
        x = self.layer_norm_2(
            x + self.dropout_2(self.multi_head_cross_attention(Q=x, K=encoder_output, V=encoder_output, mask=cross_attention_mask))  # noqa
        )
        x = self.layer_norm_3(
            x + self.dropout_3(self.feedforward(x))
        )
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            encoder_vocabulary_dimension: int,
            decoder_vocabulary_dimension: int,
            max_tokens: int,
            embedding_dimension: int,
            n_layers: int,
            number_of_attention_heads: int,
            feedforward_dimension: int,
            dropout_probability: float = 0.1
    ):
        super(Transformer, self).__init__()

        self.encoder_embedding_block = EmbeddingBlock(encoder_vocabulary_dimension, embedding_dimension)
        self.positional_encoding_block = PositionalEncoding(embedding_dimension, max_tokens)
        self.decoder_embedding_block = EmbeddingBlock(decoder_vocabulary_dimension, embedding_dimension)
        self.embedding_dimension = embedding_dimension
        self.n_layers = n_layers

        # We need to use nn.ModuleList otherwise the Transformer class will not properly register the individual
        # encoder blocks as its parameters (the optimizer may not update them and .to(device) won't have any effect)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dimension=embedding_dimension,
                n_attention_heads=number_of_attention_heads,
                feedforward_dimension=feedforward_dimension,
                dropout_probability=dropout_probability
            )
            for _ in range(n_layers)
        ])

        # We need to use nn.ModuleList otherwise the Transformer class will not properly register the individual
        # decoder blocks as its parameters (the optimizer may not update them and .to(device) won't have any effect)
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dimension=embedding_dimension,
                n_attention_heads=number_of_attention_heads,
                feedforward_dimension=feedforward_dimension,
                dropout_probability=dropout_probability
            )
            for _ in range(n_layers)
        ])
        self.fully_connected_layer = nn.Linear(embedding_dimension, decoder_vocabulary_dimension)  # just one?

    def forward(
            self,
            encoder_input_tokens: torch.Tensor,
            decoder_input_tokens: torch.Tensor,
            encoder_self_attention_mask: Optional[torch.Tensor] = None,
            decoder_self_attention_mask: Optional[torch.Tensor] = None,
            decoder_cross_attention_mask: Optional[torch.Tensor] = None

    ):

        encoder_input_embeddings = self.positional_encoding_block(self.encoder_embedding_block(encoder_input_tokens))
        decoder_input_embeddings = self.positional_encoding_block(self.decoder_embedding_block(decoder_input_tokens))
        # todo: do we need to detach() the masks?

        for i in range(self.n_layers):
            encoder_output = self.encoder_blocks[i](
                x=encoder_input_embeddings if i == 0 else encoder_output, # noqa
                mask=encoder_self_attention_mask  # We need to feed the mask to each layer, not only the first one.
            )

        for i in range(self.n_layers):
            decoder_output = self.decoder_blocks[i](
                x=decoder_input_embeddings if i == 0 else decoder_output,  # noqa
                encoder_output=encoder_output,  # called "memory" in other implementations  # noqa
                self_attention_mask=decoder_self_attention_mask,  # We need to feed the mask to each layer, not only the first one.  # noqa
                cross_attention_mask=decoder_cross_attention_mask,
            )
        return self.fully_connected_layer(decoder_output) # noqa




