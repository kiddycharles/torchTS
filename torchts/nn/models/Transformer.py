from torchts.nn.layers.TransformerEnc import Encoder, EncoderLayer
from torchts.nn.layers.TransformerDec import Decoder, DecoderLayer
from torchts.nn.layers.attention import FullAttention, AttentionLayer
from torchts.nn.layers.embedding import *


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self,
                 enc_in=7,
                 dec_in=7,
                 c_out=7,
                 out_len=24,
                 factor=5,
                 d_model=512,
                 n_heads=8,
                 e_layers=2,
                 d_layers=1,
                 d_ff=2048,
                 dropout=0.05,
                 embedding_type="fixed",
                 frequency="h",
                 activation="gelu",
                 output_attention=False,
                 **kwargs):
        super(Transformer, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embedding_type, frequency, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embedding_type, frequency, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
            self,
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None
    ):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attentions = self.encoder(enc_out, attention_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attentions
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
