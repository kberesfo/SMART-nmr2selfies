from ..config import DecoderSettings


class DecoderLayer(nn.):
    def __init__(self, cfg: DecoderSettings)
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.num_heads,
            dropout=cfg.dropout
            )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model)
        )

    def forward(self, x, mask=None):
        

class Decoder():
    def __init__(self, cfg: DecoderSettings):
        pass

    def decode(self, token_ids):
