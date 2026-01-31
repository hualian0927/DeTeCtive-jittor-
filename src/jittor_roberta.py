import jittor as jt
from jittor import nn
import math
import torch # 仅用于加载权重，不参与整个复现逻辑，复现逻辑仍然全部时jittor

class RobertaConfig:
    def __init__(self, vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=514):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = hidden_size * 4

class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def execute(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = jt.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Mask 处理
            scores = scores + (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -1e9
        
        attn = nn.softmax(scores, dim=-1)
        out = jt.matmul(attn, v).transpose(0, 2, 1, 3).view(batch_size, seq_len, -1)
        return out

class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def execute(self, x, mask=None):
        x = self.ln1(x + self.attention(x, mask))
        x = self.ln2(x + self.ffn(x))
        return x

class RobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) 
        self.layers = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_layers)])
        self.padding_idx = 1
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

    def execute(self, input_ids, mask=None):
        seq_len = input_ids.shape[1]
        pos_ids = jt.arange(self.padding_idx + 1, seq_len + self.padding_idx + 1).unsqueeze(0).expand(input_ids.shape[0], seq_len)
        
        x = self.embeddings(input_ids) + self.pos_embeddings(pos_ids)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        # 计算 Pooled Output (取第一个 token [CLS])
        first_token_tensor = x[:, 0]
        pooled_output = self.pooler_activation(self.pooler_dense(first_token_tensor))
        
        return x, pooled_output

def load_pytorch_weights(jittor_model, pt_path):
    print(f"Loading weights from {pt_path}...")
    try:
        pt_state = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load pytorch file: {e}")
        return
    print("Weight file loaded (Partial/Random init for now).")