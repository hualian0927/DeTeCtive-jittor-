import jittor as jt
from jittor import nn
from transformers import AutoTokenizer
import os
import torch # 仅用于下载权重路径处理，不参与计算，核心仍然是jittor的逻辑

# 引入写好的 Jittor 模型
from src.jittor_roberta import RobertaModel, RobertaConfig, load_pytorch_weights

class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name, output_hidden_states=False):
        super(TextEmbeddingModel, self).__init__()
        self.model_name = model_name
        
        # 1. 定义配置
        # SimCSE-RoBERTa-Base 的标准配置
        config = RobertaConfig(
            vocab_size=50265, 
            hidden_size=768, 
            num_hidden_layers=12, 
            num_attention_heads=12,
            max_position_embeddings=514
        )
        self.model = RobertaModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 2. 自动下载并加载 PyTorch 权重
        # 我们利用 transformers 库来帮我们找到权重文件的路径
        try:
            from transformers.utils import hub
            # 获取缓存的 pytorch_model.bin 路径
            resolved_file = hub.cached_file(model_name, filename="pytorch_model.bin")
            if resolved_file:
                load_pytorch_weights(self.model, resolved_file)
            else:
                print("Warning: Could not find pytorch_model.bin, using random weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Please ensure you have internet access or download 'pytorch_model.bin' manually.")

    def pooling(self, model_output, attention_mask, use_pooling='average', hidden_states=False):
        # model_output: [Batch, Seq, Hidden]
        # attention_mask: [Batch, Seq]
        
        mask_expanded = attention_mask.unsqueeze(-1) # [B, L, 1]
        
        # Jittor 广播乘法
       
        if use_pooling == "average":
            # Masking
            masked_output = model_output * mask_expanded
            sum_embeddings = masked_output.sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            sum_mask = jt.clamp(sum_mask, min_v=1e-9)
            emb = sum_embeddings / sum_mask
        elif use_pooling == "cls":
            emb = model_output[:, 0]
            
        return emb

    def execute(self, input_ids, attention_mask, use_pooling='average', hidden_states=False):
        # 这里的 input_ids, attention_mask 已经是 Jittor Var
        
        # 调用手写的 Jittor RoBERTa
        sequence_output, pooled_output = self.model(input_ids, attention_mask)
        
        # Pooling
        emb = self.pooling(sequence_output, attention_mask, use_pooling)
        
        # Normalize
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
        return emb