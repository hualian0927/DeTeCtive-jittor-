import jittor as jt
from jittor import nn
import argparse
import os
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoConfig
import torch

# === 核心引入 ===
from src.jittor_roberta import RobertaModel
from src.dataset import PassagesDataset
from utils.Turing_utils import load_Turing

# 开启 CUDA
jt.flags.use_cuda = 1

# === 自定义分类模型 Wrapper ===
class RobertaForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        
        print(f"Loading Config for {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        
        # === [终极修复] 兼容性全量补丁 ===
        # 1. 映射 layers
        if not hasattr(self.config, "num_layers"):
            setattr(self.config, "num_layers", self.config.num_hidden_layers)
            print(f"🔧 Patch: num_layers -> {self.config.num_layers}")
            
        # 2. 映射 heads (修复当前的报错)
        if not hasattr(self.config, "num_heads"):
            setattr(self.config, "num_heads", self.config.num_attention_heads)
            print(f"🔧 Patch: num_heads -> {self.config.num_heads}")

        # 3. 映射 hidden_size (预防性修复)
        if not hasattr(self.config, "dim"):
            # 有些简版实现可能叫 dim
            setattr(self.config, "dim", self.config.hidden_size)

        # 初始化基础模型
        self.model = RobertaModel(self.config)
        self.num_labels = num_labels
        
        # 定义分类头
        self.classifier = nn.Linear(768, num_labels)

        # 加载权重
        self.load_pytorch_weights(model_name)

    def load_pytorch_weights(self, model_name):
        print(f"Loading PyTorch weights from {model_name}...")
        try:
            from transformers.utils import cached_file
            archive_file = cached_file(model_name, "pytorch_model.bin")
            if not archive_file:
                archive_file = cached_file(model_name, "model.bin")
                
            if archive_file:
                print(f"Found weights at: {archive_file}")
                pt_state = torch.load(archive_file, map_location='cpu')
                
                jt_state = {}
                for name, param in pt_state.items():
                    # 映射权重名称
                    new_name = name
                    if name.startswith("roberta."):
                        new_name = "model." + name[8:]
                    elif name.startswith("classifier.") or name.startswith("pooler."):
                        continue 
                    
                    jt_state[new_name] = param.detach().cpu().numpy()
                
                self.load_state_dict(jt_state)
                print("✅ Pre-trained weights loaded successfully!")
            else:
                print("⚠️ Warning: Could not find pytorch_model.bin")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")

    def execute(self, input_ids, attention_mask=None, labels=None):
        # 1. 跑基础模型
        outputs = self.model(input_ids, attention_mask)
        
        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs
            
        # 2. 提取 [CLS] Token
        cls_token = sequence_output[:, 0, :]
        
        # 3. 过分类层
        logits = self.classifier(cls_token)
        
        # 4. 计算 Loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return [loss, logits]
        
        return [logits]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=str, default='1')
    parser.add_argument('--per_gpu_batch_size', type=int, default=32)
    parser.add_argument('--total_epoch', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--model_name', type=str, default='princeton-nlp/unsup-simcse-roberta-base')
    parser.add_argument('--dataset', type=str, default='TuringBench')
    parser.add_argument('--path', type=str, default='./datasets/TuringBench/AA')
    parser.add_argument('--name', type=str, default='a800-run-final')
    parser.add_argument('--freeze_embedding_layer', action='store_true')
    parser.add_argument('--database_name', type=str, default='train')
    parser.add_argument('--test_dataset_name', type=str, default='test')
    parser.add_argument('--savedir', type=str, default='./runs/turing_a800')
    return parser.parse_args()

def evaluate(model, dataloader, tokenizer):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running Evaluation...")
    with jt.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts, labels, _, _ = batch
            encoded = tokenizer(list(texts), padding='max_length', truncation=True, max_length=512, return_tensors='np')
            input_ids = jt.array(encoded['input_ids'])
            attention_mask = jt.array(encoded['attention_mask'])
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            preds = np.argmax(logits.numpy(), axis=1)
            
            all_preds.extend(preds)
            if hasattr(labels, 'tolist'):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(list(labels))
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

def train(args, tokenizer):
    log_dir = os.path.join(args.savedir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard Log Dir: {log_dir}")

    print(f"Loading Raw Data from {args.path}...")
    full_data = load_Turing(args.path)
    train_raw = full_data[args.database_name]
    test_raw = full_data[args.test_dataset_name]

    train_dataset = PassagesDataset(train_raw, mode='Turing', need_ids=False)
    test_dataset = PassagesDataset(test_raw, mode='Turing', need_ids=False)

    train_loader = list(train_dataset.set_attrs(
        batch_size=args.per_gpu_batch_size, shuffle=True, num_workers=args.num_workers
    ))
    test_loader = list(test_dataset.set_attrs(
        batch_size=args.per_gpu_batch_size, shuffle=False, num_workers=args.num_workers
    ))

    print(f"Loading model: {args.model_name}, Num Classes: {len(train_dataset.classes)}")
    
    # 初始化模型
    model = RobertaForSequenceClassification(args.model_name, num_labels=len(train_dataset.classes))
    
    if args.freeze_embedding_layer:
        for param in model.model.embeddings.parameters():
            param.stop_grad()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = nn.AdamW(params, lr=args.lr)

    global_step = 0
    total_steps = len(train_loader) * args.total_epoch
    
    model.train()
    
    for epoch in range(args.total_epoch):
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.total_epoch}")
        loss_sum = 0.0
        step_count = 0

        for batch in iterator:
            texts, labels, _, _ = batch
            encoded = tokenizer(list(texts), padding='max_length', truncation=True, max_length=512, return_tensors='np')
            
            input_ids = jt.array(encoded['input_ids'])
            attention_mask = jt.array(encoded['attention_mask'])
            labels = jt.array(labels)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            
            optimizer.step(loss)
            
            global_step += 1
            if global_step < args.warmup_steps:
                lr_factor = float(global_step) / float(max(1, args.warmup_steps))
            else:
                progress = float(global_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
                lr_factor = max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
            current_lr = args.lr * lr_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            current_loss = loss.item()
            loss_sum += current_loss
            step_count += 1
            avg_loss = loss_sum / step_count

            iterator.set_description(f"Loss: {current_loss:.4f} Avg: {avg_loss:.4f} LR: {current_lr:.6f}")
            
            if global_step % 10 == 0:
                writer.add_scalar("Train/Loss", current_loss, global_step)
                writer.add_scalar("Train/LR", current_lr, global_step)

        print(f"\nEnd of Epoch {epoch+1}, Evaluating...")
        acc, f1 = evaluate(model, test_loader, tokenizer)
        print(f"🏆 Epoch {epoch+1} Result: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
        
        writer.add_scalar("Test/Accuracy", acc, global_step)
        writer.add_scalar("Test/F1", f1, global_step)

        model.train()
        save_path = os.path.join(args.savedir, f"checkpoint-epoch-{epoch+1}.pkl")
        model.save(save_path)
        print(f"Saved model to {save_path}")

    print("Training Complete!")
    writer.close()

if __name__ == "__main__":
    args = get_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        print("Tokenizer load failed, trying roberta-base...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
    train(args, tokenizer)