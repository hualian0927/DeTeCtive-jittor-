import os
import pickle
import random
import numpy as np
import argparse
from tqdm import tqdm
import jittor as jt
from jittor import nn
from src.index import Indexer
from src.text_embedding import TextEmbeddingModel
from src.dataset import PassagesDataset
from utils.Turing_utils import load_Turing
from utils.Deepfake_utils import load_deepfake
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4

# 开启 CUDA
jt.flags.use_cuda = 1

def infer(dataset, tokenizer, model):
    """
    Jittor 版推理函数
    """
    model.eval()
    
    allids, allembeddings, alllabels = [], [], []
    
    # Jittor 的 Dataset 可以直接迭代，自带 batch 处理
    # 进度条包装
    iterator = tqdm(dataset, total=len(dataset) // dataset.batch_size)
    
    # Jittor 不需要 no_grad，且计算图是动态构建的，推理时仅执行计算即可
    for batch in iterator:
        # 解包 Batch
        # 假设 PassagesDataset 返回的是 (text_list, label_list, ..., ids_list)
        text, label, write_model, write_model_set, ids = batch
        
        # 1. Tokenizer 处理 (关键：使用 return_tensors="np")
        encoded_batch = tokenizer.batch_encode_plus(
            text,
            return_tensors="np",  
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        
        # 2. 转换为 Jittor Var
        input_ids = jt.array(encoded_batch['input_ids'])
        attention_mask = jt.array(encoded_batch['attention_mask'])
        
        # 3. 模型前向传播
        # 假设你的 Jittor 模型 forward/execute 接收 input_ids 和 attention_mask
        embeddings = model(input_ids, attention_mask)
        
        # 4. 收集结果
        # 将 Jittor Var 转回 numpy 存入列表
        allembeddings.append(embeddings.numpy())
        
        # 处理 ID 和 Label (Jittor Dataset 返回的通常是 numpy array 或 list)
        if isinstance(ids, np.ndarray) or isinstance(ids, jt.Var):
            allids.extend(ids.tolist())
        else:
            allids.extend(ids) # 如果是 list
            
        if isinstance(label, np.ndarray) or isinstance(label, jt.Var):
            alllabels.extend(label.tolist())
        else:
            alllabels.extend(label)

    # 拼接所有 Embeddings
    allembeddings = np.concatenate(allembeddings, axis=0)
    
    # 5. 归一化 (在 Numpy 中进行，与 torch.nn.functional.normalize 效果一致)
    # L2 Norm
    norms = np.linalg.norm(allembeddings, axis=1, keepdims=True)
    allembeddings = allembeddings / (norms + 1e-6)
    
    # 6. 去重逻辑 
    emb_dict, label_dict = {}, {}
    for i in range(len(allids)):
        emb_dict[allids[i]] = allembeddings[i]
        label_dict[allids[i]] = alllabels[i]
        
    final_ids, final_embeddings, final_labels = [], [], []
    for key in emb_dict:
        final_ids.append(key)
        final_embeddings.append(emb_dict[key])
        final_labels.append(label_dict[key])
        
    # Stack 对应 numpy 的 array
    return final_ids, np.array(final_embeddings), final_labels

def set_seed(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(opt):
    # 初始化模型
    print(f"Loading model: {opt.model_name}")
    model = TextEmbeddingModel(opt.model_name)
    
    # 加载权重
    print(f"Loading checkpoint from: {opt.model_path}")
    state_dict = jt.load(opt.model_path)
    
    # 权重 Key 映射处理 (去除 'model.' 前缀)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    
    tokenizer = model.tokenizer

    # 加载数据库元数据
    if opt.mode == 'deepfake':
        database = load_deepfake(opt.database_path)[opt.database_name]
    elif opt.mode == 'OUTFOX':
        database = load_OUTFOX(opt.database_path)[opt.database_name]
    elif opt.mode == 'Turing':
        database = load_Turing(opt.database_path)[opt.database_name]
    elif opt.mode == 'M4':
        database = load_M4(opt.database_path)[opt.database_name] + \
                   load_M4(opt.database_path)[opt.database_name.replace('train', 'dev')]
        
    # 初始化 Dataset
    # 注意：需确保 src.dataset.PassagesDataset 继承自 jittor.dataset.Dataset
    passage_dataset = PassagesDataset(database, mode=opt.mode, need_ids=True)
    
    # Jittor 设置 Batch Size 和 Workers (替代 DataLoader)
    passage_dataset.set_attrs(
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False, # 生成数据库通常不需要 shuffle
        drop_last=False
    )
    
    print(f"Total samples: {len(passage_dataset)}")

    # 执行推理
    train_ids, train_embeddings, train_labels = infer(passage_dataset, tokenizer, model)

    # 建立索引 (Assuming Indexer is FAISS or similar independent of torch)
    index = Indexer(opt.embedding_dim)
    index.index_data(train_ids, train_embeddings)
    
    label_dict = {}
    for i in range(len(train_ids)):
        label_dict[train_ids[i]] = train_labels[i]

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
        
    print(f"Saving database to {opt.save_path}...")
    index.serialize(opt.save_path)
    
    # 保存 Label Dict
    with open(os.path.join(opt.save_path, 'label_dict.pkl'), 'wb') as f:
        pickle.dump(label_dict, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device_num 在 Jittor 中通常不需要手动指定，jt.flags.use_cuda=1 会自动使用 GPU
    parser.add_argument('--device_num', type=int, default=1) 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=768)

    parser.add_argument('--mode', type=str, default='deepfake', help="deepfake,MGT or MGTDetect_CoCo")
    parser.add_argument("--database_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--database_name', type=str, default='train', help="train,valid,test,test_ood")
    parser.add_argument("--model_path", type=str, default="/home/heyongxin/detect-LLM-text/DAT/pth/unseen_model/model_best_gpt35.pth",\
                         help="Path to the embedding model checkpoint")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base", help="Model name")
    parser.add_argument("--save_path", type=str, default="database", help="Path to save the database")
    
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    
    set_seed(opt.seed)
    test(opt)