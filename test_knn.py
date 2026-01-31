import os
import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import jittor as jt
from jittor import nn

# 引入项目模块 (确保 src/dataset.py 和 src/text_embedding.py 已经适配 Jittor)
from src.index import Indexer
from utils.utils import compute_metrics
from src.text_embedding import TextEmbeddingModel
from utils.Turing_utils import load_Turing
from utils.Deepfake_utils import load_deepfake
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4
from src.dataset import PassagesDataset

# 开启 CUDA
jt.flags.use_cuda = 1

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    jt.set_global_seed(seed)

def infer(dataloader, tokenizer, model):
    model.eval()
    allids, allembeddings, alllabels = [], [], []
    
    # Jittor 不需要 no_grad() 上下文，eval模式下通常不计算梯度，
    # 但显式用 jt.no_grad() 也是好习惯，或者直接在 model forward 时 jt.no_grad()
    with jt.no_grad():
        for batch in tqdm(dataloader):
            # Jittor dataset 返回的数据解包
            text, label, write_model, write_model_set, ids = batch
            
            # 使用 Tokenizer (返回 numpy 格式以便转 Jittor Var)
            encoded_batch = tokenizer.batch_encode_plus(
                text,
                return_tensors="np",  # 关键：这里用 np 而不是 pt
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            
            # 转为 Jittor Var
            input_ids = jt.array(encoded_batch['input_ids'])
            attention_mask = jt.array(encoded_batch['attention_mask'])
            
            # 模型推理
            # 假设 TextEmbeddingModel 接受 input_ids 和 attention_mask
            embeddings = model(input_ids, attention_mask=attention_mask)
            
            # 收集结果 (转回 numpy)
            all_embeddings.append(embeddings.numpy())
            
            # 处理 ids 和 labels
            # 如果 ids 是字符串列表，直接用；如果是 tensor，转 list
            if hasattr(ids, 'numpy'):
                allids.extend(ids.numpy().tolist())
            else:
                allids.extend(ids)
                
            if hasattr(label, 'numpy'):
                alllabels.extend(label.numpy().tolist())
            else:
                alllabels.extend(label)

    # 合并所有 batch 的结果
    if len(all_embeddings) > 0:
        allembeddings = np.concatenate(all_embeddings, axis=0)
    else:
        return [], [], []

    # 归一化 (L2 Norm)
    epsilon = 1e-6
    norms = np.linalg.norm(allembeddings, axis=1, keepdims=True) + epsilon
    allembeddings = allembeddings / norms

    # 去重逻辑 (通过字典)
    emb_dict, label_dict = {}, {}
    for i in range(len(allids)):
        emb_dict[allids[i]] = allembeddings[i]
        label_dict[allids[i]] = alllabels[i]

    clean_ids, clean_embeddings, clean_labels = [], [], []
    for key in emb_dict:
        clean_ids.append(key)
        clean_embeddings.append(emb_dict[key])
        clean_labels.append(label_dict[key])
    
    # Stack 成矩阵
    clean_embeddings = np.stack(clean_embeddings, axis=0)
    
    return clean_ids, clean_embeddings, clean_labels

def test(opt):
    print(f"Loading model from {opt.model_path} ...")
    # 加载模型
    model = TextEmbeddingModel(opt.model_name)
    
    # 加载权重 (Jittor 方式)
    # 注意：如果 checkpoint 是 PyTorch 的 .pth，需要转换；如果是 Jittor 的 .pkl，直接加载
    if opt.model_path.endswith('.pkl'):
        state_dict = jt.load(opt.model_path)
    else:
        # 如果是 .pth，尝试用 torch 加载并转 numpy (需要安装 torch)
        import torch
        print("Detected .pth file, attempting to convert from PyTorch checkpoint...")
        state_dict = torch.load(opt.model_path, map_location='cpu')
    
    # 移除 'model.' 前缀 (如果有)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
            
    model.load_state_dict(new_state_dict)
    tokenizer = model.tokenizer

    # 加载数据集
    if opt.mode=='deepfake':
        database = load_deepfake(opt.database_path)[opt.database_name]
        test_database = load_deepfake(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='OUTFOX':
        database=load_OUTFOX(opt.database_path,opt.attack)[opt.database_name]
        test_database = load_OUTFOX(opt.test_dataset_path,opt.attack)[opt.test_dataset_name]
    elif opt.mode=='Turing':
        database=load_Turing(opt.database_path)[opt.database_name]
        test_database = load_Turing(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='M4':
        database=load_M4(opt.database_path)[opt.database_name]+load_M4(opt.database_path)[opt.database_name.replace('train','dev')]
        test_database = load_M4(opt.test_dataset_path)[opt.test_dataset_name]
        
    passage_dataset = PassagesDataset(database, mode=opt.mode, need_ids=True)
    test_dataset = PassagesDataset(test_database, mode=opt.mode, need_ids=True)

    # Jittor DataLoader 配置
    passages_loader = passage_dataset.set_attrs(
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers
    )
    test_loader = test_dataset.set_attrs(
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers
    )

    print("Inferring Test Set...")
    test_ids, test_embeddings, test_labels = infer(test_loader, tokenizer, model)
    print("Inferring Database Set...")
    train_ids, train_embeddings, train_labels = infer(passages_loader, tokenizer, model)

    # 建立索引 (Faiss or Numpy based)
    index = Indexer(opt.embedding_dim)
    index.index_data(train_ids, train_embeddings)
    
    label_dict = {}
    for i in range(len(train_ids)):
        label_dict[train_ids[i]] = train_labels[i]

    if opt.save_database:
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        index.serialize(opt.save_path)
        with open(os.path.join(opt.save_path, 'label_dict.pkl'), 'wb') as f:
            pickle.dump(label_dict, f)

    test_labels = [str(l) for l in test_labels]
    preds = {i: [] for i in range(1, opt.max_K+1)}
    
    if len(test_embeddings.shape) == 1:
        test_embeddings = test_embeddings.reshape(1, -1)
        
    # KNN 搜索
    top_ids_and_scores = index.search_knn(test_embeddings, opt.max_K)
    
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        # 排序
        sorted_indices = np.argsort(scores)[::-1] # 降序
        
        zero_num, one_num = 0, 0
        for k in range(1, opt.max_K+1):
            idx = ids[sorted_indices[k-1]]
            # 确保 id 类型匹配 (int vs str)
            try:
                lbl = label_dict[int(idx)]
            except:
                lbl = label_dict[idx]
                
            if lbl == 0:
                zero_num += 1
            else:
                one_num += 1
            
            if zero_num > one_num:
                preds[k].append('0')
            else:
                preds[k].append('1')

    # 计算指标并绘图
    K_values = list(range(1, opt.max_K+1))
    metrics = {'human_rec': [], 'machine_rec': [], 'avg_rec': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    
    for k in range(1, opt.max_K+1):
        res = compute_metrics(test_labels, preds[k], test_ids)
        # unpack res: human_rec, machine_rec, avg_rec, acc, precision, recall, f1
        print(f"K={k}, HumanRec: {res[0]}, MachineRec: {res[1]}, AvgRec: {res[2]}, Acc:{res[3]}, F1:{res[6]}")
        metrics['human_rec'].append(res[0])
        metrics['machine_rec'].append(res[1])
        metrics['avg_rec'].append(res[2])
        metrics['acc'].append(res[3])
        metrics['precision'].append(res[4])
        metrics['recall'].append(res[5])
        metrics['f1'].append(res[6])
        
    # 绘图逻辑
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    keys = ['human_rec', 'machine_rec', 'avg_rec', 'acc', 'precision', 'recall', 'f1']
    titles = ['Human Rec', 'Machine Rec', 'Average Rec', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    markers = ['o', 'x', '^', 's', 'p', '*', 'D']
    
    # 简单的绘图循环
    idx = 0
    for i in range(3):
        for j in range(3):
            if idx < len(keys):
                k = keys[idx]
                axs[i, j].plot(K_values, metrics[k], marker=markers[idx], label=titles[idx])
                axs[i, j].set_title(titles[idx])
                axs[i, j].grid(True)
                idx += 1
            else:
                axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('performance_metrics_subplot.png', dpi=300)
    
    # 寻找最佳 K
    max_ids = np.argmax(metrics['avg_rec'])
    print(f"Find opt.max_K is {max_ids+1}")
    print(f"Best Metrics: HumanRec: {metrics['human_rec'][max_ids]}, AvgRec: {metrics['avg_rec'][max_ids]}, F1:{metrics['f1'][max_ids]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 移除 lightning/fabric 相关参数，保留核心参数
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=768)

    parser.add_argument('--mode', type=str, default='deepfake', help="deepfake,MGT or MGTDetect_CoCo")
    parser.add_argument("--database_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--database_name', type=str, default='train', help="train,valid,test,test_ood")
    parser.add_argument("--test_dataset_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--test_dataset_name', type=str, default='test', help="train,valid,test,test_ood")
    parser.add_argument("--attack", type=str, default="none", help="Attack type only for OUTFOX dataset")
    # 注意：这里默认路径可能需要改为你训练好的 savedir 下的模型
    parser.add_argument("--model_path", type=str, default="./runs/turing_a800/checkpoint-epoch-15.pkl")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base")

    parser.add_argument('--max_K', type=int, default=5)
    parser.add_argument('--pooling', type=str, default="average")
    parser.add_argument("--save_database", action='store_true')
    parser.add_argument("--save_path", type=str, default="database")
    
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    set_seed(opt.seed)
    test(opt)