import os
import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import jittor as jt
from jittor import nn

from src.index import Indexer
from utils.utils import compute_metrics
from src.text_embedding import TextEmbeddingModel
from utils.Turing_utils import load_Turing
from utils.Deepfake_utils import load_deepfake
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4
from src.dataset import PassagesDataset

jt.flags.use_cuda = 1

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    jt.set_global_seed(seed)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# infer 函数与 test_knn.py 中相同
def infer(dataloader, tokenizer, model):
    model.eval()
    allids, allembeddings, alllabels = [], [], []
    with jt.no_grad():
        for batch in tqdm(dataloader):
            text, label, write_model, write_model_set, ids = batch
            encoded_batch = tokenizer.batch_encode_plus(
                text,
                return_tensors="np",
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            input_ids = jt.array(encoded_batch['input_ids'])
            attention_mask = jt.array(encoded_batch['attention_mask'])
            
            embeddings = model(input_ids, attention_mask=attention_mask)
            all_embeddings.append(embeddings.numpy())
            
            if hasattr(ids, 'numpy'):
                allids.extend(ids.numpy().tolist())
            else:
                allids.extend(ids)
            if hasattr(label, 'numpy'):
                alllabels.extend(label.numpy().tolist())
            else:
                alllabels.extend(label)

    if len(all_embeddings) > 0:
        allembeddings = np.concatenate(all_embeddings, axis=0)
    else:
        return [], [], []

    epsilon = 1e-6
    norms = np.linalg.norm(allembeddings, axis=1, keepdims=True) + epsilon
    allembeddings = allembeddings / norms

    emb_dict, label_dict = {}, {}
    for i in range(len(allids)):
        emb_dict[allids[i]] = allembeddings[i]
        label_dict[allids[i]] = alllabels[i]

    clean_ids, clean_embeddings, clean_labels = [], [], []
    for key in emb_dict:
        clean_ids.append(key)
        clean_embeddings.append(emb_dict[key])
        clean_labels.append(label_dict[key])
    
    clean_embeddings = np.stack(clean_embeddings, axis=0)
    return clean_ids, clean_embeddings, clean_labels

def test(opt):
    print(f"Loading model from {opt.model_path} ...")
    model = TextEmbeddingModel(opt.model_name)
    
    if opt.model_path.endswith('.pkl'):
        state_dict = jt.load(opt.model_path)
    else:
        import torch
        state_dict = torch.load(opt.model_path, map_location='cpu')
    
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    tokenizer = model.tokenizer

    if opt.mode=='deepfake':
        test_database = load_deepfake(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='OUTFOX':
        test_database = load_OUTFOX(opt.test_dataset_path,opt.attack)[opt.test_dataset_name]
    elif opt.mode=='Turing':
        test_database = load_Turing(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='M4':
        test_database = load_M4(opt.test_dataset_path)[opt.test_dataset_name]
        
    test_dataset = PassagesDataset(test_database, mode=opt.mode, need_ids=True)
    test_loader = test_dataset.set_attrs(batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    print("Inferring Test Set...")
    test_ids, test_embeddings, test_labels = infer(test_loader, tokenizer, model)

    # 加载已保存的索引
    print(f"Loading database from {opt.database_path}...")
    index = Indexer(opt.embedding_dim)
    index.deserialize_from(opt.database_path)
    label_dict = load_pkl(os.path.join(opt.database_path, 'label_dict.pkl'))
    
    test_labels = [str(l) for l in test_labels]
    preds = {i: [] for i in range(1, opt.max_K+1)}
    
    if len(test_embeddings.shape) == 1:
        test_embeddings = test_embeddings.reshape(1, -1)
        
    top_ids_and_scores = index.search_knn(test_embeddings, opt.max_K)
    
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        sorted_indices = np.argsort(scores)[::-1]
        zero_num, one_num = 0, 0
        for k in range(1, opt.max_K+1):
            idx = ids[sorted_indices[k-1]]
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
    
    # 指标计算 (与 test_knn 相同，此处简化输出)
    for k in range(1, opt.max_K+1):
        res = compute_metrics(test_labels, preds[k], test_ids)
        print(f"K={k}, AvgRec: {res[2]}, F1:{res[6]}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--database_path', type=str, default="database", help="Path to the index file")

    parser.add_argument('--mode', type=str, default='deepfake')
    parser.add_argument("--test_dataset_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument('--test_dataset_name', type=str, default='test')
    parser.add_argument("--attack", type=str, default="none")
    parser.add_argument("--model_path", type=str, default="./runs/turing_a800/checkpoint-epoch-15.pkl")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base")
    parser.add_argument('--max_K', type=int, default=51)
    parser.add_argument('--pooling', type=str, default="average")
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    set_seed(opt.seed)
    test(opt)