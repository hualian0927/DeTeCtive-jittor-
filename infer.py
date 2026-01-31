import os
import pickle
import numpy as np
import argparse
import random
import jittor as jt
from src.index import Indexer
from src.text_embedding import TextEmbeddingModel

# 开启 CUDA
jt.flags.use_cuda = 1

def set_seed(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def infer(opt):
    # 1. 初始化模型
    print(f"Loading model: {opt.model_name}")
    model = TextEmbeddingModel(opt.model_name)
    
    # 2. 加载权重
    print(f"Loading checkpoint: {opt.model_path}")
    state_dict = jt.load(opt.model_path)
    
    # 处理权重键名 (移除 'model.' 前缀以匹配模型定义)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    model.eval() # 切换到评估模式
    
    tokenizer = model.tokenizer

    # 3. 加载索引数据库
    print(f"Loading index from: {opt.database_path}")
    index = Indexer(opt.embedding_dim)
    index.deserialize_from(opt.database_path)
    
    label_path = os.path.join(opt.database_path, 'label_dict.pkl')
    print(f"Loading labels from: {label_path}")
    label_dict = load_pkl(label_path)
    
    # 4. 处理输入文本
    text = opt.text
    # 关键修改：使用 return_tensors="np"
    encoded_text = tokenizer.batch_encode_plus(
                        [text],
                        return_tensors="np", 
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                    )
    
    # 转换为 Jittor Var
    input_ids = jt.array(encoded_text['input_ids'])
    attention_mask = jt.array(encoded_text['attention_mask'])
    
    # 5. 模型推理
    # 直接调用模型，获取 Numpy 结果
    embeddings = model(input_ids, attention_mask).numpy()
    
    # 6. KNN 搜索
    top_ids_and_scores = index.search_knn(embeddings, opt.K)
    
    # 7. 输出结果
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        print(f"\nTop {opt.K} results for text:")
        cnt = {0: 0, 1: 0}
        for j, (id, score) in enumerate(zip(ids, scores)):
            # 确保 id 是 int 类型用于查字典
            label = label_dict.get(int(id), -1) 
            label_str = "Human" if label == 1 else ("AI" if label == 0 else "Unknown")
            print(f"{j+1}. ID {id} Label {label} ({label_str}) Score {score:.4f}")
            
            if label in cnt:
                cnt[label] += 1
                
        # 投票机制
        print("-" * 30)
        if cnt[0] > cnt[1]:
            print(">>> Predicted result: AI-generated Text")
        else:
            print(">>> Predicted result: Human-written Text")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--database_path', type=str, default="database", help="Path to the index file")

    # 这里的文件路径替换本地的实际路径
    parser.add_argument("--model_path", type=str, default="/home/heyongxin/detect-LLM-text/DAT/pth/unseen_model/model_best_gpt35.pth",\
                         help="Path to the embedding model checkpoint")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base", help="Model name")

    parser.add_argument('--K', type=int, default=5, help="Search [1,K] nearest neighbors,choose the best K")
    parser.add_argument('--pooling', type=str, default="average", help="Pooling method, average or cls")
    
    # 默认测试文本
    default_text = "I really want someone to change my view on this, since everyone I know are frowning on me for thinking this way. My argument is, that just with my single vote wouldn't have any effect in the result and thus, it's not worth voting at all But if you don't vote then your opinion doesn't count"
    parser.add_argument('--text', type=str, default=default_text)
    
    parser.add_argument('--seed', type=int, default=0)

    opt = parser.parse_args()
    set_seed(opt.seed)
    infer(opt)