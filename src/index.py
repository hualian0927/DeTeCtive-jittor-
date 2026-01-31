# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

# 辅助函数：如果输入是 Jittor Var，自动转 Numpy
def to_numpy(x):
    if hasattr(x, 'numpy'): # 兼容 Jittor Var
        return x.numpy()
    return x

class Indexer(object):

    def __init__(self, vector_sz, device='cuda'):
        self.index = faiss.IndexFlatIP(vector_sz)
        self.device = device
        
        # Faiss 的 GPU 处理是独立的，不受 Jittor 控制
        # 只要你装了 faiss-gpu，这里就可以工作
        if self.device == 'cuda':
            try:
                # Standard Faiss GPU transfer
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_all_gpus(self.index)
            except Exception as e:
                print(f"Warning: Failed to move Faiss index to GPU: {e}")
                print("Falling back to CPU index.")
                self.device = 'cpu'
                
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        
        # 自动转换 Jittor Var -> Numpy
        embeddings = to_numpy(embeddings)
        embeddings = embeddings.astype('float32')
        
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        print(f'Total data indexed {self.index.ntotal}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 8) -> List[Tuple[List[object], List[float]]]:
        # 自动转换 Jittor Var -> Numpy
        query_vectors = to_numpy(query_vectors)
        query_vectors = query_vectors.astype('float32')
        
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        # 如果数据很少，nbatch 可能为 0 或 1，tqdm 可能会闪烁，不影响逻辑
        iterator = range(nbatch)
        if nbatch > 1:
            iterator = tqdm(iterator, desc="KNN Searching")
            
        for k in iterator:
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')
        
        if self.device == 'cuda':
            save_index = faiss.index_gpu_to_cpu(self.index)
        else:
            save_index = self.index
            
        faiss.write_index(save_index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        if self.device == 'cuda':
            try:
                self.index = faiss.index_cpu_to_all_gpus(self.index)
            except Exception as e:
                print(f"Warning: Failed to move Faiss index to GPU during load: {e}")
                
        print(f'Loaded index of type {type(self.index)} and size {self.index.ntotal}')

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)

    def reset(self):
        self.index.reset()
        self.index_id_to_db_id = []
        print(f'Index reset, total data indexed {self.index.ntotal}')