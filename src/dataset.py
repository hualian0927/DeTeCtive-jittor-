from jittor.dataset import Dataset
from utils.Turing_utils import turing_model_set, turing_name_dct

# 可导入 OUTFOX，如果没有也不影响 TuringBench 运行
try:
    from utils.OUTFOX_utils import outfox_model_set 
except ImportError:
    pass

class PassagesDataset(Dataset):
    def __init__(self, dataset, mode='Turing', need_ids=False):
        super().__init__()
        self.mode = mode
        self.dataset = dataset
        self.need_ids = need_ids
        self.classes = []
        self.model_name_set = {}
        
        # ================== TuringBench 逻辑 ==================
        if mode == 'Turing':
            cnt = 0
            for model_set_name, model_set in turing_name_dct.items():
                for name in model_set:
                    self.model_name_set[name] = (cnt, turing_model_set[model_set_name])
                    self.classes.append(name)
                    cnt += 1
        
        # ================== 通用逻辑 (OUTFOX/M4) ==================
        elif mode == 'OUTFOX' or mode == 'M4':
            LLM_name = set()
            for item in self.dataset:
                LLM_name.add(item[2])
            for i, name in enumerate(LLM_name):
                self.model_name_set[name] = (i, i)
                self.classes.append(name)
        
        else:
            print(f"Warning: Unknown mode {mode}, dataset might not load correctly.")

        print(f'Dataset Mode: {mode}, Classes found: {len(self.classes)}')
        
        # Jittor Dataset 基础配置
        self.set_attrs(batch_size=1, shuffle=False)
    
    def get_class(self):
        return self.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 读取数据
        text, label, src, id_val = self.dataset[idx]
        
        write_model, write_model_set = 1000, 1000
        
        # 1. 尝试通过 key 匹配
        for name in self.model_name_set.keys():
            if name in src:
                write_model, write_model_set = self.model_name_set[name]
                break
        
        # 2. TuringBench 特殊兜底：如果 src 本身就是模型名
        if write_model == 1000 and self.mode == 'Turing':
             if src in self.model_name_set:
                 write_model, write_model_set = self.model_name_set[src]

        # 如果还是没匹配到，报错提示（方便排查数据问题）
        assert write_model != 1000, f'write_model matching failed for src: {src}'

        if self.need_ids:
            return text, int(label), int(write_model), int(write_model_set), int(id_val)
        else:
            return text, int(label), int(write_model), int(write_model_set)