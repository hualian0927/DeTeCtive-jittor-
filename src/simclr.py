import jittor as jt
from jittor import nn
from src.text_embedding import TextEmbeddingModel

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, in_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.dense1 = nn.Linear(in_dim, in_dim//4)
        self.dense2 = nn.Linear(in_dim//4, in_dim//16)
        self.out_proj = nn.Linear(in_dim//16, out_dim)

    def execute(self, features):
        x = features
        x = self.dense1(x)
        x = jt.tanh(x)
        x = self.dense2(x)
        x = jt.tanh(x)
        x = self.out_proj(x)
        return x

class SimCLR_Classifier(nn.Module):
    def __init__(self, opt):
        super(SimCLR_Classifier, self).__init__()
        self.opt = opt
        self.temperature = opt.temperature
        
        # 初始化 RoBERTa 模型
        self.model = TextEmbeddingModel(opt.model_name)
        
        # 分类头
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        
        # 损失函数权重
        self.a = opt.a
        self.b = opt.b
        self.c = opt.c
        self.d = opt.d
        self.esp = 1e-6
        self.only_classifier = opt.only_classifier

    def _compute_logits(self, q, q_index1, q_index2, q_label):
        """
        计算多级对比学习的 Logits。
        q: [Batch, Hidden]
        q_index1: Write Model ID (细粒度模型)
        q_index2: Write Model Set ID (模型家族)
        q_label: 0 (Machine) / 1 (Human)
        """
        # 单卡模式下，Query 和 Key 是同一个 Batch
        k = q 
        k_index1 = q_index1
        k_index2 = q_index2
        k_label = q_label

        # 1. 归一化
        q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-9)
        
        # 2. 计算余弦相似度矩阵 [N, N]
        logits = jt.matmul(q_norm, k_norm.transpose()) / self.temperature

        # 3. 准备 Mask
        # 调整形状以便广播: [N, 1] vs [1, N]
        q_idx1 = q_index1.view(-1, 1)
        k_idx1 = k_index1.view(1, -1)
        
        q_idx2 = q_index2.view(-1, 1)
        k_idx2 = k_index2.view(1, -1)
        
        q_lbl = q_label.view(-1, 1)
        k_lbl = k_label.view(1, -1)

        # 生成布尔掩码
        same_model = (q_idx1 == k_idx1)  # 同一模型
        same_set = (q_idx2 == k_idx2)    # 同一家族
        same_label = (q_lbl == k_lbl)    # 同一类别(人/机)

        # 区分人类和机器的样本索引
        is_human = (q_label == 1)
        is_machine = (q_label == 0)

        # ==================== 1. Human Loss (仅针对人类样本) ====================
        # 正样本: 其他人类样本 (same_label)
        # 负样本: 机器样本 (not same_label)
        # 正样本分数聚合: Sum(Pos) / Count(Pos)
        sum_pos_human = jt.sum(logits * same_label, dim=1)
        count_pos_human = jt.sum(same_label, dim=1)
        pos_logits_human = sum_pos_human / jt.maximum(count_pos_human, self.esp)
        
        # 负样本分数
        neg_logits_human = logits * jt.logical_not(same_label)
        
        # 拼接: [Pos_Score, Neg_Scores...] -> [N, N+1]
        logits_human_all = jt.concat((pos_logits_human.unsqueeze(1), neg_logits_human), dim=1)
        
        # 只保留人类样本的行
        logits_human_final = logits_human_all[is_human]

        # ==================== 2. Model Loss (仅针对机器样本) ====================
        # 正样本: 同模型 (same_model)
        # 负样本: 非同模型 (not same_model)
        sum_pos_model = jt.sum(logits * same_model, dim=1)
        count_pos_model = jt.sum(same_model, dim=1)
        pos_logits_model = sum_pos_model / jt.maximum(count_pos_model, self.esp)
        
        neg_logits_model = logits * jt.logical_not(same_model)
        logits_model_all = jt.concat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1)
        logits_model_final = logits_model_all[is_machine]

        # ==================== 3. Set Loss (仅针对机器样本) ====================
        # 正样本: 同家族 但 不同模型 (same_set XOR same_model)
        # 负样本: 不同家族 (not same_set)
        mask_set_pos = jt.logical_xor(same_set, same_model)
        
        sum_pos_set = jt.sum(logits * mask_set_pos, dim=1)
        count_pos_set = jt.sum(mask_set_pos, dim=1)
        pos_logits_set = sum_pos_set / jt.maximum(count_pos_set, self.esp)
        
        neg_logits_set = logits * jt.logical_not(same_set)
        logits_set_all = jt.concat((pos_logits_set.unsqueeze(1), neg_logits_set), dim=1)
        logits_set_final = logits_set_all[is_machine]

        # ==================== 4. Label Loss (仅针对机器样本) ====================
        # 正样本: 同为机器 但 不同家族 (same_label XOR same_set)
        # 负样本: 人类 (not same_label)
        mask_label_pos = jt.logical_xor(same_label, same_set)
        
        sum_pos_lbl = jt.sum(logits * mask_label_pos, dim=1)
        count_pos_lbl = jt.sum(mask_label_pos, dim=1)
        pos_logits_lbl = sum_pos_lbl / jt.maximum(count_pos_lbl, self.esp)
        
        neg_logits_lbl = logits * jt.logical_not(same_label)
        logits_lbl_all = jt.concat((pos_logits_lbl.unsqueeze(1), neg_logits_lbl), dim=1)
        logits_label_final = logits_lbl_all[is_machine]

        return logits_model_final, logits_set_final, logits_label_final, logits_human_final

    def execute(self, input_ids, mask, write_model, write_model_set, label):
        # 1. 前向传播获取 Embedding
        q = self.model(input_ids, mask)
        
        # 2. 计算对比学习 Logits
        logits_model, logits_set, logits_label, logits_human = self._compute_logits(
            q, write_model, write_model_set, label
        )

        # 3. 计算分类 Logits
        out = self.classifier(q)
        
        # 4. 计算损失
        # 分类损失
        if self.opt.AA:
            loss_classify = nn.cross_entropy_loss(out, write_model)
        else:
            loss_classify = nn.cross_entropy_loss(out, label)

        # 对比损失辅助函数 (处理空 Batch 情况)
        def safe_ce(logits):
            if logits.shape[0] == 0:
                return jt.float32(0.0)
            # 目标全是第0列 (因为我们将 Positive Score 放在了 concat 的第0个位置)
            targets = jt.zeros((logits.shape[0],), dtype='int32')
            return nn.cross_entropy_loss(logits, targets)

        loss_model = safe_ce(logits_model)
        loss_set = safe_ce(logits_set)
        loss_label = safe_ce(logits_label)
        loss_human = safe_ce(logits_human)

        if self.only_classifier:
            loss_label = jt.float32(0.0)
        
        # 5. 总损失加权
        loss = self.a * loss_model + \
               self.b * loss_set + \
               self.c * loss_label + \
               (self.a + self.b + self.c) * loss_human + \
               self.d * loss_classify

        # 返回值 (保持与 train_classifier.py 接收数量一致)
        # k_out 和 k_outlabel 这里直接返回 q 和 label 用于兼容
        return loss, loss_model, loss_set, loss_label, loss_classify, loss_human, q, label