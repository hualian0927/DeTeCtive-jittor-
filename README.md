# DeTeCtive-jittor

这是关于论文《DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning》的 **Jittor** 代码复现仓库。

> 📌 **重要提示**：本实验所需的数据集文件（`.zip`）存放在本仓库的 `master` 分支中。

---

## 📂 项目结构说明

```text
.
├── script/                # 自动化运行脚本
│   ├── gen_database.sh    # 生成数据库批处理脚本
│   ├── infer.sh           # 推理批处理脚本
│   └── train.sh           # 训练批处理脚本
├── src/                   # 核心源代码
│   ├── __init__.py
│   ├── dataset.py         # 数据集加载与处理逻辑
│   ├── index.py           # 索引构建相关
│   ├── jittor_roberta.py  # 基于 Jittor 框架的 RoBERTa 模型实现
│   ├── simclr.py          # SimCLR 对比学习算法实现
│   └── text_embedding.py  # 文本向量化处理
├── utils/                 # 工具函数库
│   ├── OUTFOX_utils.py    # 针对 OUTFOX 数据集的特化工具
│   └── Turing_utils.py    # 针对 TuringBench 数据集的特化工具
├── OUTFOX.zip             # [LFS] OUTFOX 原始数据集 (master 分支)
├── TuringBench.zip        # [LFS] TuringBench 原始数据集 (master 分支)
├── gen_database.py        # 数据库生成主程序
├── infer.py               # 模型推理/预测主程序
├── requirements.txt       # 项目依赖环境列表
├── test_from_database.py  # 基于数据库的性能测试脚本
├── test_knn.py            # K-近邻分类测试
├── train_classifier.py    # 分类器训练主程序
└── README.md              # 项目说明文档

##📊 实验复现结果对比
指标,本仓库复现 (单张 A800),原文结果 (10张 A100),差异
Accuracy,0.9919,0.9990,< 1.0%
F1-Score,0.9574,~0.9900,< 4.0%
