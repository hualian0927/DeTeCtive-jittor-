# DeTeCtive-jittor-
这是关于论文《DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning》的jittor代码复现仓库,本实验的所需数据集见此仓库的master分支
本仓库主要结构如下：
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
├── OUTFOX.zip             # [LFS] OUTFOX 原始数据集压缩包（在master分支内）
├── TuringBench.zip        # [LFS] TuringBench 原始数据集压缩包（在maser分支内）
├── gen_database.py        # 数据库生成主程序
├── infer.py               # 模型推理/预测主程序
├── requirements.txt       # 项目依赖环境列表
├── test_from_database.py  # 基于数据库的性能测试脚本
├── test_knn.py            # K-近邻分类测试
├── train_classifier.py    # 分类器训练主程序
└── README.md              # 项目说明文档


# DeTeCtive-jittor-
本实验通过jittor代码复现DeTeCtive模型，并且在TuringBench数据集上进行相应的训练和预测，采用单张A800显卡，比原文的10张A100所需资源消耗更低，最终复现Accuracy = 0.9919, F1 = 0.9574，Accuracy与原文的Accuracy = 0.999相差不到1%,F1与原文F1=0.99左右的pytorch版本相差不到4%（单张卡限制在模型对比上显示出劣势），但是所需资源大幅度降低，时间上由于资源限制则有一定延长，整个实验大概在5h左右，但是经过测试，若稍微提高显卡配置，采用A100显卡，将batchsize从128调整为256，则只需要2h以内，时间会大幅度降低，效果相差无几。
您若想用此jittor仓库进行复现，则需要按照代码中的注释修改相应的本地地址。
