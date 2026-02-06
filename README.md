# DeTeCtive-jittor

这是关于论文《DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning》的 **Jittor** 代码复现仓库。

> 📌 **重要提示**：本实验所需的数据集文件（`.zip`）存放在本仓库的 `master` 分支中。

---
## 1.环境配置
### 见requirements.txt文件

## 2.数据准备脚本
### 核心主程序：gen_database.py
### 自动化批处理脚本：script/gen_database.sh
### 特化工具类：utils/OUTFOX_utils.py 和 utils/Turing_utils.py（负责具体数据集格式的解析）

## 3.训练脚本
### 核心主程序：train_classifier.py
### 自动化批处理脚本：script/train.sh
### 核心算法实现：src/simclr.py（对比学习逻辑）及 src/jittor_roberta.py（模型迁移）

## 4.测试脚本
### 核心主程序：test_from_database.py 或 test_knn.py
### 自动化批处理脚本：script/infer.sh
### 推理主程序：infer.py

## 5.实验记录
### pytorch版训练loss
<img width="665" height="155" alt="image" src="https://github.com/user-attachments/assets/e3dfc818-8c96-4e5d-92d0-5af9385b2d52" />


### jittor版训练loss
<img width="616" height="214" alt="image" src="https://github.com/user-attachments/assets/90da4971-68d1-4a25-87af-f18e01a4c94c" />

### pytorch版Acc（左）和F1结果（右）
<img width="280" height="169" alt="image" src="https://github.com/user-attachments/assets/a70a948a-f962-4516-88a8-8ccd2248aa44" />
<img width="294" height="173" alt="image" src="https://github.com/user-attachments/assets/82d643ad-79e9-4a22-a307-803729af7093" />


### jittor版Acc（上）和F1结果（下）
<img width="672" height="215" alt="image" src="https://github.com/user-attachments/assets/23d6b5b4-5b98-4d4f-a6e5-f2c262a1c95f" />
<img width="624" height="215" alt="image" src="https://github.com/user-attachments/assets/c26c5777-0f29-4185-b941-573a9dd937c6" />



##  6.项目结构总说明

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
```

##   7.实验复现结果
### 本实验通过 Jittor 框架复现了 DeTeCtive 模型，并在 TuringBench 数据集上完成了训练和预测。

###  （1）核心性能对比
#### Accuracy (准确率): 本复现版本达到 0.9918。与原论文 0.9974 相比，差异不到 1%。

#### F1-Score (F1值): 本复现版本达到 0.9574。与原论文 PyTorch 版本的 0.9935 左右相比，差异不到 4%。

### （2）资源消耗与耗时
#### 硬件需求: 本版本仅需 单张 A800 显卡 即可运行，远低于原文所需的 10 张 A100 显卡，极大降低了复现门槛。

#### 实验时间: 在当前单卡配置下，整体流程约需 5小时。

#### 加速潜力: 若采用 A100 显卡并将 batch_size 调整为 256，总耗时可缩短至 2小时 以内。 

## 8.复现需知
### （1）路径修改: 您若想使用此仓库进行复现，请务必根据代码中的注释，修改相应的本地文件存储地址。

### （2）分支说明: 如果在主界面找不到数据集，请手动将分支切换至 master 下载对应的压缩包文件。

### （3）操作流程：按照要求和文件框架配置好环境和数据集——进入文件根目录——在终端输入"./run_a100"——等待程序运行

### （4）本次复现使用过a100和a800两张显卡，二者均能满足环境要求，如果使用a100训练时可以手动调整run_a100这一文件中的batchsize，可减少训练时间；若使用a800则直接按操作运行即可。

## 9.联系
### 若您在复现中遇见问题或有优化与思考想讨论一下，可邮箱联系我2412351@mail.nankai.edu.cn




