# Transformer



## 📰 项目介绍

- 本项目手工实现了完整的 **Encoder-Decoder Transformer** 架构，基于 IWSLT2017 数据集完成英语→德语的机器翻译任务。核心目标是深入理解 Transformer 的底层原理（多头注意力、位置编码、残差连接等），并验证模型在小规模序列到序列任务上的性能。

- 完整复现 Transformer 原论文核心模块（多头注意力、FFN、位置编码、残差连接 + 层归一化）

  ✅ 支持多 GPU 并行训练（适配单 / 多卡环境，已在双 RTX 4090 上验证）

  ✅ 实现带重复惩罚的束搜索解码，提升翻译流畅度

  ✅ 包含完整的数据预处理流程（词汇表构建、序列截断 / 填充、掩码生成）

  ✅ 训练日志可视化（损失曲线）+ 可复现实验配置

  ✅ 支持消融实验（可快速关闭位置编码、多头注意力等组件验证作用）

## 环境配置

```python
conda create -n transformer python=3.9.12
conda activate transformer
pip install -r requirements.txt
```

##  数据集
### 1. Data Preparation
##### 使用IWSLT2017英德翻译数据集，下载流程如下：

1.下载数据集：IWSLT2017en-de数据集

2.解压后将所有文件放入项目目录：./iwslt17_data/

3.数据集结构要求如下：

```bash
./iwslt17_data/
└── train.tags.en-de.en
├── train.tags.en-de.de
├── IWSLT17.TED.dev2010.en-de.en.xml
├── IWSLT17.TED.dev2010.en-de.de.xml
├── IWSLT17.TED.tst2010.en-de.en.xml
└── IWSLT17.TED.tst2010.en-de.de.xml
```

### 2. Training

* To train the CascadedGaze model:

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/SIDD/CascadedGaze-SIDD.yml --launcher pytorch
```


### 3. Evaluation


#### Note: Due to the file size limitation, we are not able to share the pre-trained models in this code submission. However, they will be provided with an open-source release of the code.


##### Testing the model

  * To evaluate the pre-trained model use this command:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8080 basicsr/test.py -opt ./options/test/SIDD/CascadedGaze-SIDD.yml --launcher pytorch
```

### 4. Model complexity and inference speed
* To get the parameter count, MAC, and inference speed use this command:
```
python CascadedGaze/basicsr/models/archs/CGNet_arch.py
```





# 致谢
基于 Vaswani et al. (2017)《Attention Is All You Need》原论文实现
数据集来源于 IWSLT2017 机器翻译任务官方数据集
部分实现参考 PyTorch 官方文档与开源社区最佳实践

