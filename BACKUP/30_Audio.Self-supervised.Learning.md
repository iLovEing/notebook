# [Audio Self-supervised Learning](https://github.com/iLovEing/notebook/issues/30)

## Audio SSL

SSL的思想可以抽象为让模型学习对应的数据的内在空间结构和表达，SSL在audio上的效果要差于NLP和CV，这体现在：
1. 现实生活中音频的不确定性，比如人与人之间、甚至是个人的不同时期，不同情绪下说话的差异，气息、声调都有区别，录音设备的不同和摆放方式也会导致数据的差异，这使得SSL较难学到声音的潜在结构；
2. 不同噪声对音频的叠加干扰，会扭曲SSL学习的内容；
3. 音频数据本身的结构，是复杂的高维流形，比如，音频整体的横向平移，对人而言几乎没有差别，因此模型loss的设计难度较高，这点和CV类似。

根据训练中是否设置负样本，Audio SSL 训练框架可以分为两类：预测类和对比类（Predictive and contrastive），预测类模型有**Auto-encoding**、**Siamese Models**、**Clustering**等，