# [Audio Self-supervised Learning](https://github.com/iLovEing/notebook/issues/30)

## Audio SSL

SSL的思想可以抽象为让模型学习对应的数据的内在空间结构和表达，SSL在audio上的效果要差于NLP和CV，这体现在：
1. 现实生活中音频的不确定性，比如人与人之间、甚至是个人的不同时期，不同情绪下说话的差异，气息、声调都有区别，录音设备的不同和摆放方式也会导致数据的差异，这使得SSL较难学到声音的潜在结构；
2. 不同噪声对音频的叠加干扰，会扭曲SSL学习的内容；
3. 音频数据本身的结构，是复杂的高维流形，比如，音频整体的横向平移，对人而言几乎没有差别，因此模型loss的设计难度较高，这点和CV类似。

深度学习中，常把音频转化成频谱、fband等作为模型的初始输入，转化后数据形式和图片类似，因此很多audio SSL方法来自于成熟的Image SSL model。凭根据训练中是否设置负样本，Audio SSL 训练框架可以分为两类：预测类和对比类（Predictive and contrastive），预测类模型有**Auto-encoding**、**Siamese Models**、**Clustering**等，对比类模型多从图像领域的对比学习变体而来。

**Predictive SSL frameworks (a-c) and contrastive SSL framework (d).**
![image](https://github.com/iLovEing/notebook/assets/109459299/d2fce172-9c30-4242-b245-9a5d3bc69e2d)
> - auto-encoder包含编码器和解码器，编码器学习信号表示，解码器还原信号，对比解码器输出和原信号构造损失。
> - a Siamese network processes two views of the same data point, hence the latent representation of one sub-network is seen as pseudo-label of the other sub-network
> - clustering is applied for grouping the learnt representations – the clustering centroids are used as pseudo-labels for training;
> - contrastive SSL通过负样本构造loss。



**overview of audio SSL methords. FOS abbreviates field of study.**
![image](https://github.com/iLovEing/notebook/assets/109459299/0dd663a2-157d-4ed2-97ee-ccec2e390a9c)

