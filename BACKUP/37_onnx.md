# [onnx](https://github.com/iLovEing/notebook/issues/37)

# 基础知识
## 基本流程：
1. 使用pytorch训练、保存模型
2. 使用torch onnx将model转化成静态图
3. python推理：使用onnx runtime
4. c++推理：使用onnx runtime C++

## onnx介绍
onnx (Open Neural Network Exchange)，一种开放式神经网络格式，支持将pytorch、tensorflow等框架的模型转化正onnx，并在onnx runtime上推理。onnx runtime支持多平台多语言，并对算子有一定优化。