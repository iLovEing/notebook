# [Pytorch](https://github.com/iLovEing/notebook/issues/38)

记录一些pytorch相关的tips

---

#### 1. model.eval() & torch.no_grad()
使用pytorch在推理时经常会用到`model.eval()`和`torch.no_grad()`两个api，这两个具体的左右和异同：
###### 函数作用
`model.eval()`：将模型设置为评估模式。在评估模式下，模型的所有层都将正常运行，但不会进行反向传播和参数更新。此外，某些层的行为也会发生改变，如Dropout层将停止dropout，BatchNorm层将使用训练时得到的全局统计数据而不是评估数据集中的批统计数据。
`torch.no_grad()`：用于禁用梯度计算。在使用`torch.no_grad()`生效范围内，所有涉及张量操作的函数都将不会计算梯度，从而节省内存和计算资源。

###### 使用区别
`model.eval()`：在模型的评估阶段，确保模型的行为与预期一致时使用（比如评估时不需要dropout），此时模型不进行反向传播和参数更新。此外，`model.eval()` 必须在模型已经完成训练之后才能调用，对整个模型有效。
`torch.no_grad()`：某些情况下，只需要进行前向传播而不需要计算梯度时使用，例如在测试阶段或某些特定的预测任务中，希望模型的前向传播行为和训练一致，但又不进行反向传播。此外`torch.no_grad()`只对作用范围生效，不会影响到其他线程的计算。