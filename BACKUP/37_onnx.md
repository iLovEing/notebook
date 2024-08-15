# [onnx](https://github.com/iLovEing/notebook/issues/37)

# 基础知识
## 基本流程：
1. 使用pytorch训练、保存模型
2. 使用torchscript将model转化成静态图
3. python推理：使用torch.jit
4. c++推理：使用libtorch

## 计算图
模型推理、训练（梯度传播）的底层结构是计算图。关于计算图的介绍详见 [机器学习系统：设计与实现](https://openmlsys.github.io/chapter_computational_graph/index.html)。这里做简单的介绍。

计算图分为静态图的动态图：

**静态图**：根据前端语言描述的神经网络拓扑结构以及参数变量等信息构建一份固定的计算图，再放入计算后端执行正向推理或反向传播。因此静态图在执行期间可以不依赖前端语言描述，常用于神经网络模型的部署，比如移动端人脸识别场景中的应用等。
![image](https://github.com/user-attachments/assets/a1877fcb-16bc-465d-9c73-30c79c73a942)

**动态图**：在每一次执行神经网络模型依据前端语言描述动态生成一份临时的计算图，其正向推理和反向传播依赖于框架的算子。这意味着计算图的动态生成过程灵活可变，该特性有助于在神经网络结构调整阶段提高效率。
![image](https://github.com/user-attachments/assets/e7e37366-5ff5-47aa-b249-48b2fdae78f0)

一般来说，动态图容易获取中间结果，方便调试，但是内存开销大一些不可直接部署；静态图的特点则相反。另一方面，动态图允许使用原生编程语言做分支控制，而静态图需要使用计算框架来构建（比如*tf.cond()*）。综上，训练模型、调试的时候一般使用静态图；部署时，尤其是端侧部署，一般使用动态图。
主流机器学习框架TensorFlow、MindSpore均支持动态图和静态图模式。***TorchScript***就是PyTorch则可以通过工具将构建的动态图神经网络模型转化为静态结构，以获得高效的计算执行效率。

## 静态图转换
`torch.jit.trace` 和 `torch.jit.script` 是TorchScript保存静态图的两种方式：
### 1. `torch.jit.trace(model, example_inputs)`
- 原理：使用跟踪的方法保存，原理是给模型输入一个example tensor，框架会根据这个tensor在在模型内经过的计算流程保存静态图。
- 优势：只要输入正确，模型可以正常计算，trace就可以work。
- 缺点：只能跟踪某一个tensor的计算过程，如果模型中出现分支，该输入只经过其中一个分支，则分支的其他选择无法记录

### 2.  `torch.jit.script(model)`
- 原理：框架分析模型代码，自动生成。
- 优势：可以兼容分支代码。
- 缺点：兼容性差，不是所有python原生语言都能“翻译”成功，只能处理常见的数据结构。