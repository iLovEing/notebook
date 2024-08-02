# [TorchScript](https://github.com/iLovEing/notebook/issues/35)

# TorchScript
[torch script tutorials](https://pytorch.org/docs/stable/jit.html)

---

## 计算图
模型推理、训练（梯度传播）的底层结构是计算图。关于计算图的介绍详见 [机器学习系统：设计与实现](https://openmlsys.github.io/chapter_computational_graph/index.html)。这里做简单的介绍。

计算图分为静态图的动态图：

**静态图**：根据前端语言描述的神经网络拓扑结构以及参数变量等信息构建一份固定的计算图，再放入计算后端执行正向推理或反向传播。因此静态图在执行期间可以不依赖前端语言描述，常用于神经网络模型的部署，比如移动端人脸识别场景中的应用等。
![image](https://github.com/user-attachments/assets/a1877fcb-16bc-465d-9c73-30c79c73a942)

**动态图**：在每一次执行神经网络模型依据前端语言描述动态生成一份临时的计算图，其正向推理和反向传播依赖于框架的算子。这意味着计算图的动态生成过程灵活可变，该特性有助于在神经网络结构调整阶段提高效率。
![image](https://github.com/user-attachments/assets/e7e37366-5ff5-47aa-b249-48b2fdae78f0)

一般来说，动态图容易获取中间结果，方便调试，但是内存开销大一些不可直接部署；静态图的特点则相反。另一方面，动态图允许使用原生编程语言做分支控制，而静态图需要使用计算框架来构建（比如*tf.cond()*）。综上，训练模型、调试的时候一般使用静态图；部署时，尤其是端侧部署，一般使用动态图。
主流机器学习框架TensorFlow、MindSpore均支持动态图和静态图模式。***TorchScript***就是PyTorch则可以通过工具将构建的动态图神经网络模型转化为静态结构，以获得高效的计算执行效率。

---

## torch.jit.trace / torch.jit.script
*torch.jit.trace* 和 *torch.jit.script* 是TorchScript保存静态图的两种方式：
### 1. `torch.jit.trace(model, example_inputs)`
- 原理：使用跟踪的方法保存，原理是给模型输入一个example tensor，框架会根据这个tensor在在模型内经过的计算流程保存静态图。
- 优势：只要输入正确，模型可以正常计算，trace就可以work。
- 缺点：只能跟踪某一个tensor的计算过程，如果模型中出现分支，该输入只经过其中一个分支，则分支的其他选择无法记录

### 2.  `torch.jit.script(model)`
- 原理：框架分析模型代码，自动生成。
- 优势：可以兼容分支代码。
- 缺点：兼容性差，不是所有python原生语言都能“翻译”成功，只能处理常见的数据结构。

### 3. 建议
建议优先选择 `torch.jit.trace` ，把有分支的code剥离成函数单独使用 `@torch.jit.script_method` 修饰，或抽离成 `nn.module` 类单独用 `torch.jit.script` 包装。
在实际使用trace的时候，有分支的地方都会报warning
![image](https://github.com/user-attachments/assets/4a15319a-0a47-46cb-a387-b6eba9ee1299)


---

## case
- 转化成 JIT
```
class TestScript(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.relu = torch.nn.ReLU()

    @torch.jit.script
    def my_abs(x):
        if x.sum() > 0:
            return x
        else:
            return -x

    def forward(self, x, h):
        x = self.my_abs(x)
        x = self.linear(x)
        x = x + h
        x = self.relu(x)
        return x

model = TestScript()
traced_model = torch.jit.trace(model, (torch.rand(3, 4), torch.rand(3, 2)))
print(traced_model.code)
print(traced_model(torch.randn(3, 4), torch.randn(3, 2)))
```

- 保存
```
traced_model.save('jit.pt')
```

- 加（注意，静态图的加载不需要类实现）
```
loaded_model = torch.jit.load('jit.pt')
print(loaded_model.code)
print(loaded_model((torch.rand(3, 4), torch.rand(3, 2)))
```