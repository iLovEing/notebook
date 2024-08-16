# [onnx](https://github.com/iLovEing/notebook/issues/37)

# 基础知识
> 部分原理可以参考[torchscript](https://github.com/iLovEing/notebook/issues/36)
## 基本流程：
1. 使用pytorch训练、保存模型
2. 使用torch onnx将model转化成静态图
3. python推理：使用onnx runtime
4. c++推理：使用onnx runtime C++

## onnx介绍
onnx (Open Neural Network Exchange)，一种开放式神经网络格式，支持将pytorch、tensorflow等框架的模型转化正onnx，并在onnx runtime上推理。onnx runtime支持多平台多语言，并对算子有一定优化。

## pyTorch 转 onnx 
#### 导出接口
pytorch 模型转 onnx依赖函数 `torch.onnx.export()`。重要的函数参数：
`def export(model, args, f, export_params=True, input_names=None, output_names=None, opset_version=None, dynamic_axes=None):`
- model: pytorch model 或 torchscript model
- args: 模型输入(forward)，用于静态图trace生成
- f: 保存模型的名称
- export_params: 模型中是否存储模型权重。一般中间表示包含两大类信息：模型结构和模型权重，这两类信息可以在同一个文件里存储，也可以分文件存储。ONNX 是用同一个文件表示记录模型的结构和权重的。 我们部署时一般都默认这个参数为 True。如果 onnx 文件是用来在不同框架间传递模型（比如 PyTorch 到 Tensorflow）而不是用于部署，则可以令这个参数为 False。
- input_names, output_names: 设置输入和输出张量的名称。如果不设置的话，会自动分配一些简单的名字（如数字）。 ONNX 模型的每个输入和输出张量都有一个名字。很多推理引擎在运行 ONNX 文件时，都需要以“名称-张量值”的数据对来输入数据，并根据输出张量的名称来获取输出数据。在进行跟张量有关的设置（比如添加动态维度）时，也需要知道张量的名字。 在实际的部署流水线中，我们都需要设置输入和输出张量的名称，并保证 ONNX 和推理引擎中使用同一套名称。
- opset_version: 转换时参考哪个 ONNX 算子集版本。
- dynamic_axes: 指定输入输出张量的哪些维度是动态的。 为了追求效率，ONNX 默认所有参与运算的张量都是静态的（张量的形状不发生改变）。但在实际应用中，可能希望模型的输入张量是动态的，比如没有形状限制的全卷积模型。因此，需要显式地指明输入输出张量的哪几个维度的大小是可变的。

#### 导出原理
![image](https://github.com/user-attachments/assets/b1258d09-f050-448c-b50c-c44539071d6d)
TorchScript 是一种序列化PyTorch模型的格式（静态图），在序列化过程中，一个torch.nn.Module模型会被转换成 TorchScript 的torch.jit.ScriptModule模型。torch.onnx.export中需要的模型实际上是一个torch.jit.ScriptModule，然后通过算子转换将torch.jit.ScriptModule转化为onnx model。
torch.onnx.export 支持 nn.Module 和 jit.ScriptModule 两种格式作为输入。如果使用 ScriptModule，则直接执行算子转换流程；如果使用nn.Module，则 torch.onnx.export 默认以 trace 方式导出 ScriptModule，再进行算子转换，特别地，这个过程中 @torch.jit.script 修饰同样生效。

---

```
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.relu = torch.nn.ReLU()

    @torch.jit.script
    def _abs(x):
        if x.sum() > 0:
            return x
        else:
            return -x

    # input: x-(batch, 4), h-(batch, 2)
    # output: x1-(batch, 2), x2-(batch, 2)
    def forward(self, x, h):
        x = self._abs(x)
        x = self.linear(x)
        x1 = x + h
        x2 = self.relu(x1)
        return x1, x2

model = TestModel()
model.eval()
torch.onnx.export(model,
                  (torch.randn([1, 4]), torch.randn([1, 2])),
                  'model.onnx',
                  input_names=['in_x', 'in_h'],
                  output_names=['out0', 'out1'],
                  dynamic_axes={
                      'in_x': {0: 'batch'},
                      'in_h': {0: 'batch'},
                      'out0': {0: 'batch'},
                      'out1': {0: 'batch'}
                  })
```

---

```
onnx_model0 = onnx.load("model0.onnx")
onnx.checker.check_model(onnx_model0)

ort_session0 = onnxruntime.InferenceSession("model0.onnx")
ort_inputs1 = {'in_x': a.detach().numpy(), 'in_h': b.detach().numpy()}
ort_inputs2 = {'in_x': x.detach().numpy(), 'in_h': y.detach().numpy()}
ort_output1 = ort_session0.run(['out0', 'out1'], ort_inputs1)[0]
ort_output2 = ort_session0.run(['out0', 'out1'], ort_inputs2)[0]
ort_output1.sum(), ort_output2.sum()
```