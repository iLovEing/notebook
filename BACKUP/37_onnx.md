# [onnx](https://github.com/iLovEing/notebook/issues/37)

# 基础知识
## 基本流程：
1. 使用pytorch训练、保存模型
2. 使用torch onnx将model转化成静态图
3. python推理：使用onnx runtime
4. c++推理：使用onnx runtime C++

## onnx介绍
onnx (Open Neural Network Exchange)，一种开放式神经网络格式，支持将pytorch、tensorflow等框架的模型转化正onnx，并在onnx runtime上推理。onnx runtime支持多平台多语言，并对算子有一定优化。

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