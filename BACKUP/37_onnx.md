# [onnx](https://github.com/iLovEing/notebook/issues/37)

## 基础知识
> 部分原理可以参考[torchscript](https://github.com/iLovEing/notebook/issues/36)
### 1. 基本流程
1. 使用pytorch训练、保存模型
2. 使用torch onnx将model转化成静态图
3. python推理：使用onnx runtime
4. c++推理：使用onnx runtime C++

### 2. onnx介绍
onnx (Open Neural Network Exchange)，一种开放式神经网络格式，支持将pytorch、tensorflow等框架的模型转化正onnx，并在onnx runtime上推理。onnx runtime支持多平台多语言，并对算子有一定优化。

### 3. pytorch 转 onnx 
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
TorchScript 是一种序列化PyTorch模型的格式（静态图），在序列化过程中，一个`torch.nn.Module`模型会被转换成 TorchScript 的`torch.jit.ScriptModule`模型。`torch.onnx.export`中需要的模型实际上是一个ScriptModule，然后通过算子转换将 ScriptModule 转化为 onnx model。
`torch.onnx.export`支持 nn.Module 和 jit.ScriptModule 两种格式作为输入。如果使用 ScriptModule，则直接执行算子转换流程；如果使用nn.Module，则`torch.onnx.export`默认以 trace 方式导出 ScriptModule，再进行算子转换，特别地，这个过程中`@torch.jit.script`修饰同样生效。

---

## Sample Code 1 - 导出onnx模型
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

## Sample Code 2 - python推理
```
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model0)

ort_session = onnxruntime.InferenceSession("model.onnx")
ort_inputs1 = {'in_x': a.detach().numpy(), 'in_h': b.detach().numpy()}
ort_inputs2 = {'in_x': x.detach().numpy(), 'in_h': y.detach().numpy()}
ort_output1 = ort_session0.run(['out0', 'out1'], ort_inputs1)[0]
ort_output2 = ort_session0.run(['out0', 'out1'], ort_inputs2)[0]
ort_output1.sum(), ort_output2.sum()
```

---

## Sample Code 3 - C++推理
cmake
```
cmake_minimum_required(VERSION 3.20)  # cmake version
set(CMAKE_CXX_STANDARD 17)  # c++ standard

project(ONNX_TEST)  # project name

include_directories(${PROJECT_SOURCE_DIR}/inc)  # add head file dir
find_library (libshare onnxruntime)
add_executable(onnx_test test.cpp)  # build executable
target_link_libraries(onnx_test ${libshare})  # link after build executable
```

main
```
#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <cstdlib>
#include <time.h>
#include <typeinfo>

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

void prinf_result(const std::vector<std::vector<float>>& result)
{
    for (int i = 0; i < result.size(); ++i) {
        std::cout << "node" << i << " :(";
        for (const auto& _n: result[i]) {
            std::cout << " " << _n;
        }
        std::cout << " )" << std::endl;
    }
}

int main()
{
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    session_options.SetIntraOpNumThreads(1);
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const char* model_path = "/home/tlzn/users/zlqiu/project/onnx_test/model.onnx";
    Ort::Session session(env, model_path, session_options);
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::cout << "num args of input and output: " <<
        num_input_nodes << ", " << num_output_nodes << std::endl;

    const int64_t BATCH_SIZE = 2;

    // input info
    for (int i = 0; i < num_input_nodes; i++) {
        // 得到输入节点的名称 std::string
        Ort::AllocatedStringPtr node_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "input info of node " << i << ": " << std::endl;
        std::cout << "    name: " << node_name.get() << std::endl;

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        // 输入节点的数据类型 ONNXTensorElementDataType
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "    type: " << type << std::endl;

        // 输入节点的输入维度 std::vector<int64_t>，这里有-1是因为模型有dynamic_axes
        std::vector<int64_t> node_shape = tensor_info.GetShape();
        std::cout << "    shape: (";
        for (const auto& _d: node_shape) {
            std::cout << " " << _d;
        }
        std::cout << " )" << std::endl;
    }

    // output info
    for (int i = 0; i < num_output_nodes; i++) {
        // 得到输入节点的名称 std::string
        Ort::AllocatedStringPtr node_name = session.GetOutputNameAllocated(i, allocator);
        std::cout << "output info of node " << i << ": " << std::endl;
        std::cout << "    name: " << node_name.get() << std::endl;

        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        // 输出节点的数据类型 ONNXTensorElementDataType
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "    type: " << type << std::endl;

        // 输出节点的输入维度 std::vector<int64_t>，这里有-1是因为模型有dynamic_axes
        std::vector<int64_t> node_shape = tensor_info.GetShape();
        std::cout << "    shape: (";
        for (const auto& _d: node_shape) {
            std::cout << " " << _d;
        }
        std::cout << " )" << std::endl;
    }

    // vector输入
    std::vector<std::vector<float>> ori_inputs;
    std::vector<float> input_node_1(BATCH_SIZE * 4, 0);
    std::vector<float> input_node_2(BATCH_SIZE * 2, 0);
    ori_inputs.emplace_back(input_node_1);
    ori_inputs.emplace_back(input_node_2);
    std::vector<std::vector<int64_t>> ori_inputs_shape;
    std::vector<int64_t> input_node_shape_1{BATCH_SIZE, 4};
    std::vector<int64_t> input_node_shape_2{BATCH_SIZE, 2};
    ori_inputs_shape.emplace_back(input_node_shape_1);
    ori_inputs_shape.emplace_back(input_node_shape_2);

    srand((unsigned)time(NULL));
    for (auto& _in_vec: ori_inputs) {
        for (int i = 0; i < _in_vec.size(); ++i) {
            _in_vec[i] = float(rand() % 100) / 50.0 - 1.0;
        }
    }

    // vector输出
    std::vector<std::vector<float>> result;
    std::vector<float> output_node_1(BATCH_SIZE * 2, 0);
    std::vector<float> output_node_2(BATCH_SIZE * 2, 0);
    result.emplace_back(output_node_1);
    result.emplace_back(output_node_2);
    std::vector<std::vector<int64_t>> result_shape;
    std::vector<int64_t> output_node_shape_1{BATCH_SIZE, 2};
    std::vector<int64_t> output_node_shape_2{BATCH_SIZE, 2};
    result_shape.emplace_back(output_node_shape_1);
    result_shape.emplace_back(output_node_shape_2);

    // tensor 输入
    std::vector<Ort::Value> ort_inputs;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  ori_inputs[i].data(),
                                                                  ori_inputs[i].size(),
                                                                  ori_inputs_shape[i].data(),
                                                                  ori_inputs_shape[i].size());
        ort_inputs.push_back(std::move(input_tensor));
    }

    // tensor 输出
    std::vector<Ort::Value> ort_outputs;
    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  result[i].data(),
                                                                  result[i].size(),
                                                                  result_shape[i].data(),
                                                                  result_shape[i].size());
        ort_outputs.push_back(std::move(output_tensor));
    }

    prinf_result(result);
    // 推理
    const char* input_names[] = {"in_x", "in_h"};
    const char* output_names[] = {"out0", "out1"};
    session.Run(Ort::RunOptions{ nullptr },
                input_names,
                ort_inputs.data(),
                num_input_nodes,
                output_names,
                ort_outputs.data(),
                num_output_nodes);

/*
    // 获取输出
    float* output0 = output_tensors[0].GetTensorMutableData<float>();
    float* output1 = output_tensors[1].GetTensorMutableData<float>();
*/
        prinf_result(result);
    return 0;
}

```

---

onnx可视化
https://netron.app/