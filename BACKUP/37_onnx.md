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
import torch  # 这是同时使用PyTorch和TorchScript所需的全部导入！
import torch.nn as nn
import onnx
import onnxruntime

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
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
with torch.no_grad():
    torch.onnx.export(model=model,
                      args=(torch.randn([1, 4]), torch.randn([1, 3])),
                      f='test_onnx_model.onnx',
                      opset_version=20,
                      input_names=['in_x', 'in_h'],
                      output_names=['out0', 'out1'],
                      dynamic_axes={
                          'in_x': {0: 'batch'},
                          'in_h': {0: 'batch'},
                          'out0': {0: 'batch'},
                          'out1': {0: 'batch'}
                      })
```

**重要参数解释**（斜体为必要参数）
```
torch.onnx.export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=None, _retain_param_name=True,
           do_constant_folding=True, example_outputs=None, strip_doc_string=True,
           dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
           enable_onnx_checker=True, use_external_data_format=False)
```
- *mode*l: 传入模型，可以是`torch.nn.Module`或者`torch.jit.ScriptModule`
- *args*：tensor， 模型forward函数的输入
- *f*：保存的模型名
- export_params：模型中是否存储模型权重。一般中间表示包含两大类信息：模型结构和模型权重，这两类信息可以在同一个文件里存储，也可以分文件存储。ONNX 用同一个文件表示记录模型的结构和权重， 部署时一般都默认这个参数为 True。如果 onnx 文件是用来在不同框架间传递模型（比如 PyTorch 到 Tensorflow）而不是用于部署，则可以令这个参数为 False
- input_names, output_names：设置输入和输出张量的名称。如果不设置的话，会自动分配一些简单的名字（如数字）。 ONNX 模型的每个输入和输出张量都有一个名字。很多推理引擎在运行 ONNX 文件时，都需要以“名称-张量值”的数据对来输入数据，并根据输出张量的名称来获取输出数据。在进行跟张量有关的设置（比如添加动态维度）时，也需要知道张量的名字。 在实际的部署流水线中，需要设置输入和输出张量的名称，并保证 ONNX 和推理引擎中使用同一套名称
- opset_version：转换时参考哪个 ONNX 算子集版本
- dynamic_axes：指定输入输出张量的哪些维度是动态的。 为了追求效率，ONNX 默认所有参与运算的张量都是静态的（张量的形状不发生改变）。但在实际应用中，模型输入张量的某些维度是动态的，尤其是本来就没有形状限制的全卷积模型。此时，需要显式地指明输入输出张量的哪几个维度的大小是可变的。

---

## Sample Code 2 - python推理
```
onnx_model = onnx.load("test_onnx_model.onnx")
onnx.checker.check_model(onnx_model)

x, h = torch.randn([1, 4]), torch.randn([1, 3])
ort_session = onnxruntime.InferenceSession("test_onnx_model.onnx")
ort_inputs = {'in_x': x.detach().numpy(), 'in_h': h.detach().numpy()}
ort_output = ort_session.run(['out0', 'out1'], ort_inputs)
print(ort_output[0], ort_output[1])
```

---

# Sample Code 3 - C++推理
## Step 1: 配置c++环境
1. 安装make、cmake、gcc等工具
sudo apt install cmake
sudo apt install make
sudo apt install gcc
2. onnxruntime库安装，在[官方release](https://github.com/microsoft/onnxruntime/releases)选择适合的版本下载，下载完成后解压并记下目录，也可以自己根据build pipeline构建，这里使用直接下载的lib。

## step 2：编写c++代码
- CMakeLists.txt
```
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
project(test_onnx)

find_package(onnxruntime REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
get_target_property(ONNXRT_INC onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)

message( STATUS "build type ${CMAKE_BUILD_TYPE}" )
message(STATUS "onnxruntime include dir: ${ONNXRT_INC}")

add_executable(test_onnx main.cpp)
target_link_libraries(test_onnx onnxruntime::onnxruntime)
```

- main.cpp
```
#include <iostream>
#include <onnxruntime_cxx_api.h>


void prinf_result(const std::vector<std::vector<float>>& result, std::string str)
{
    std::cout << str << std::endl;
    for (int i = 0; i < result.size(); ++i) {
        std::cout << "node " << i << " :(";
        for (const auto& _n: result[i]) {
            std::cout << " " << _n;
        }
        std::cout << " )" << std::endl;
    }
}

int main(int argc, char** argv)
{
    const int64_t BATCH_SIZE = 2;
    std::cout << "Hello onnxruntime!" << std::endl;

    Ort::Env env;
    Ort::Session model(nullptr);
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (argc < 2) {
        std::cout << "no model name or path input, return." << std::endl;
        return 0;
    }

    // load model
    std::string model_path = argv[1];
    if (model_path[0] != '/') {
        model_path = "../model/" + model_path;
    }
    std::cout << "\nload model: " << model_path << std::endl;
    try {
        model = Ort::Session(env, model_path.data(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "error loading the model, what(): " << e.what() << std::endl;
        return 0;
    }

    // get model info
    size_t num_input_nodes = model.GetInputCount();
    size_t num_output_nodes = model.GetOutputCount();
    // input info
    std::cout << "\nread model input info:\n" << "num of input nodes: " << num_input_nodes <<std::endl;
    for (int i = 0; i < num_input_nodes; i++) {
        // 得到输入节点的名称 std::string
        Ort::AllocatedStringPtr node_name = model.GetInputNameAllocated(i, allocator);
        std::cout << "node " << i << ": " << "name " << node_name.get();

        Ort::TypeInfo type_info = model.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        // 输入节点的数据类型 ONNXTensorElementDataType
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << ", type " << type;

        // 输入节点的输入维度 std::vector<int64_t>，这里有-1是因为模型有dynamic_axes
        std::vector<int64_t> node_shape = tensor_info.GetShape();
        std::cout << ", shape: (";
        for (const auto& _d: node_shape) {
            std::cout << " " << _d;
        }
        std::cout << " )" << std::endl;
    }
    // output info
    std::cout << "\nread model output info:\n" << "num of output nodes: " << num_output_nodes <<std::endl;
    for (int i = 0; i < num_output_nodes; i++) {
        // 得到输入节点的名称 std::string
        Ort::AllocatedStringPtr node_name = model.GetOutputNameAllocated(i, allocator);
        std::cout << "node " << i << ": " << "name: " << node_name.get();

        Ort::TypeInfo type_info = model.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        // 输出节点的数据类型 ONNXTensorElementDataType
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << ", type " << type;

        // 输出节点的输入维度 std::vector<int64_t>，这里有-1是因为模型有dynamic_axes
        std::vector<int64_t> node_shape = tensor_info.GetShape();
        std::cout << ", shape: (";
        for (const auto& _d: node_shape) {
            std::cout << " " << _d;
        }
        std::cout << " )" << std::endl;
    }

    // 输入
    //vector
    std::vector<std::vector<float>> ori_inputs;
    std::vector<float> input_node_1(BATCH_SIZE * 4, 0);
    std::vector<float> input_node_2(BATCH_SIZE * 3, 0);
    ori_inputs.emplace_back(input_node_1);
    ori_inputs.emplace_back(input_node_2);
    std::vector<std::vector<int64_t>> ori_inputs_shape;
    std::vector<int64_t> input_node_shape_1{BATCH_SIZE, 4};
    std::vector<int64_t> input_node_shape_2{BATCH_SIZE, 3};
    ori_inputs_shape.emplace_back(input_node_shape_1);
    ori_inputs_shape.emplace_back(input_node_shape_2);
    srand((unsigned)time(NULL));
    for (auto& _in_vec: ori_inputs) {
        for (int i = 0; i < _in_vec.size(); ++i) {
            _in_vec[i] = float(rand() % 100) / 50.0 - 1.0;
        }
    }
    // tensor
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

    // 输出
    // vector
    std::vector<std::vector<float>> result;
    std::vector<float> output_node_1(BATCH_SIZE * 3, 0);
    std::vector<float> output_node_2(BATCH_SIZE * 3, 0);
    result.emplace_back(output_node_1);
    result.emplace_back(output_node_2);
    std::vector<std::vector<int64_t>> result_shape;
    std::vector<int64_t> output_node_shape_1{BATCH_SIZE, 3};
    std::vector<int64_t> output_node_shape_2{BATCH_SIZE, 3};
    result_shape.emplace_back(output_node_shape_1);
    result_shape.emplace_back(output_node_shape_2);
    // tensor, 包装vector时使用的是地址，因此推理后result的值就是结果
    std::vector<Ort::Value> ort_outputs;
    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  result[i].data(),
                                                                  result[i].size(),
                                                                  result_shape[i].data(),
                                                                  result_shape[i].size());
        ort_outputs.push_back(std::move(output_tensor));
    }

    prinf_result(result, "\nbefore infer:");
    // 推理
    const char* input_names[] = {"in_x", "in_h"};
    const char* output_names[] = {"out0", "out1"};
    model.Run(Ort::RunOptions{ nullptr },
                input_names,
                ort_inputs.data(),
                num_input_nodes,
                output_names,
                ort_outputs.data(),
                num_output_nodes);

    // 这里所有batch数据在一个vecror内连续排列
    prinf_result(result, "\nafter infer:");
    return 0;
}
```

## step 3：编译运行
1. 代码目录结构：
```
- CMakeLists.txt
- main.cpp
- model/
  - test_onnx_model.onnx
- bin
- build
```

2. 在编译之前
在下载 *onnxruntime-linux-x64-1.19.0.tgz* 时，发现此版本对 linux 系统的三方配置很不友好，源码提供了 cmake package和 pkg-config 两种配置方案，但都或多或少有 bug。总的来说 onnxruntime lib 配置方案有三种：
    1. 直接使用 `include_directories()` 声明头文件位置，用 `link_directories()` 或 `CMAKE_PREFIX_PATH` 声明库位置，link时库名称为 onnxruntime
    2. 使用 pkg-config 方式，pc文件位于 `${onnxruntime dir}/lib/pkgconfig` 下，但是里面并没有很好的定义头文件和库的位置，而是需要把对应文件放到系统目录下，很是麻烦。
    3. 使用cmake find_package 方式，cmake文件位于 `${onnxruntime dir}/lib/cmake` 下，**这里使用该方案**，link的库名为 onnxruntime::onnxruntime，但是编译时会发现两个bug：
        - 64位系统下的lib路径为lib64，但是包里没有。解决方案：建立lib64目录指向lib的软链接
```
CMake Error at /home/tlzn/users/zlqiu/libs/onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake:84 (message):
  The imported target "onnxruntime::onnxruntime" references the file

     "/home/tlzn/users/zlqiu/libs/onnxruntime/lib64/libonnxruntime.so.1.19.0"

  but this file does not exist.  Possible reasons include:
```
        - onnxruntimeTargets.cmake 文件中 include dir 定义有问题，多了onnxruntime。解决方案：删除onnxruntime，到include就行
```
CMake Error in CMakeLists.txt:
  Imported target "onnxruntime::onnxruntime" includes non-existent path

    "/home/tlzn/users/zlqiu/libs/onnxruntime/include/onnxruntime"

  in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
```

3. 编译
设置环境变量：export onnxruntime_DIR=[your onnxruntimedir]/lib/cmake/onnxruntime

    > mkdir build
    > cd build
    > cmake ..
    > make

4. 运行
    > ../bin/test_onnx test_onnx_model.onnx

---

# other
## 1. 一些有用的网站
- [ONNX Runtime c++ doc](https://onnxruntime.ai/docs/api/c/index.html)
- [ONNX github](https://github.com/onnx/onnx)
- [ONNX Runtime github](https://github.com/microsoft/onnxruntime)
- [ONNX infer examples](https://github.com/microsoft/onnxruntime-inference-examples)
- [MMDeploy guide](https://mmdeploy.readthedocs.io/zh-cn/v1.2.0/tutorial/01_introduction_to_model_deployment.html)

## 2. onnx可视化
onnx模型可以用[netron](https://netron.app/)可视化，如图
![image](https://github.com/user-attachments/assets/23a67136-1774-493f-b381-6ba17edb3f7d)

## 3. onnx模型优化
onnx模型可以用simplify进行优化，其中会进行onnx自有的图优化和算子优化，不会改变计算结果，可能会加速模型推理。
```
from onnxsim import simplify

model = onnx.load("model.onnx")
model_simp, check = simplify(model)
onnx.save(model_simp, "model_sim.onnx")
```

## 3. onnx模型量化
- **动态量化**
1. （建议）使用`simplify`对模型进行优化
2. （建议）对模型进行前处理：
    >  python -m onnxruntime.quantization.preprocess --input model_sim.onnx --output model_sim_pre.onnx
3. 使用`quantize_dynamic`量化模型，一般使用int8或者uint8量化，加速效果随平台和模型而异。
```
from onnxruntime.quantization import QuantType, quantize_dynamic 

model_fp32 = "model_sim_pre.onnx"
model_quant_dynamic = "model_sim_pre_quant_u8.onnx"
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant_dynamic,
    weight_type=QuantType.QUInt8,  # or QuantType.QInt8
)
```

- **静态量化**

---

# 有效问题记录