# [libtorch(c++)](https://github.com/iLovEing/notebook/issues/36)

## sample code of libtorch
基本流程：
pytorch -> torchscript -> libtorch

---

## step 1
生成jit model，注意这里forward函数是多输入多输出类型
```
import torch
import torch.nn as nn

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
script_model = torch.jit.trace(model, (torch.rand([6, 4]), torch.rand([6, 2])))
print(script_model.code)
script_model.save('test_jit_model.pt')
```

---

## step 2
配置libtorch环境

1. 安装make、cmake、gcc等工具
    - sudo apt install cmake
    - sudo apt install make
    - sudo apt install gcc
2. libtorch库安装，在[官网](https://pytorch.org/get-started/locally/)选择适合的版本下载，下载完成后解压。

---

## step 3
编写c++代码。
- CMakeLists.txt
```
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

project(test_libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(test_libtorch main.cpp)
target_link_libraries(test_libtorch "${TORCH_LIBRARIES}")
```

- main.cpp
```
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


int main()
{
    torch::Tensor test_tensor = torch::rand({2, 3});
    std::cout << test_tensor << std::endl;

    std::string path = "/home/tlzn/users/zlqiu/project/ctorch_test/poc.pt";
    std::cout << path << std::endl;
    torch::jit::script::Module module;

    try {
        module = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    torch::Tensor input = torch::randn({1, 975359});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto outputs = module.forward(inputs).toTuple();
    torch::Tensor output = outputs->elements()[0].toTensor();
    std::cout << output << '\n';

    return 0;
}
```