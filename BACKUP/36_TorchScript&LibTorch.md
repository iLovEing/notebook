# [TorchScript&LibTorch](https://github.com/iLovEing/notebook/issues/36)

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

---

# Sample Code - 保存静态图
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
jit_model = torch.jit.trace(model, (torch.rand(3, 4), torch.rand(3, 2)))
print(jit_model.code)
print(jit_model(torch.randn(3, 4), torch.randn(3, 2)))
jit_model.save('test_jit_model.pt')
```

---

# Sample Code - python 推理
python 推理相对简单，在torch框架下进行推理即可。注意这里不再需要模型实现的代码，直接读取“图模型”进行推理即可。
```
loaded_jit_model = torch.jit.load('test_jit_model.pt')
print(loaded_jit_model.code)
print(loaded_jit_model((torch.rand(3, 4), torch.rand(3, 2)))
```

---

# Sample Code - C++推理
## step 1：配置libtorch环境

1. 安装make、cmake、gcc等工具
    - sudo apt install cmake
    - sudo apt install make
    - sudo apt install gcc
2. libtorch库安装，在[官网](https://pytorch.org/get-started/locally/)选择适合的版本下载，下载完成后解压并记下目录。

## step 2：编写c++代码。
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

## step 3：编译运行
代码目录结构：
- 方案一
设置环境变量：export Torch_DIR=[your libtorch dir]/share/cmake/Torch
```
mkdir build
cd build
cmake ..
make
```
- 方案二
cmake中添加头文件路径：
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=[your libtorch dir]
make
```


---

## step 4
编译
把test_jit_model.pt放到同一文件夹下
- 方案一
设置环境变量：export Torch_DIR=[your libtorch dir]/share/cmake/Torch
```
mkdir build
cd build
cmake ..
make
```
- 方案二
cmake中添加头文件路径：
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=[your libtorch dir]
make
```