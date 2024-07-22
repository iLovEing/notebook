# [pytorch分布式训练 - DDP](https://github.com/iLovEing/notebook/issues/32)

# pytorch分布式训练
### 并行策略
1. 分布式训练根据并行策略的不同，可以分为模型并行和数据并行。
- 模型并行：模型并行主要应用于模型相比显存来说更大，一块 GPU 无法加载的场景，通过把模型切割为几个部分，分别加载到不同的 GPU 上，来进行训练

- 数据并行：这个是日常会应用的比较多的情况。即每个 GPU 复制一份模型，将一批样本分为多份分发到各个GPU模型并行计算。因为求导以及加和都是线性的，数据并行在数学上也有效。采用数据并行相当于加大了batch_size，得到更准确的梯度或者加速训练。

### pytorch api
1. pytorch 常用的并行训练 API 有两个：
- torch.nn.DataParallel(**DP**)
- torch.nn.DistributedDataParallel(**DDP**)

2. DP和DDP的区别：
    1. DDP 通过多进程实现，每个进程包含独立的解释器和 GIL，而 DP 通过单进程控制多线程来实现。
GIL是Python用于同步线程的工具，使得任何时刻仅有一个线程在执行，导致Python在多线程效能上表现不佳。DDP中，由于每个进程拥有独立的解释器和 GIL，消除了来自单个 Python 进程中多个执行线程，模型副本或 GPU 的额外解释器开销和 GIL-thrashing ，因此可以减少解释器和 GIL 使用冲突。对于严重依赖 Python runtime 的 models 而言，比如说包含 RNN 层或大量小组件的 models 而言，这尤为重要。
    2. DDP 每个进程对应一个独立的训练过程，且只对梯度等少量数据进行信息交换
DDP 在每次迭代中，每个进程具有自己的 optimizer ，并独立完成所有的优化步骤。在各进程梯度计算完成之后，各进程需要将梯度进行汇总平均，然后再由 rank=0 的进程，将其 broadcast 到所有进程。之后，各进程用该梯度来独立的更新参数。而 在DP中，全程维护一个 optimizer，梯度汇总到 gpu0 ，反向传播更新参数，再广播模型参数给其他的 gpu 。
DDP 由于各进程中的模型，初始参数一致 (初始时刻进行一次 broadcast)，而每次用于更新参数的梯度也一致，因此，各进程的模型参数始终保持一致。相较于 DP，DDP 传输的数据量更少，因此速度更快，效率更高。
    3. DDP 可以和模型并行一起使用，支持多机多卡，DP 则不支持多机。即使在单机训练中，DDP 还预先复制模型，而不是在每次迭代时复制模型，并避免了全局解释器锁定，仍比 DP 快。
 
> ***attention***: 这里只讨论使用 ***DDP*** 进行 ***数据并行*** 的做法。

---

2.3 cuda()函数解释
cuda() 函数返回一个存储在CUDA内存中的复制，其中device可以指定cuda设备。 但如果此storage对象早已在CUDA内存中存储，并且其所在的设备编号与cuda()函数传入的device参数一致，则不会发生复制操作，返回原对象。

cuda()函数的参数信息:

device ([int](https://docs.python.org/3/library/functions.html#int)) – 指定的GPU设备id. 默认为当前设备，即 [torch.cuda.current_device()](https://www.cntofu.com/book/169/docs/1.0/cuda.html#torch.cuda.current_device)的返回值。

non_blocking ([bool](https://docs.python.org/3/library/functions.html#bool)) – 如果此参数被设置为True, 并且此对象的资源存储在固定内存上(pinned memory)，那么此cuda()函数产生的复制将与host端的原storage对象保持同步。否则此参数不起作用。

---

# DDP 基本概念

- **node**：物理节点，可以是一个容器也可以是一台机器，节点内部可以有多个GPU；nnodes指物理节点数量， nproc_per_node指每个物理节点上面进程的数量，通常每个GPU对应一个线程，所以nproc_per_node通常是每台机器上GPU的数量。
- **local_rank**：一个node上的进程(GPU)ID，比如一台机器上有4块GPU，那么他们的local rank分别是0、1、2、3，local_rank在node之间相互独立。
- **rank**：全局进程(GPU)ID，比如两台各4块GPU，那么他们的rank分别为0~7，rank = nproc_per_node * node_rank + local_rank；默认rank 0为主进程；world size 为全局进程总数。
- **backend**： 通信后端，可选的包括：nccl（NVIDIA推出）、gloo（Facebook推出）、mpi（OpenMPI）。一般建议GPU训练选择nccl，CPU训练选择gloo。
- **master_addr**、**master_port**：主节点的地址以及端口，供init_method 的tcp方式使用。 因为pytorch中网络通信建立是从机去连接主机，运行ddp只需要指定主节点的IP与端口，其它节点的IP不需要填写。 这个两个参数可以通过环境变量或者init_method传入。

如下图所示，共有3个node(机器)，每个node上有4个GPU，每台机器上起4个进程，每个进程占一块GPU，那么图中一共有12个rank，nproc_per_node=4，nnodes=3，每个节点都一个对应的node_rank。
![image](https://github.com/user-attachments/assets/71694762-3aa5-48df-b7d4-37b72a53413b)

---

> ***attention***: rank与GPU之间没有必然的对应关系，一个rank可以包含多个GPU；一个GPU也可以为多个rank服务（多进程共享GPU），在torch的分布式训练中习惯默认一个rank对应着一个GPU，因此local_rank可以当作GPU号。