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

# DDP 基本概念

- **node**：物理节点，可以是一个容器也可以是一台机器，节点内部可以有多个GPU；nnodes指物理节点数量， nproc_per_node指每个物理节点上面进程的数量，通常每个GPU对应一个线程，所以nproc_per_node通常是每台机器上GPU的数量。
- **local_rank**：一个node上的进程(GPU)ID，比如一台机器上有4块GPU，那么他们的local rank分别是0、1、2、3，local_rank在node之间相互独立。
- **rank**：全局进程(GPU)ID，比如两台各4块GPU，那么他们的rank分别为0~7，rank = nproc_per_node * node_rank + local_rank；默认rank 0为主进程；world size 为全局进程总数。
- **backend**： 通信后端，可选的包括：nccl（NVIDIA推出）、gloo（Facebook推出）、mpi（OpenMPI）。一般建议GPU训练选择nccl，CPU训练选择gloo。
- **master_addr**、**master_port**：主节点的地址以及端口，供init_method 的tcp方式使用。 因为pytorch中网络通信建立是从机去连接主机，运行ddp只需要指定主节点的IP与端口，其它节点的IP不需要填写。 这个两个参数可以通过环境变量或者init_method传入。
- **group**： 即进程组。默认情况下，只有一个组。一个 job 为一个组，即一个 world。

如下图所示，共有3个node(机器)，每个node上有4个GPU，每台机器上起4个进程，每个进程占一块GPU，那么图中一共有12个rank，nproc_per_node=4，nnodes=3，每个节点都一个对应的node_rank。
![image](https://github.com/user-attachments/assets/71694762-3aa5-48df-b7d4-37b72a53413b)

---

> ***attention***: rank与GPU之间没有必然的对应关系，一个rank可以包含多个GPU；一个GPU也可以为多个rank服务（多进程共享GPU），在torch的分布式训练中习惯默认一个rank对应着一个GPU，因此local_rank可以当作GPU号。在后续的表述中，默认一个进程对应一个GPU。

---

# DDP 基本流程 & 代码

### 训练流程

使用DDP分布式训练，一共如下几个步骤：

1. 初始化：使用dist.init_process_group初始化进程组
2. 封装数据：设置分布式采样器 DistributedSampler 对数据进行自动分割
3. 封装模型：使用DistributedDataParallel封装模型
4. 每次epoch之前对sampler使用set epoch；如有自定义的数据需要在多卡之间通信，比如eval时计算acc，需要手动all_gather/all_reduce
5. 使用torchrun 或者 mp.spawn 启动分布式训练

### 代码

torchrun 启动方式优雅一些，这里只展示torchrun代码。通过一个MNIST训练实例来看ddp如何实现，方便起见，train和eval用相同的数据。

- **without DDP**
  保存代码为 mnist.py，命令行输入 python mnist.py 可直接运行
  ```
  import os
  import argparse
  from datetime import datetime
  import torch
  import torch.nn as nn
  import torchvision
  from tqdm import tqdm
  import torchvision.transforms as transforms
  from torch.utils.data import Dataset, DataLoader
  
  
  class ConvNet(nn.Module):
      def __init__(self, num_classes=10):
          super(ConvNet, self).__init__()
          self.layer1 = nn.Sequential(
              nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2))
          self.layer2 = nn.Sequential(
              nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(32),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2))
          self.fc = nn.Linear(7 * 7 * 32, num_classes)
  
      def forward(self, x):
          out = self.layer1(x)
          out = self.layer2(out)
          out = out.reshape(out.size(0), -1)
          out = self.fc(out)
          return out
  
  
  def train(args):
      # 0. preset
      batch_size = args.batch_size
      epochs = args.epochs
  
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
      # 1. model
      model = ConvNet()
      model = model.to(device)
  
      # 2. dataset
      train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                             transform=transforms.ToTensor(), download=True)
      train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                shuffle=True, pin_memory=True)
      eval_set = torchvision.datasets.MNIST(root='./data', train=False,
                                            transform=transforms.ToTensor(), download=True)
      eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size,
                               shuffle=True, pin_memory=True)
  
      # 3. loss and optimizer
      criterion = nn.CrossEntropyLoss().to(device)
      optimizer = torch.optim.SGD(model.parameters(), 1e-4)
  
      # 4. training & eval
      start = datetime.now()
      step = 0
      for epoch in range(epochs):
          # train
          model.train()
          progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} train')
          for batch in progress_bar_train:
              image, label = batch
              image = image.to(device, non_blocking=True)
              label = label.to(device, non_blocking=True)
              # Forward pass
              outputs = model(image)
              loss = criterion(outputs, label)
  
              # Backward and optimize
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              if step % 10 == 0:
                  progress_bar_train.set_postfix(loss=loss.item())
              step += 1
  
          with torch.no_grad():
              model.eval()
              preds = []
              targets = []
              progress_bar_eval = tqdm(eval_loader, desc=f'Epoch {epoch + 1}/{epochs} eval')
              for batch in progress_bar_eval:
                  image, label = batch
                  image = image.to(device, non_blocking=True)
                  label = label.to(device, non_blocking=True)
                  output = model(image)
                  pred = output.argmax(dim=1)
  
                  preds.append(pred)
                  targets.append(label)
              preds = torch.cat(preds)
              targets = torch.cat(targets)
              acc = (preds == targets).sum().cpu().numpy() / len(preds)
              print(f'Epoch {epoch + 1}/{epochs} acc: {format(acc, ".4f")}')
  
      print("Training complete in: " + str(datetime.now() - start))
      # 5. save
      torch.save(model.state_dict(), 'mnist.pth')
  
  
  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument('--epochs', default=10, type=int)
      parser.add_argument('--batch_size', default=128, type=int)
      args = parser.parse_args()
      train(args)
  
  
  # run: python mnist.py
  if __name__ == '__main__':
      main()
  ```

- **add DDP code (torchrun)**
保存代码为 ddp_mnist.py，使用torch run运行：
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr="192.168.1.250" --master_port=23456 ddp_mnist.py
```
import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
########## DDP ##########
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
###########################


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


########## DDP-6 ##########
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
###########################


def train(args):
    # 0. preset
    batch_size = args.batch_size
    epochs = args.epochs

    ########## DDP-1 ##########
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    device = torch.device(f"cuda:{gpu}")  # or torch.cuda.set_device(gpu)
    print(f'pid: {os.getpid()}, ppid: {os.getppid()}, gpu: {gpu}-{rank}')
    ###########################

    # 1. model
    model = ConvNet()
    model = model.to(device)  # or mode.cuda()
    ########## DDP-2 ##########
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # for batch normal layer
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    ###########################

    # 2. dataset
    ########## DDP-3 ##########
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transforms.ToTensor(), download=True)
    # train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
    #                           shuffle=True, pin_memory=True)
    # dataset will be divided into {world_size} parts automatic by sampler,
    # every rank will obtain its own dataset
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=False,  # sampler will do shuffle
                              sampler=train_sampler)

    eval_set = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform=transforms.ToTensor(), download=True)
    # eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size,
    #                          shuffle=True, pin_memory=True)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=rank)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, sampler=eval_sampler)
    #########################

    # 3. loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # 4. training & eval
    start = datetime.now()
    step = 0
    for epoch in range(epochs):
        ########## DDP-3 ##########
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        ###########################

        # train
        model.train()
        ########## DDP-4 ##########
        # progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} train')
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} train') \
            if gpu == 0 else train_loader
        ##########################
        for batch in progress_bar_train:
            image, label = batch
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ########## DDP-4 ##########
            # if step % 10 == 0:
            if gpu == 0 and step % 10 == 0:
                progress_bar_train.set_postfix(loss=loss.item())
            ##########################
            step += 1

        with torch.no_grad():
            model.eval()
            preds = []
            targets = []
            ########## DDP-4 ##########
            # progress_bar_eval = tqdm(eval_loader, desc=f'Epoch {epoch + 1}/{epochs} eval')
            progress_bar_eval = tqdm(eval_loader, desc=f'Epoch {epoch + 1}/{epochs} eval') \
                if gpu == 0 else eval_loader
            ##########################
            for batch in progress_bar_eval:
                image, label = batch
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                output = model(image)
                pred = output.argmax(dim=1)

                ########## DDP-5 ##########
                pred = concat_all_gather(pred)
                label = concat_all_gather(label)
                ##########################
                preds.append(pred)
                targets.append(label)

            ########## DDP-4 ##########
            if gpu == 0:
                preds = torch.cat(preds)
                targets = torch.cat(targets)
                acc = (preds == targets).sum().cpu().numpy() / len(preds)
                print(f'Epoch {epoch + 1}/{epochs} acc: {format(acc, ".5f")}')
            ##########################

    ########## DDP-4 ##########
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    ##########################
    # 5. save
    if rank == 0:
        torch.save(model.module.state_dict(), 'mnist.pth')
    ##########################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()
    train(args)

########## DDP-0 ##########
# run(dual GPU on one device): torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 \
# --master_addr="192.168.1.250" --master_port=23456 ddp_mnist.py
##########################
if __name__ == '__main__':
    main()
```