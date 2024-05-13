# [Ubuntu AI服务器系统安装](https://github.com/iLovEing/notebook/issues/29)

123

---

# linux ubuntu

## 系统安装

### 分区方案
1. efi分区: 主分区/逻辑分区 EFI系统分区 512MB  
UEFI引导分区
2. swap分区: 主分区/逻辑分区 交换空间 8192MB(8GB)  
内存交换分区，Redhat官方文档关于swap分区大小设置的建议

|   物理内存  |   建议大小  | 开启休眠功能建议大小 |
| ---------- | --------- | ---------------- |
| ⩽ 2GB      |  内存的2倍  |     内存的3倍     |
| 2GB – 8GB  | 等于内存大小 |     内存的2倍     |
| 8GB – 64GB |   至少4G   |    内存的1.5倍    |
| \> 64GB    |   至少4G   |   不建议使用休眠   |

3. /: 主分区 Ext4日志文件系统 131072MB(128GB)  
ubuntu 根目录
4. /usr: 逻辑分区 Ext4日志文件系统 262144MB(256GB)  
存放用户程序，一般在/usr/bin中存放发行版提供的程序，用户自行安装的程序默认安装到/usr/local/bin中
5. /home: 逻辑分区 Ext4日志文件系统 rest storage  
存放用户文件，这个分区尽量设置大。
6. /var: 个人用不分区，服务器可考虑  
存放一些临时文件比如日志，服务器可考虑单独分区，否则放在根分区下

> ***attention***: 系统安装选择安装到efi分区
> ***attention***: 系统开始时选择语言为english，关系到home目录文件夹名称；开机后替换
> ***tips***: 安装为windows&ubuntu双系统，先关闭win快速启动 
---

### grub修改
1. 修改配置文件: sudo gedit /etc/default/grub
   - GRUB_DEFAULT 代表默认的进入选项，从0开始
   - GRUB_TIMEOUT: 代表操作等待时间。默认10s
2. 更新grub: sudo update-grub

---

### 系统设置
1. 修改语言为中文，应用到全系统
2. 修改输入法为中文智能拼音（这里不要选择其他，点中文进去选择）
3. 4k屏幕显示太小: 修改显示器缩放至 175%
	如果没有分数比例调节，使用如下命令（不知道可都用）
    - wayland：gsettings set org.gnome.mutter experimental-features "['scale-monitor-framebuffer']"
    - X11：gsettings set org.gnome.mutter experimental-features "['x11-randr-fractional-scaling']"
4. 修改终端复制快捷键
5. 修改高危文件夹背景色
    1. ～/.bashrc最后加上LS_COLORS=$LS_COLORS:'ow=1;32:'
    2. source /etc/profile
   > profile和 bashrc, 主要用来存放开机自运行的程序和命令，比如环境变量  
   > /etc/profile 和 /etc/bash.bashrc, 影响所有用户，先加载 profile，profile里面加载 bashrc  
   > ~/.profile 和 ~/.bashrc, 影响单个用户，加载顺序同理  
   >   
   > ***tips***: 一般自己的修改放 bashrc 就可以了


---

### 系统环境和驱动
#### apt软件源修改
1. 使用中科大源: sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
2. 更新索引: sudo apt update

#### linux-windows时间同步
- 方法1: 使用timedatectl修改为local rtc
  - timedatectl status # 查看状态
  - timedatectl set-local-rtc 1 # 使用local rtc
- 方法2: 使用ntpdate
  1. sudo apt install ntpdate
  2. sudo ntpdate time.windows.com
  3. sudo hwclock --localtime --systohc

#### 显卡驱动
> ***attention***: 建议先安装ssh，防止用户界面无法进入
> ***attention***: 安装系统时勾选自动安装驱动，开机检查驱动版本，若版本cuda满足可不手动升级
1. 查看设备：lspci -nn | grep -i nvidia
2. 屏蔽 nouveau
    1. sudo vim /etc/modprobe.d/blacklist.conf
    2. 添加 blacklist nouveau
    3. 添加 options nouveau modeset=0
    4. sudo update-initramfs -u 修改生效
    5. 重启验证 lsmod | grep nouveau

3. 卸载原有驱动：sudo apt purge nvidia-* 

4. 安装驱动
    1. 安装编译工具
    > sudo apt install gcc
    > sudo apt install g++
    > sudo apt install make
    > sudo apt install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
    > sudo apt install --no-install-recommends libboost-all-dev
    > sudo apt install libopenblas-dev liblapack-dev libatlas-base-dev
    > sudo apt install libgflags-dev libgoogle-glog-dev liblmdb-dev
    > ***attention***:  如果有编译报错 尝试用gcc-12
    > sudo apt install gcc-12
    > sudo ln -sf /usr/bin/gcc-12 /etc/alternatives/cc

    2. 下载
    官网下载runf文件：https://www.nvidia.cn/Download/index.aspx?lang=cn

    3. 关闭图形界面
    - 打开命令行界面: sudo telinit 3
    - 关闭图形界面: sudo service lightdm stop / sudo /etc/init.d/gdm3 stop

    4. 安装
    - sudo chmod a+x NVIDIA-Linux-x86_64-525.89.02.run
    - sudo ./NVIDIA-Linux-x86_64-525.89.02.run -no-opengl-files
    > –no-x-check 安装驱动时不检查X服务
    > –no-nouveau-check 安装驱动时不检查nouveau
    > –no-opengl-files 只安装驱动文件,不安装OpenGL文件
    > 安装32位兼容库  --no
    > 运行x配置	-- yes

    > ***attention***: 开机黑屏，不加载图形界面, 去掉opengl
    5. 验证
    nvidia-smi

####  cuda enviroment
- cuda
1. 先安装 gcc g++ make
2. 按[官网](https://developer.nvidia.com/cuda-toolkit-archive)命令安装, 选择run file  local,不选driver  不选 kernel object
3. 环境变量, add to ~/.bashrc
    - export PATH=$PATH:/usr/local/cuda-12.1/bin  
    - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
4. 验证 nvcc -V

- cudnn
1. 按[官网](https://developer.nvidia.com/cudnn)命令安装
2. 验证 sudo find / -name *cudnn*.h, 看看有没有cudnn version相关的头文件

---

### 开发环境和软件安装
#### 硬盘挂载
1. 手动挂载
   - df -l # 查看当前挂载情况
   - lsblk/fdisk -l 查看磁盘情况
   1. sudo mount /dev/nvme0n1p4 /mnt/test # 挂载
   2. sudo umount -v /mnt/test # 卸载
   
2. 开机自动挂载
   1. sudo blkid /dev/nvme0n1p6 或者 lsblk -f # 查看uuid
   2. /mnt下创建挂载文件夹
   3. sudo vim/gedit /etc/fstab
   4. 末尾添加 UUID=63F93D743FFF6146 /home/cache ntfs defaults 0 2
   > defaults: 文件参数，比如权限，默认即可  
   > 0: 是否dump备份, 0表示重不，1表示每天，2表示其他定期  
   > 2: 是否开机进行fsck检查，一般系统文件会进行检验，0表示从不，1/2代表检验顺序，比如根目录为1
   5. sudo mount -a # 执行挂载
   
3. 挂载nas (smb)

#### 开发环境
1. python  
sudo ln -s /usr/bin/python3 /usr/bin/python # 使用linux自带即可

2. vim  
sudo apt install vim

3. ssh  
   1. sudo apt install net-tools  # 安装linux网络工具，比如ifconfig
   2. sudo apt install openssh-server  # 安装ssh server
   3. sudo gedit /etc/ssh/sshd_config # 配置文件 todo: 备份
   4. systemctl start ssh 或 sudo service ssh start # 打开服务
   5. systemctl status ssh 或 sudo service ssh status #查看服务状态
   6. sudo systemctl enable ssh # ssh开机启动  

4. git
sudo apt install git

5. conda
https://www.anaconda.com/download/ 下载
bash Anaconda3-2024.02-1-Linux-x86_64.sh

conda config --show channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

conda config --set auto_activate_base yes/false 打开终端自动进入conda base环境

vim ~condarc
envs_dirs:
  - /home/iloveing/datas/linux/conda/envs
  - /home/iloveing/anaconda3/envs
  - /home/iloveing/.conda/envs

pkgs_dirs:
  - /home/iloveing/cache/conda/pkgs
  - /home/iloveing/anaconda3/pkgs
  - /home/iloveing/.conda/pkgs

#### 软件安装
chrome
官网下载
sudo dpkg -i google-chrome-stable_current_amd64.deb

qq
官网下载deb

Typora
官网下载deb




pycharme
wps
vpn
OpenRGB
indicator-sysmonitor
steam
baidunetdisk



---


VPN
7.1 copy配置文件改名为config.yaml， clash文件Country.mmdb
7.2 下载amd64版本 https://github.com/Dreamacro/clash/releases
7.3 三个文件放一起，可能要chmod 777 clash
7.4 创建service
sudo vim /etc/systemd/system/clash.service
[Unit]
Description=Clash

[Service]
Type=simple
Restart=always
ExecStart=/usr/local/clash/clash -d /usr/local/clash

[Install]
WantedBy=multi-user.target
7.5 启动service并设置为开机启动
sudo systemctl daemon-reload
sudo systemctl enable clash
sudo systemctl start clash
7.6 网络设置代理
手动 http和https都用127.0.0.1 端口和config.yaml一致，这里是7890

7.7 测试
systemctl status clash
curl -i google.com 可能要重启终端

7.8 开关方法
设置里代理开关，终端内要重启生效

7.9 设置
http://clash.razord.top
网址 端口 密钥要和config.yaml（external-controller， secret）

7.10 可以在某个终端单独开启/关闭
export http_proxy="http://127.0.0.1:7890"
export https_proxy="https://127.0.0.1:7890"
unset http_proxy
unset https_proxy