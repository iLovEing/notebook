# [Ubuntu AI服务器系统安装](https://github.com/iLovEing/notebook/issues/29)

# linux ubuntu

本记录为个人安装ubuntu桌面和windows双系统流程，安装ubuntu服务器也可参考，其中一些部分可以忽略。

---

## 一、系统安装

### 1.1 安装设置
1. ubuntu桌面版本安装系统时选择安装第三方显卡驱动，否则新显卡有可能卡开机界面。
2. 若安装windows&ubuntu双系统，先关闭win快速启动
3. 系统开始时选择语言为english，关系到home目录文件夹名称，有需要开机后改回即可

### 1.2 分区方案
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
    存放用户程序，一般在/usr/bin中存放发行版提供的程序，用户自行安装的程序默认安装到/usr/local/bin中，个人首次使用可以不用单独创建此分区，把根目录空间设置大一些即可
5. /home: 逻辑分区 Ext4日志文件系统 rest storage
    存放用户文件，这个分区尽量设置大。
6. /var: 个人用不分区，服务器可考虑
    存放一些临时文件比如日志，服务器可考虑单独分区，否则放在根分区下
7. 分区完成后，选择安装到efi分区所在的磁盘号

---

## 二、系统配置
### 2.1 before use（个人需求）
#### 2.1.1. 挂载硬盘
```sh
# 1. 手动挂载
df -l # 查看当前挂载情况
lsblk/fdisk -l 查看磁盘情况
sudo mount /dev/nvme0n1p5 /mnt/test # 挂载
sudo umount -v /mnt/test # 卸载

# 2. 开机自动挂载
sudo blkid /dev/nvme0n1p6 或者 lsblk -f # 查看uuid
sudo vim/gedit /etc/fstab  # 编辑fstab，添加以下内容
	# iloveing @ mount dual sys common dir
    UUID=14563F8B563F6C9C     	/mnt/dual_sys     ntfs    defaults,nofail 0       0

sudo chown -R iloveing test  # 修改用户
sudo chown -R :iloveing test  # 修改组
sudo mount -a  # 执行挂载
```
> **参数解释**
>
> - defaults: 文件参数，比如权限，默认即可 
> - 0: 是否dump备份, 0表示重不，1表示每天，2表示其他定期 
> - 2: 是否开机进行fsck检查，一般系统文件会进行检验，0表示从不，1/2代表检验顺序，比如根目录为1
> - nofail：不阻塞开机
>
> **问题指引**
>
> 双系统可能遇到问题：Falling back to read-only mount because the NTFS partition is in an unsafe state.Please resume and shutdown Windows fully (no hibernation or fast restarting.)。原因是当 Windows 快速启动、休眠、非正常关机时 Linux 无法写入 NTFS ，需要关闭快速启动，清除休眠文件，可以通过 ntfs-3g -o remove_hiberfile 删除非正常关机保存的休眠文件：
>
> 1. 关闭windows快速启动
> 2. sudo umount /mnt/dual_sys
> 3. sudo ntfs-3g -o remove_hiberfile /dev/nvme1n1p1 /mnt/dual_sys

#### 2.1.2. 挂载nas
这里使用nfs方式挂载威联通nas，[参考链接](https://post.smzdm.com/p/a0x5q649/#:~:text=%E5%A8%81%E8%81%94%E9%80%9A%E8%87%AA%E5%B8%A6%E6%9C%89%20Ubuntu%20Linux%20%E5%B7%A5%E4%BD%9C%E7%AB%99%EF%BC%8C%E7%9B%B8%E4%BF%A1%E4%B8%8D%E5%B0%91%E6%9C%8B%E5%8F%8B%E9%83%BD%E6%9C%89%E4%BD%93%E9%AA%8C%E8%BF%87%E6%88%96%E6%AD%A3%E5%9C%A8%E7%94%A8%EF%BC%8C%E6%9C%AC%E6%9C%9F%E6%88%91%E5%B0%B1%E6%9D%A5%E8%AE%B2%E8%A7%A3%E4%B8%80%E4%B8%8B%E5%A6%82%E4%BD%95%E5%9C%A8Ubuntu%20%E6%8C%82%E8%BD%BDNAS%E4%B8%AD%E7%9A%84%E6%96%87%E4%BB%B6%E5%A4%B9,%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C%20%E2%96%BC%E9%A6%96%E5%85%88%E5%8E%BBNAS%E7%AB%AF%E6%8E%A7%E5%88%B6%E5%8F%B0%EF%BC%8C%E5%BC%80%E5%90%AFNFS%E6%9C%8D%E5%8A%A1%EF%BC%8C%E5%A6%82%E4%B8%8B%E5%9B%BE%E6%89%80%E7%A4%BA%E5%85%A8%E9%83%A8%E5%8B%BE%E9%80%89%E5%90%8E%E7%82%B9%E5%87%BB%E3%80%90%E5%BA%94%E7%94%A8%E3%80%91%20%E2%96%BC%E6%8E%A5%E4%B8%8B%E6%9D%A5%EF%BC%8C%E6%88%91%E4%BB%AC%E5%8E%BB%E7%BB%99%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9NFS%E6%9D%83%E9%99%90%EF%BC%8C%E3%80%90%E6%9D%83%E9%99%90%E3%80%91%E2%86%92%E3%80%90%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9%E3%80%91%EF%BC%8C%E9%80%89%E6%8B%A9%E4%BB%BB%E4%B8%80%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9%E8%BF%9B%E8%A1%8C%E6%9D%83%E9%99%90%E8%AE%BE%E7%BD%AE%20%E2%96%BC%E9%80%89%E6%8B%A9%E6%9D%83%E9%99%90%E7%B1%BB%E5%88%AB%E3%80%90NAS%E4%B8%BB%E6%9C%BA%E8%AE%BF%E9%97%AE%E3%80%91%20%E2%96%BC%E5%8B%BE%E9%80%89%E3%80%90%E8%AE%BF%E9%97%AE%E6%9D%83%E9%99%90%E3%80%91%EF%BC%8C%E5%8B%BE%E9%80%89%E3%80%90sync%E3%80%91%2C%E7%82%B9%E5%87%BB%E3%80%90%E5%BA%94%E7%94%A8%E3%80%91%E5%90%8E%EF%BC%8C%E2%80%9Cmmmm%E2%80%9D%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9%E8%B5%8B%E6%9D%83%E5%AE%8C%E6%88%90)。

1. nas 端：
   1. 启用nfs，协议全部勾选；
   1. 打开共享文件夹nfs权限：权限 -> 共享文件夹-> 编辑权限 -> nas主机访问 -> 勾选访问权限、sync、wdelay、ip限制、权限

2. 主机端：
    ```sh
    # 1. 安装nfs
    sudo apt install nfs-common
    # 2. 查看可挂载文件夹
    showmount -e 192.168.0.100
    # 3. 修改fstab
    sudo vim /etc/fstab
        # iloveing @ mount nas
        192.168.0.100:/ai      	    /mnt/nas/ai       nfs     defaults,nofail 0       0
        192.168.0.100:/backup       /mnt/nas/backup   nfs     defaults,nofail 0       0
        192.168.0.100:/doc          /mnt/nas/doc      nfs     defaults,nofail 0       0
        192.168.0.100:/share        /mnt/nas/share    nfs     defaults,nofail 0       0
        192.168.0.100:/snapshot     /mnt/nas/snapshot nfs     defaults,nofail 0       0
        192.168.0.100:/sundry       /mnt/nas/sundry   nfs     defaults,nofail 0       0
        # 192.168.0.100:/_entry       /mnt/nas/_entry   nfs     defaults,nofail 0       0
    
    # 4. 修改用户和组
    sudo chown -R iloveing nas  # 修改用户
    sudo chown -R :iloveing nas  # 修改组
    
    # 挂载
    sudo mount -a
    ```

#### 2.1.3. 配置ssh

```sh
# 1. 安装相关服务和工具
sudo apt install net-tools
sudo apt install openssh-server
# 2. 打开服务
sudo systemctl start ssh
# 3. 开机启动 
sudo systemctl enable ssh
# 4. 按需配置
sudo systemctl status ssh  # 查看服务状态
sudo vim /etc/ssh/sshd_config  # 修改配置文件 
```

#### 2.1.4. 配置zerotier

```sh
# 1. 下载并安装
curl -s https://install.zerotier.com | sudo bash
# 2. 启动服务
sudo zerotier-cli start
# 3. 开机启动
sudo systemctl enable zerotier-one.service
# 4. 常用命令：
sudo zerotier-cli join/leave  # 加入/离开网段
sudo zerotier-cli status  # 查看服务状态
sudo zerotier-cli listnetworks  # 查看网络状态
```

### 2.2 系统设置

#### 2.2.1. 双系统修改grub
1. 修改配置文件: sudo gedit /etc/default/grub
   - GRUB_DEFAULT 代表默认的进入选项，从0开始
   - GRUB_TIMEOUT: 代表操作等待时间。默认10s
2. 更新grub: sudo update-grub

#### 2.2.2. 系统设置
1. 4k屏幕显示太小: 修改显示器缩放至 150% or 175%
2. 修改系统语言为中文，这里不要动home目录名称
3. 修改输入法为中文智能拼音（这里不要选择其他，点中文进去选择），修改默认语言、候选词个数
4. 文件夹最近项目
    ```sh
    # 打开
    gsettings reset org.gnome.desktop.privacy remember-recent-files
    # 关闭
    gsettings reset org.gnome.desktop.privacy remember-recent-files
    ```
5. apt软件源修改
    ```sh
    # 使用中科大源
    sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g'  /etc/apt/sources.list
    # 更新索引
    sudo apt update
    ```

6. linux-windows时间同步

    ```sh
    # 方法1: 使用timedatectl修改为local rtc
    ## 查看状态
    timedatectl status
    ## 使用local rtc
    timedatectl set-local-rtc 1
    
    # 方法2: 使用ntpdate
    sudo apt install ntpdate
    sudo ntpdate time.windows.com
    sudo hwclock --localtime --systohc
    ```

7. python(按需)
    ```sh
    sudo ln -s /usr/bin/python3 /usr/bin/python # 使用linux自带
    ```
#### 2.2.3. 终端设置
1. 修改终端复制快捷键
2. 修改高危文件夹背景色（绿色背景很恼人）
   1. ~/.bashrc最后加上LS_COLORS=$LS_COLORS:'ow=1;32:'
   2. source /etc/profile

---

## 三、软件安装

### 3.1 cuda环境
#### 3.1.1. 显卡驱动

先到[官网](https://www.nvidia.cn/Download/index.aspx?lang=cn)下载对应系统的run文件：

```sh
# 1.查看设备
lspci -nn | grep -i nvidia

# 2.安装编译环境
sudo apt install gcc
sudo apt install g++
sudo apt install make
sudo apt install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt install --no-install-recommends libboost-all-dev
sudo apt install libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt install gcc-12
sudo ln -sf /usr/bin/gcc-12 /etc/alternatives/cc

# 3.屏蔽nouveau
# 3.1 查看nouveau设备，如果没有直接跳过
lsmod | grep nouveau
# 3.2 修改blacklist
sudo vim /etc/modprobe.d/blacklist.conf
## 添加以下内容
blacklist nouveau
options nouveau modeset=0
# 3.3 修改生效
sudo update-initramfs -u
# 3.4 重启后验证
lsmod | grep nouveau

# 4.安装驱动
# 4.1 关闭图形界面
sudo telinit 3
# 4.2 关闭图形服务
sudo /etc/init.d/gdm3 stop
sudo service lightdm stop
# 4.3 卸载原有驱动
sudo apt purge nvidia-*
# 4.4 安装驱动（32 lib - no;x config - yes）
sudo chmod a+x NVIDIA-Linux-x86_64-525.89.02.run
sudo ./NVIDIA-Linux-x86_64-525.89.02.run
# 4.5 验证
nvidia-smi
```

**禁用内核升级!（重要）**
​nvidia驱动安装时会自动对内核版本签名，内核不匹配有概率卡开机界面（或者显卡掉驱动，nvidia-smi无反应）。而ubuntu默认打开系统和内核更新，因此如果在某一次正常关机后再开机卡在开机界面，大概率是内核版本更新导致的。禁用内核升级参考tips，如果已经出现卡开机现象，可以进grub选择旧内核进入系统，卸载新内核修复或者尝试ssh进系统更改内核启动顺序（没尝试过，如果ssh可用可有线尝试）。

####  3.1.2. cuda
1. 参考[官网](https://developer.nvidia.com/cuda-toolkit-archive)命令安装
   - 官网选择run file  local
   - 安装选项不选driver  不选 kernel object

2. 添加环境变量

```sh
# add to ~/.bashrc
export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
```

3. 验证 nvcc -V

#### 3.1.3. cudnn

1. 参考按[官网](https://developer.nvidia.com/cudnn)命令安装，同样选择 run file  local
2. 验证 sudo find / -name *cudnn*.h, 看看有没有cudnn version相关的头文件



### 3.2. 软件安装

- **conda**
    ```sh
    # 1.下载：https://www.anaconda.com/download/
    # 2.安装
    bash Anaconda3-2024.02-1-Linux-x86_64.sh
    # 3.配置
    sudo ln -s /usr/anaconda3/bin/conda /usr/bin/conda
    conda config --show channels
    conda config --set show_channel_urls yes
    conda config --set auto_activate_base yes/false 打开终端自动进入conda base环境
    conda config --add envs_dirs /home/iloveing/cache/conda_envs
    conda config --add pkgs_dirs /home/iloveing/cache/conda_pkgs
    ```

- **vpn**
  
    1. 下载clash verge deb
    2. 安装deb
    3. 下载或导入[配置](https://mojie0201.xn--8stx8olrwkucjq3b.com/api/v1/client/subscribe?token=b9b5782ec9fe6fb3cc179378eaa28cc0)
    4. 打开无界面问题解决：
    - vim /usr/share/applications/clash-verge.desktop 
    - Exec=env WEBKIT_DISABLE_COMPOSITING_MODE=1 clash-verge %u
    
- **vim、git、7z、mpv (for vedio)、remmina**
  1. sudo apt install vim git p7zip-full mpv remmina
  2. remmina 远程参数：ip、用户名、密码、协议选RDP、色深选真彩色24bpp
  
  
  
- **chrome、qq、typora、baidunetdisk、wps、steam、[微信](https://github.com/lovechoudoufu/wechat_for_linux/releases)**
  
  1. 官网下载deb
  2. 安装命令：sudo dpkg -i google-chrome-stable_current_amd64.deb

- **jetbrains**
1. 官网下载tar.gz
2. 不用安装，直接解压 tar -zxf *.gz
3. 启动脚本：bin/pycharm.sh，按教程破解
4. 创建桌面快捷方式 vim pycharm.desktop，并允许运行：
   
    ```sh
    [Desktop Entry]
    Version=1.0
    Name=pycharm
    Exec=/usr/jetbrains/pycharm-2024.2/bin/pycharm
    Terminal=false
    Type=Application
    ```

- **系统监视器：indicator-sysmonitor**
    ```sh
    sudo add-apt-repository ppa:fossfreedom/indicator-sysmonitor
    sudo apt update
    sudo apt install indicator-sysmonitor
    sudo add-apt-repository --remove ppa:fossfreedom/indicator-sysmonitor
    
    # txt: ║{net}║CPU {cpu} - {cputemp}║GPU {nvgpu} - {nvgputemp}║MEM {mem}║
    ```

- **notepad--**

    ```sh
    # 1. 重命名appimage文件
    # 2. 在终端直接运行
    # 3. 需要fuse
    sudo apt install libfuse2
    ```

- **fancontrol、OpenRGB、bt**