# [Ubuntu AI服务器系统安装](https://github.com/iLovEing/notebook/issues/29)

## 分区方案
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

## 系统设置
1. grub修改
    1. 修改配置文件: sudo gedit /etc/default/grub
       - GRUB_DEFAULT 代表默认的进入选项，从0开始
       - GRUB_TIMEOUT: 代表操作等待时间。默认10s
    2. 更新grub: sudo update-grub
2. 修改语言为中文，应用到全系统 

3. 4k屏幕显示太小: 修改显示器缩放至 150% or 175%
    如果没有分数比例调节，使用如下命令（不知道可都用）
    - wayland：gsettings set org.gnome.mutter experimental-features "['scale-monitor-framebuffer']"
    - X11：gsettings set org.gnome.mutter experimental-features "['x11-randr-fractional-scaling']"

4. 修改输入法为中文智能拼音（这里不要选择其他，点中文进去选择），修改默认语言、候选词个数

5. 修改终端复制快捷键

6. 修改高危文件夹背景色
   1. ~/.bashrc最后加上LS_COLORS=$LS_COLORS:'ow=1;32:'
   2. source /etc/profile
   > profile和 bashrc, 主要用来存放开机自运行的程序和命令，比如环境变量
   > /etc/profile 和 /etc/bash.bashrc, 影响所有用户，先加载 profile，profile里面加载 bashrc
   > ~/.profile 和 ~/.bashrc, 影响单个用户，加载顺序同理
   > 
   > ***tips***: 一般自己的修改放 bashrc 就可以了

7. 卸载snap(先安装chrome)
    1. 删除相关软件，执行多次remove_snap1，直到没有提示snap软件
    2. 删除 Snap 的 Core 文件，执行remove_snap2
    3. 删除snap管理工具：sudo apt autoremove --purge snapd
    4. 删除 Snap 的目录：
        > rm -rf ~/snap
        > sudo rm -rf /snap
        > sudo rm -rf /var/snap
        > sudo rm -rf /var/lib/snapd
        > sudo rm -rf /var/cache/snapd
    5. 配置 APT 参数：禁止 apt 安装 snapd
        > sudo sh -c "cat > /etc/apt/preferences.d/no-snapd.pref" << EOL
        > Package: snapd
        > Pin: release a=*
        > Pin-Priority: -10
        > EOL
    6. 禁用 snap Firefox 的更新
        > sudo sh -c "cat > /etc/apt/preferences.d/no-firefox.pref" << EOL
        > Package: firefox
        > Pin: release a=*
        > Pin-Priority: -10
        > EOL
    8. 重新安装Gnome商店
    - sudo apt install gnome-software
    - (or) sudo apt install --install-suggests gnome-software

8. 文件夹最近项目：

    - 打开：gsettings reset org.gnome.desktop.privacy remember-recent-files
    - 关闭：gsettings reset org.gnome.desktop.privacy remember-recent-files

---

## 系统环境和驱动
### apt软件源修改
1. 使用中科大源:  sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g'  /etc/apt/sources.list
   
2. 更新索引: sudo apt update

### linux-windows时间同步
- 方法1: 使用timedatectl修改为local rtc
  - timedatectl status # 查看状态
  - timedatectl set-local-rtc 1 # 使用local rtc
- 方法2: 使用ntpdate
  1. sudo apt install ntpdate
  2. sudo ntpdate time.windows.com
  3. sudo hwclock --localtime --systohc

### 显卡驱动
> ***attention***: 建议先安装ssh，防止用户界面无法进入
> ***attention***: 安装系统时勾选自动安装驱动，开机检查驱动版本，若版本cuda满足可不手动升级
1. 查看设备：lspci -nn | grep -i nvidia

2. 下载相关文件

    1. 安装编译工具

    > sudo apt install gcc
    > sudo apt install g++
    > sudo apt install make
    > sudo apt install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
    > sudo apt install --no-install-recommends libboost-all-dev
    > sudo apt install libopenblas-dev liblapack-dev libatlas-base-dev
    > sudo apt install libgflags-dev libgoogle-glog-dev liblmdb-dev
    > ***attention***:  如果有编译报错 尝试用gcc-12（新版本都需要）
    > sudo apt install gcc-12
    > sudo ln -sf /usr/bin/gcc-12 /etc/alternatives/cc

    2. 下载
       官网下载runf文件：https://www.nvidia.cn/Download/index.aspx?lang=cn

3. 屏蔽 nouveau
    1. lsmod | grep nouveau, 如果没有直接跳过
    1. sudo vim /etc/modprobe.d/blacklist.conf
    2. 添加 blacklist nouveau
    3. 添加 options nouveau modeset=0
    4. sudo update-initramfs -u 修改生效
    5. 重启验证 lsmod | grep nouveau

4. 安装驱动

    1. 关闭图形界面

    - 关闭图形界面：sudo telinit 3
    - 关掉图形服务：sudo /etc/init.d/gdm3 stop *or* sudo service lightdm stop

    2. 卸载原有驱动：sudo apt purge nvidia-*

    3. 安装

    - sudo chmod a+x NVIDIA-Linux-x86_64-525.89.02.run
    - sudo ./NVIDIA-Linux-x86_64-525.89.02.run
    > –no-nouveau-check 安装驱动时不检查nouveau
    > –no-opengl-files 只安装驱动文件,不安装OpenGL文件
    > 安装32位兼容库  --no
    > 运行x配置	-- yes

    > ***attention***: 开机黑屏，不加载图形界面, 去掉opengl
    4. 验证
       nvidia-smi

###  cuda enviroment
- cuda
1. 先安装 gcc g++ make
2. 按[官网](https://developer.nvidia.com/cuda-toolkit-archive)命令安装, 选择run file  local,不选driver  不选 kernel object

    > sudo sh cuda_12.4.1_550.54.15_linux.run
3. 环境变量, add to ~/.bashrc
    - export PATH=$PATH:/usr/local/cuda-12.1/bin
    - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
4. 验证 nvcc -V

- cudnn
1. 按[官网](https://developer.nvidia.com/cudnn)命令安装
    > sudo dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
    > sudo cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    > sudo apt-get update
    > sudo apt-get -y install cudnn-cuda-12
2. 验证 sudo find / -name *cudnn*.h, 看看有没有cudnn version相关的头文件

---

## 开发环境和软件安装
### 硬盘挂载
1. 手动挂载
   - df -l # 查看当前挂载情况
   - lsblk/fdisk -l 查看磁盘情况
   1. sudo mount /dev/nvme0n1p5 /mnt/test # 挂载
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
    > ***attention***: 把所有挂载的fsck检查关掉，似乎碰到过一直检查不开机的问题。
3. 挂载nas (smb)
    [参考链接](https://post.smzdm.com/p/a0x5q649/#:~:text=%E5%A8%81%E8%81%94%E9%80%9A%E8%87%AA%E5%B8%A6%E6%9C%89%20Ubuntu%20Linux%20%E5%B7%A5%E4%BD%9C%E7%AB%99%EF%BC%8C%E7%9B%B8%E4%BF%A1%E4%B8%8D%E5%B0%91%E6%9C%8B%E5%8F%8B%E9%83%BD%E6%9C%89%E4%BD%93%E9%AA%8C%E8%BF%87%E6%88%96%E6%AD%A3%E5%9C%A8%E7%94%A8%EF%BC%8C%E6%9C%AC%E6%9C%9F%E6%88%91%E5%B0%B1%E6%9D%A5%E8%AE%B2%E8%A7%A3%E4%B8%80%E4%B8%8B%E5%A6%82%E4%BD%95%E5%9C%A8Ubuntu%20%E6%8C%82%E8%BD%BDNAS%E4%B8%AD%E7%9A%84%E6%96%87%E4%BB%B6%E5%A4%B9,%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C%20%E2%96%BC%E9%A6%96%E5%85%88%E5%8E%BBNAS%E7%AB%AF%E6%8E%A7%E5%88%B6%E5%8F%B0%EF%BC%8C%E5%BC%80%E5%90%AFNFS%E6%9C%8D%E5%8A%A1%EF%BC%8C%E5%A6%82%E4%B8%8B%E5%9B%BE%E6%89%80%E7%A4%BA%E5%85%A8%E9%83%A8%E5%8B%BE%E9%80%89%E5%90%8E%E7%82%B9%E5%87%BB%E3%80%90%E5%BA%94%E7%94%A8%E3%80%91%20%E2%96%BC%E6%8E%A5%E4%B8%8B%E6%9D%A5%EF%BC%8C%E6%88%91%E4%BB%AC%E5%8E%BB%E7%BB%99%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9NFS%E6%9D%83%E9%99%90%EF%BC%8C%E3%80%90%E6%9D%83%E9%99%90%E3%80%91%E2%86%92%E3%80%90%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9%E3%80%91%EF%BC%8C%E9%80%89%E6%8B%A9%E4%BB%BB%E4%B8%80%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9%E8%BF%9B%E8%A1%8C%E6%9D%83%E9%99%90%E8%AE%BE%E7%BD%AE%20%E2%96%BC%E9%80%89%E6%8B%A9%E6%9D%83%E9%99%90%E7%B1%BB%E5%88%AB%E3%80%90NAS%E4%B8%BB%E6%9C%BA%E8%AE%BF%E9%97%AE%E3%80%91%20%E2%96%BC%E5%8B%BE%E9%80%89%E3%80%90%E8%AE%BF%E9%97%AE%E6%9D%83%E9%99%90%E3%80%91%EF%BC%8C%E5%8B%BE%E9%80%89%E3%80%90sync%E3%80%91%2C%E7%82%B9%E5%87%BB%E3%80%90%E5%BA%94%E7%94%A8%E3%80%91%E5%90%8E%EF%BC%8C%E2%80%9Cmmmm%E2%80%9D%E5%85%B1%E4%BA%AB%E6%96%87%E4%BB%B6%E5%A4%B9%E8%B5%8B%E6%9D%83%E5%AE%8C%E6%88%90)

- nas 端：
    1. 启用nfs，协议全部勾选；
    2. 打开共享文件夹nfs权限：权限 -> 共享文件夹-> 编辑权限 -> nas主机访问 -> 勾选访问权限、sync、wdelay、ip限制、权限

- 主机端：
    1. 安装nfs：sudo apt install nfs-common
    2. 查看可挂载文件夹：showmount -e 192.168.0.100
    3. sudo vim /etc/fstab
    > 192.168.0.100:/test                       /mnt/nas/test   nfs     defaults        0       0
    > 192.168.0.100:/share                      /mnt/nas/share  nfs     defaults        0       0
    > 192.168.0.100:/photo_backup_Q   /mnt/nas/photo_backup_Q   nfs     defaults        0       0
    > 192.168.0.100:/photo_backup_E   /mnt/nas/photo_backup_E   nfs     defaults        0       0
    > 192.168.0.100:/mm                           /mnt/nas/mm   nfs     defaults        0       0
    > 192.168.0.100:/download               /mnt/nas/download   nfs     defaults        0       0
    > 192.168.0.100:/data_snapshot     /mnt/nas/data_snapshot   nfs     defaults        0       0
    > 192.168.0.100:/data                       /mnt/nas/data   nfs     defaults        0       0
    > 192.168.0.100:/dl                           /mnt/nas/dl   nfs     defaults        0       0

### 软件安装
1. vpn
    1. 下载clash verge deb
    2. 安装deb
    3. 打开无界面：
    - vim /usr/share/applications/clash-verge.desktop 
    - Exec=env WEBKIT_DISABLE_COMPOSITING_MODE=1 clash-verge %u

    4. 下载或导入[配置](https://mojie0201.xn--8stx8olrwkucjq3b.com/api/v1/client/subscribe?token=b9b5782ec9fe6fb3cc179378eaa28cc0)

    

2. zerotier

   1. 下载并安装：curl -s https://install.zerotier.com | sudo bash
   2. 启动服务：sudo zerotier-cli start
   3. 开机启动：sudo systemctl enable zerotier-one.service
   4. 常用命令：
   - sudo zerotier-cli join/leave
   - sudo zerotier-cli status
   - sudo zerotier-cli listnetworks

   

3. python
    sudo ln -s /usr/bin/python3 /usr/bin/python # 使用linux自带即可

    

4. vim、git、7z、mpv (for vedio)、remmina (for remote pc)
    - sudo apt install vim
    - sudo apt install git
    - sudo apt install p7zip-full
    - sudo apt install mpv
    - sudo apt install remmina (remote args: ip、用户名、密码、协议选RDP、色深选真彩色24bpp)

    

5. chrome、qq、typora、baidunetdisk、wps、steam、[微信](https://github.com/lovechoudoufu/wechat_for_linux/releases)
    1. 官网下载deb
    2. 安装命令：sudo dpkg -i google-chrome-stable_current_amd64.deb
    3. [微信](https://github.com/lovechoudoufu/wechat_for_linux)
    
    
    
6. ssh
    1. sudo apt install net-tools  # 安装linux网络工具，比如ifconfig
    2. sudo apt install openssh-server  # 安装ssh server
    3. sudo gedit /etc/ssh/sshd_config # 配置文件 
    4. sudo systemctl start ssh 或 sudo service ssh start # 打开服务
    5. sudo systemctl status ssh 或 sudo service ssh status #查看服务状态
    6. sudo systemctl enable ssh # ssh开机启动 
    7. todo sftp

    

7. conda
    1. 下载：https://www.anaconda.com/download/
    2. 安装：bash Anaconda3-2024.02-1-Linux-x86_64.sh
    3. 配置：
      - sudo ln -s /usr/anaconda3/bin/conda /usr/bin/conda
      - conda config --show channels
      - conda config --set show_channel_urls yes
      - conda config --set auto_activate_base yes/false 打开终端自动进入conda base环境
      - conda config --add envs_dirs /home/iloveing/cache/conda_envs
      - conda config --add pkgs_dirs /home/iloveing/cache/conda_pkgs

    

8. pycharm & clion
    1. 官网下载tar.gz
    2. 不用安装，直接解压 tar -zxf *.gz
    3. 启动脚本：bin/pycharm.sh，按教程破解
    4. 创建桌面快捷方式 vim pycharm.desktop，并允许运行：
        >[Desktop Entry]
        >Version=1.0
        >Name=pycharm
        >Exec=/usr/jetbrains/pycharm-2024.2/bin/pycharm
        >Terminal=false
        >Type=Application

    

9. 系统监视器：indicator-sysmonitor
    > sudo add-apt-repository ppa:fossfreedom/indicator-sysmonitor
    > sudo apt update
    > sudo apt install indicator-sysmonitor
    > sudo add-apt-repository --remove ppa:fossfreedom/indicator-sysmonitor
    >
    > > txt: ║{net}║CPU {cpu} - {cputemp}║GPU {nvgpu} - {nvgputemp}║MEM {mem}║

    

10. fancontrol、OpenRGB、bt