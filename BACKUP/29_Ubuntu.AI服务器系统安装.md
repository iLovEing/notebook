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
1. win双系统修改grub
    1. 修改配置文件: sudo gedit /etc/default/grub
        - GRUB_DEFAULT 代表默认的进入选项，从0开始
        - GRUB_TIMEOUT: 代表操作等待时间。默认10s
    2. 更新grub: sudo update-grub
2. 修改语言为中文，应用到全系统
3. 修改输入法为中文智能拼音（这里不要选择其他，点中文进去选择）
4. 4k屏幕显示太小: 修改显示器缩放至 175%
	如果没有分数比例调节，使用如下命令（不知道可都用）
    - wayland：gsettings set org.gnome.mutter experimental-features "['scale-monitor-framebuffer']"
    - X11：gsettings set org.gnome.mutter experimental-features "['x11-randr-fractional-scaling']"
5. 修改终端复制快捷键
6. 修改高危文件夹背景色
    1. ～/.bashrc最后加上LS_COLORS=$LS_COLORS:'ow=1;32:'
    3. source /etc/profile
   > profile和 bashrc, 主要用来存放开机自运行的程序和命令，比如环境变量  
   > /etc/profile 和 /etc/bash.bashrc, 影响所有用户，先加载 profile，profile里面加载 bashrc  
   > ~/.profile 和 ~/.bashrc, 影响单个用户，加载顺序同理  
   >   
   > ***tips***: 一般自己的修改放 bashrc 就可以了