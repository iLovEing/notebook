# [conda&pip](https://github.com/iLovEing/notebook/issues/8)

## conda

#### 环境相关
conda --version
conda upgrade --all
conda env list / conda info -e

conda create --name test_env python=3.4
conda remove --name test_env --all
conda create --name test_env1 --clone test_env

conda active test_env
conda deactivate

导入导出环境 配置版
conda env export > environment.yaml
conda env create -f environment.yaml

#### 设置源
conda config --show channels
conda config --remove channels 源名称或链接
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --remove-key channels

#### 在环境中
conda list
conda list numpy
conda search (--full-name) numpy
conda install package=version --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda install numpy=1.93
conda install --name test_env numpy
conda uninstall package
conda update --all
conda update anaconda
conda update numpy



## pip
#### 查看和配置信息
pip3 --version
pip3 list
pip3 list --outdated
pip3 show package_name

pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 config list
pip3 config get global.index-url
pip3 config unset global.index-url

#### 安装包
pip3 search package_name
pip3 install --upgrade pip
pip3 install package_name==version
pip3 install --package_name
pip3 install package_name -i 镜像源
pip3 install c:\users\numpy-1.20.3-cp38-cp38-win_amd64.whl
pip3 uninstall package_name
python -m pip uninstall pip

#### 保存环境
pip3 freeze > requirements.txt
pip3 install -r requirements.txt

#### ipkernel
创建了新的conda环境，怎么加入kernel？
  1. conda create --name zlqiu python=3.10  # 创建环境
  2. conda activate zlqiu  # 进入环境
  3. conda install ipykernel  # 安装ipykernel  
  4. python -m ipykernel install --user --name zlqiu
  5. 刷新网站，此时新建notebook可以看到conda内核 zlqiu
  6. remove: jupyter kernelspec remove  zlqiu # 移除内核
