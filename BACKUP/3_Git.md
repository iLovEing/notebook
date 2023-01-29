# [Git](https://github.com/iLovEing/notebook/issues/3)

### 全局设置
- git config --global  --list

---

#### 配置用户名和email
- git config --global user.name "iLovEing"
- git config --global user.email "878314708@qq.com"

---

#### 命令缩写
- git config --global alias.st status
- git config --global alias.pl pull
- git config --global alias.ps push
- git config --global alias.cm commit
- git config --global alias.lg log
- git config --global alias.ck checkout
- git config --global alias.br branch
- git config --global alias.mg merge
- git config --global alias.brs "branch -avv"

---

### 本地修改

---

#### 暂存区（add）
查看修改 ：
- git diff --cached 

撤销暂存区：
- git reset -- [文件名] 
- git rm --cached [文件名] 
- git restore --sraged [文件名] 

---

#### 查看历史
git log [分支名] 查看某分支的提交历史，不写分支名查看当前所在分支
git log --oneline 一行显示提交历史
git log -n 其中 n 是数字，查看最近 n 个提交
git log --author [贡献者名字] 查看指定贡献者的提交记录
git log --graph 图示法显示提交历史
git reflog 记录本地所有修改
git branch -avv 查看所有分支状态

---

### reset
git reset --soft
git reset --hard