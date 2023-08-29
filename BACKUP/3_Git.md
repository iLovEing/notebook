# [Git](https://github.com/iLovEing/notebook/issues/3)

# 重要！超级好用的git教程
[git教程](https://learngitbranching.js.org/?locale=zh_CN)

---

### 全局设置
- git config --global  --list

#### 配置用户名和email
- git config --global user.name "iLovEing"
- git config --global user.email "878314708@qq.com"

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

#### 暂存区（add）
查看修改 ：
- git diff --cached 

撤销暂存区：
- git reset -- [文件名] 
- git rm --cached [文件名] 
- git restore --sraged [文件名] 

#### log, tag
- git log [分支名] 查看某分支的提交历史，不写分支名查看当前所在分支
- git log --oneline 一行显示提交历史
- git log -n 其中 n 是数字，查看最近 n 个提交
- git log --author [贡献者名字] 查看指定贡献者的提交记录
- git log --graph 图示法显示提交历史
- git reflog 记录本地所有修改
- git tag tag_name hash/HEAD 给某个commit打tag，tag相当于hash的别名
- git describe HEAD/hash 查看距离某个hash最近的tag

#### branch
- git branch -avv 查看所有分支状态
- git branch -D [分支名] 删除本地分支
- git branch -m [原分支名] [新分支名] ，若修改当前所在分支的名字，原分支名可以省略不写
- git branch -f main HEAD~3 强制移动main指向HEAD上三笔提交

#### merge, rebase, cherry-pick
- git cherry-pick hash1 hash2... 将hash1、hash2 cp到当前分支，并指向当前分支最新
- git merge branch_name1 将branch_name1的修改merge到当前分支，并指向当前分支最新
- git rebase branch_name1 将当前分支修改merge到branch_name1之后，并指向当前分支最新
- git rebase -i HEAD~4 调整当前分支最近4笔的提交

#### branch
- git branch -avv 查看所有分支状态
- git branch -D [分支名] 删除本地分支
- git branch -m [原分支名] [新分支名] ，若修改当前所在分支的名字，原分支名可以省略不写
- git branch -f main HEAD~3 强制移动main指向HEAD上三笔提交
- git cherry-pick hash1 hash2... 将hash1、hash2 cp到当前分支

#### reset&revert
- git reset --soft
- git reset --hard
- git reset HEAD~ 回到上一笔哈希
- git revert HEAD 撤回当前提交，会生成新哈希，状态和上一笔相同


---

### 远程交互

#### 说明
1. clone远程分支后，本地有origin/main表示远程分支指针，表示与远程分支的通信状态，在该分支上commit会分离HEAD（该分支不会改变）；
2. git

#### 拉取远程commit
- git fetch 改变本地远程分支指针，与远程仓库同步，但是不会改变本地分支
- git pull = fetch + merge，如果本地分支有修改，会对远端修改生成新的hash（merge而非rebase）
- git rebase origin/main 

#### push
- git push [主机名] [本地分支名]:[远程分支名]  将本地分支推送到远程仓库的分支中，通常冒号前后的分支名是相同的，如果是相同的，可以省略 :[远程分支名]，如果远程分支不存在，会自动创建
- git push origin dev/dev

#### 跟踪/取消跟踪远程分支
- git branch -u [主机名/远程分支名] [本地分支名] 将本地分支与远程分支关联，或者说使本地分支跟踪远程分支。如果是设置当前所在分支跟踪远程分支，最后一个参数本地分支名可以省略不写
- git branch -u origin/dev
- git branch --unset-upstream [分支名] 即可撤销该分支对远程分支的跟踪，同样地，如果撤销当前所在的分支的跟踪，分支名可以省略不写
- git push -u origin dev 可在push时自动跟踪

#### 删除远程分支
首先，删除远程分支，使用 git push [主机名] :[远程分支名] ，如果一次性删除多个，可以这样：git push [主机名] :[远程分支名] :[远程分支名] :[远程分支名] 。此命令的原理是将空分支推送到远程分支，结果自然就是远程分支被删除。另一个删除远程分支的命令：git push [主机名] --delete [远程分支名]。删除远程分支的命令可以在任意本地分支中执行。两个命令分别试一下：
- git push origin :dev

#### 远程主机
- git remote -v 查看

使用 remote 系列命令来增加一个关联主机，执行 git remote add [主机名] [主仓库的地址]，注意，主仓库的地址使用 https 开头的：
- git remote add up https://github.com/iLovEing/hello_github
- git fetch up
- git rebase up/main        git pull --rebase up/main