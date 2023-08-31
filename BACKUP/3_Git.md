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

### 本地操作

#### 特殊符号
- “~” 接在分支、hash、HEAD后面，表示往上一笔修改，带数字表示往上n笔
- “^” 接在分支、hash、HEAD后面，表示往上一笔修改，带数字表示多个父亲的顺位序号（merge引起）
- "HEAD" 指向当前分支，当前hash的游标

#### branch
- git branch -avv 查看所有分支状态
- git branch -D [分支名] 删除本地分支
- git branch -m [原分支名] [新分支名] 修改分支名，若修改当前所在分支的名字，原分支名可以省略不写
- git branch -f [分支名] HEAD~3 强制移动main指向HEAD上三笔提交

#### merge, rebase, cherry-pick
- git cherry-pick hash1 hash2... 将hash1、hash2 cp到当前分支，并指向当前分支最新
- git merge [分支名] 将其他分支的修改merge到当前分支，并指向当前分支最新
- git rebase [分支名] 将当前分支修改merge到其他分支之后，并指向当前分支最新
- git rebase -i HEAD~4  -i表示手动调整模式，这里可以手动调整调整当前分支最近4笔的提交

---
### 暂存区相关

#### diff
- git diff 查看工作区修改
- git diff --cached 查看暂存区修改

#### reset, revert, rm, restore
- git reset --soft 修改HEAD位置并将当前左右修改和目标commit不同的地方全加入暂存区
- git reset --mixed （默认）修改HEAD位置并将当前左右修改和目标commit不同的地方全加入工作区
- git reset --hard 修改HEAD位置，当前修改和commit全部删除
- git revert HEAD 撤回当前提交，会生成新哈希，状态和上一笔相同
- git rm --cached [文件名] 撤销add，回到原始commit状态
- git restore --staged [文件名] 撤销add，但不修改内容（还可以重新add）

#### log, tag
- git log [分支名] 查看某分支的提交历史，不写分支名查看当前所在分支
- git log --oneline 一行显示提交历史
- git log -n 其中 n 是数字，查看最近 n 个提交
- git log --author [贡献者名字] 查看指定贡献者的提交记录
- git log --graph 图示法显示提交历史
- git reflog 记录本地所有修改
- git tag tag_name hash/HEAD 给某个commit打tag，tag相当于hash的别名
- git describe HEAD/hash 查看距离某个hash最近的tag


---

### 远程交互

#### 说明
1. clone远程分支后，本地有：
  - 与远程同名的分支，比如main，该分支可在本地任意改动
  - origin/main分支，表示远程分支指针，表示与远程分支的通信状态，在该分支上commit会分离HEAD（该分支不会改变）；


#### 拉取远程commit
- git fetch 改变本地远程分支指针，与远程仓库同步，但是不会改变本地分支
- git pull = fetch + merge，如果本地分支有修改，会对远端修改生成新的hash（merge而非rebase），不影响后续提交
- git pull --rebase = fetch + rebase
- git rebase origin/main 

#### push
- gut push 将本地修改提交到远端，同时更新本地远程分支指针
- git push [主机名] [本地分支名]:[远程分支名]  将本地分支推送到远程仓库的分支中，通常冒号前后的分支名是相同的，如果是相同的，可以省略 :[远程分支名]，如果远程分支不存在，会自动创建 (缺省自动关联tracking，还可以创建远程分支)
- +冒号
- git push origin dev/dev
- fetch 相反
- ~
- ^
- pull 参数

#### remote tracking
- remote tracking 隐含了本地分支pull和push默认跟踪的远程分支
- git checkout branch_name o/branch_name 创建分支时跟踪
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