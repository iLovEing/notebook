# [SVM](https://github.com/iLovEing/notebook/issues/15)

# support vector machine -- 支持向量机
###### 写在最前
个人理解，SVM最亮眼的几个点：
1. 把一个很直觉的想法用完美的数学语言描述出来，并且做了相当漂亮的求解，使得SVM天生就具有较好的鲁棒性；
2. 把原问题转化到对偶问题求解，带来了两个优点：
    1. 求解的维度从特征维度转换到样本维度，改变了问题求解的复杂度；
    2. **自然而然地引入核技巧，这是SVM性能优越的关键。**

### SVM基本思路
1. 提出分类问题确定超平面一个很直觉的想法：最大化最小间隔；
2. 将该想法转化为约束优化问题；
3. 利用拉格朗日乘子法，将原问题转成无约束形式；
4. 将原问题转化为对偶问题求解；
5. 根据slater条件，原问题和对偶问题同解，引入KKT条件；
6. 已知算法求解对偶问题；


### 大纲
- 先前知识：约束优化问题
- SVM
- Kernel SVM


---

## 先前知识: 约束优化问题求解
此章节讲解SVM中，带约束优化问题的求解，纯数学内容。

### 1. 原问题
考虑一个带不等式约束的优化问题：
![image](https://github.com/iLovEing/notebook/assets/109459299/5f25c6bd-0f31-4037-aed9-8d3123f6167a)
----- ***公式(a)***

### 2. 拉格朗日乘子法
使用拉格朗日乘子法，将原问题写成无约束形式。
引入拉格朗日函数：
![image](https://github.com/iLovEing/notebook/assets/109459299/fd66871e-3455-4fa6-b2e4-00cdae26870a)
----- ***公式(b)***


> L(x,\lambda,\eta) = f(x) + \sum_{i}^{M}\lambda_i m_i + \sum_{j}^{N}\eta_i m_i,x\propto R^P

---

## 正餐-SVM
hard margin + soft margin
![image](https://user-images.githubusercontent.com/109459299/224754112-cb9ab30c-bf30-48c9-bd99-42f8fc0507e2.png)
![image](https://user-images.githubusercontent.com/109459299/224754320-02d30bfa-4c87-4e77-a527-292d0fc8f231.png)
![image](https://user-images.githubusercontent.com/109459299/224754413-50ca05e3-60df-4302-ad01-d4c4942c101e.png)
![image](https://user-images.githubusercontent.com/109459299/224754475-c1ea23b9-7a01-4662-b77c-e17a0b778704.png)





---

## kernel SVM

挖个坑，写完核方法填坑。
（其实解出w和b已经能看出来了）