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
![image](https://github.com/iLovEing/notebook/assets/109459299/c69251f2-9229-4365-bf19-40bc7a229ed8)
----- ***公式(b)***  
则原问题可以写为一下无约束形式：
![image](https://github.com/iLovEing/notebook/assets/109459299/8859c77e-a0eb-4601-9743-f27cf4510a6a)
----- ***公式(c)***  
> 简单说明：公式(c)和公式(a)同解

考虑原问题的两个约束：
1. $n_{j}$为等式约束，对拉格朗日函数 $L(x,\lambda,\eta)$ 求导即满足；
2. $m_{i}$为不等式约束，考虑两种情况：a. 如果 $x$ 不满足约束 $m_{i}$ ，即 $m_{i}(x) > 0$ ，观察拉格朗日函数，由于 $\lambda_{i} > 0$ ，则 $\max_\lambda{L(x,\lambda,\eta)}$ 值为正无穷；b. 如果 $x$ 满足约束 $m_{i}$ ，同理，$\max_\lambda{L(x,\lambda,\eta)}$ 值小于正无穷。由a. b. 可知：  
$\min_x{\max_\lambda{L}} =  min_x{[+\infty (when: m_i > 0),  L(when: m_i \le  0)]} = min_x{[L(when: m_i \le  0)]}$
即公式(c)的解隐式地满足 $m_{i} \le  0$ 
**故**：公式(c)和公式(a)同解

### 3. 对偶问题
#### 3.1 先写出原问题的对偶形式：





---

> 公式latex
> (a): \left\{\begin{matrix}\min_x{f(x)}, x\propto R^P \\s.t. m_i(x) \le 0, i=1, 2...M \\s.t. n_j(x) = 0, j=1, 2...N\end{matrix}\right.
> (b): L(x,\lambda,\eta) = f(x) + \sum_{i}^{M}\lambda_i m_i + \sum_{j}^{N}\eta_i m_i,x\propto R^P
> (c): \left\{\begin{matrix} \min_x{\max_{\lambda,\eta} L(x,\lambda,\eta)}  \\ s.t. \lambda_{i} \ge 0\end{matrix}\right.

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