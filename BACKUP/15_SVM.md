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

### 1. 写出原问题
考虑一个带不等式约束的优化问题：
![image](https://github.com/iLovEing/notebook/assets/109459299/5f25c6bd-0f31-4037-aed9-8d3123f6167a)
----- ***公式(a)***

### 2. 使用拉格朗日乘子法求解约束优化问题
使用拉格朗日乘子法，将原问题写成无约束形式。
引入拉格朗日函数：
![image](https://github.com/iLovEing/notebook/assets/109459299/4368ede6-ee66-4e94-9fa6-6183c6b566fd)
----- ***公式(b)***  
则原问题可以写为一下无约束形式：
![image](https://github.com/iLovEing/notebook/assets/109459299/8859c77e-a0eb-4601-9743-f27cf4510a6a)
----- ***公式(c)***  
> 补充说明1：嵌套优化问题的理解
这里关于 $min$ ,  $max$ 两层嵌套优化问题的含义可以这样理解：假设固定某个 $x$ ，则遍历 $λ$ ， $η$ 寻找 $L$ 的最大值作为该 $x$ 的函数值，而最终的结果是遍历所有 $x$ ，寻找所有 $x$ 下函数值的最小值作为最终解。

> 补充说明2：公式(c)和公式(a)同解
> 考虑原问题的两个约束：
> - $n_{j}$为等式约束，对拉格朗日函数 $L(x, λ, η)$ 求导即满足；
> - $m_{i}$为不等式约束，考虑两种情况：
> a. 如果 $x$ 不满足约束 $m_{i}$ ，即 $m_{i}(x) > 0$ ，观察拉格朗日函数，由于 $λ_{i} > 0$ ，则 $\max_λ{L(x, λ, η)}$ 值为正无穷；
> b. 如果 $x$ 满足约束 $m_{i}$ ，同理， $\max_λ{L(x, λ, η)}$ 值小于正无穷。  
> 由a. b. 易推断：  
> $\min_x{\max_λ{L}} =  min_x{\\{+\infty (when: m_i > 0),  L(when: m_i \le  0)\\}} = min_x{\\{L(when: m_i \le  0)\\}}$
> 即公式(c)的解隐式地满足 $m_{i} \le  0$ 
> 
> **故**：公式(c)和公式(a)同解

### 3. 转化为对偶问题
#### 3.1 先写出原问题的对偶形式，其实就是将 $min$ 和 $max$ 对调：
![image](https://github.com/iLovEing/notebook/assets/109459299/4d0d5305-5d62-45ed-9e4d-cf7311a16fec)
----- ***公式(d)***  

#### 3.2 弱对偶性与强对偶性
这一节主要讨论原问题和对偶问题解的关系（这里的解指的是满足条件下拉格朗日函数 $L$ 的值）。
- 弱对偶性：对偶问题的解**小于等于**原问题，**天然拥有**，即
$\max_{λ, η}{\min_x L(x, λ, η)} \le \min_x{\max_{λ, η} L(x, λ, η)} $
证明：
显然有： $\min_x{L} \le L \le \max_{λ, η}L$
$\therefore \max_{λ, η}{\min_x{L}} \le L \le \min_x{\max_{λ, η}L}$
即，在小的值里面取最大的，肯定小于等于 大的值里面取最小的。

- 强对偶性：对偶问题的解**等于**原问题，当优化问题满足某些条件时拥有强对偶性。这里直接写结论，凸优化问题，满足slater条件时拥有强对偶性，原问题和对偶问题同解。
> 补充说明3：对偶问题的几何理解
> 简化原问题，考虑一维不等式约束问题： $\min_x{f(x)}, s.t. m(x) \le 0$ , 其中 $x$ 定义域为 $D$
> 拉格朗日函数： $L(x, λ) = f(x) + λ m(x)$
> 原问题的无约束形式： $\min_x{\max_λ{L(x, λ)}}, s.t. λ \ge 0$
> 对偶问题： $\max_λ{\min_x{L(x, λ)}}, s.t. λ \ge 0$
> 定义：
> - $\tilde{p}$ 是原问题的解， $\tilde{d}$ 是对偶问题的解
> - $u=m(x)$ ， $t=f(x)$ 
> - 区域 $G = \\{(u, t) | x \propto D\\}$
> 
> 则有：
> - $\tilde{p} = inf\\{t | (u, t) \propto D, u \le 0\\} $
> - $\tilde{d} = \max_λ{\min_x{L(x, λ)}} := \max_λ{g(λ)},  其中： g(λ) = inf\\{t+λu | (u, t) \propto G, λ \ge 0\\}$
> 
> 如图，在图上表示区域G，注意G可以非凸，其中阴影部分为约束条件 $u \le 0$，可以看出：
> - $\tilde{p}$ 是满足 $u \le 0$ 下， $t$ 能取到的最小值
> - 当固定 $λ$ 时， $g(λ)$ 是直线 $t + λu = a$ 和 $t$ 轴交点，又 $(u, t) \propto G$ ，且斜率 $-λ$ 小于0，故 $g(λ)$ 与 $G$ 相切时，取到下确界。现在，考虑  $λ$ 可变，要取到最大的 $g(λ)$ ，当且仅当 $t + λu = a$ 同时和 A、B相切，此时 $\tilde{d}$ 为 $g(λ)$ 和 竖轴交点。可以直观地看出 $\tilde{d} \le \tilde{p}$ 。
> ![image](https://github.com/iLovEing/notebook/assets/109459299/615d0b21-1b24-444a-96fe-14ea54a37a0c)


> 补充说明4：slater条件
> 在*补充说明3*中，如果G是凸的，可以直观地感觉到 $\tilde{d} \= \tilde{p}$ 可以成立
> ![image](https://github.com/iLovEing/notebook/assets/109459299/c9e06d12-9122-4ba2-ade4-6cb71583bb98)
> 数学上，凸函数不是强对偶的充分条件，还需要满足一些其他条件，比如slater条件：
> $\exists \hat{x} \propto relint-D， s.t. \forall i=1, 2, ..., M， m_i(\hat(x)) < 0$
> 放松的slater：若 $M$ 中有 $k$ 个仿射函数，则不需要校验这 $k$ 个条件（ $m_i(\hat(x)) < 0， i=1,...,k$ ）

#### 3.3 KKT条件
如果一个凸优化问题满足强对偶性，则可以引出KKT条件，借助KKT条件求解问题。
1. 先写出原问题和对偶问题
拉格朗日函数： $L(x, λ, η)$
原问题（无约束）： $\min_x{\max_λ{L(x, λ, η)}}, s.t. λ \ge 0$
对偶问题： $\max_λ{\min_x{L(x, λ, η)}}, s.t. λ \ge 0$
**定义**： $\tilde{p}$ 为原问题的解，在 $\tilde{x}$ 处取得； $\tilde{d}$ 为对偶问题的解，在 $\tilde{λ}, \tilde{η}$ 处取得。

2. KKT条件
    1. 可行性条件（由定义的约束直接得出）:
        - $m_i(\tilde{x}) \le 0$
        - $n_j(\tilde{x}) = 0$
        - $\tilde{λ} \ge 0$
    2. 互补松弛条件： $\tilde{λ}_i m_i = 0$
    3. 梯度为0： $\frac{\partial L}{\partial x} |_{x=\tilde{x}} = 0$

3. KKT条件证明
显然有：
![image](https://github.com/iLovEing/notebook/assets/109459299/9df2fb48-df8e-45a0-84a2-9a5e513a8738)
----- ***公式(e)***  

观察式e的两个不等号，当优化问题满足强对偶性时，不等号取等：
- 第一个不等号取等：梯度为0条件 $\frac{\partial L}{\partial x} |_{x=\tilde{x}} = 0$
- 第二个不等号取等：互补松弛条件： $\tilde{λ}_i m_i = 0$
至此，KKT条件的2、3证毕。

> 补充说明5：关于引入对偶问题求解优化问题
> 主要目的是，如果优化问题满足强对偶性，则可以使用KKT条件，降低问题求解复杂度（或使问题从不可解变为可解？）
> 这一章的**整体逻辑**是：凸优化+slater条件 -> 强对偶性（原问题和对偶问题同解） -> KKT条件


---

> 公式latex附录
> (a): \left\{\begin{matrix}\min_x{f(x)}, x\propto R^P \\s.t. m_i(x) \le 0, i=1, 2...M \\s.t. n_j(x) = 0, j=1, 2...N\end{matrix}\right.
> (b): L(x,\lambda,\eta) = f(x) + \sum_{i}^{M}\lambda_i m_i + \sum_{j}^{N}\eta_j m_j，x\propto R^P
> (c): \left\{\begin{matrix} \min_x{\max_{\lambda,\eta} L(x,\lambda,\eta)}  \\ s.t. \lambda_{i} \ge 0\end{matrix}\right.
> (d): \left\{\begin{matrix} \max_{\lambda,\eta}{\min_x L(x,\lambda,\eta)}  \\ s.t. \lambda_{i} \ge 0\end{matrix}\right.
> (e): \tilde{d} = \max_{\lambda, \eta}{\min_x{L(x, \lambda, \eta)}} = \min_x{L(x, \tilde{\lambda}, \tilde{\eta})} {\color{Red} \le}  L(\tilde{x}, \tilde{\lambda}, \tilde{\eta}) = f(\tilde{x}) + \sum_{i}^{M}\tilde{\lambda}_i m_i + 0 {\color{Red} \le}  f(\tilde{x}) = \tilde{p}

---

## SVM
> SVM有三宝：间隔、对偶、核技巧

### 1. SVM要解决的问题：寻找“最好”的分类平面
考虑一个二分类问题：
![image](https://github.com/iLovEing/notebook/assets/109459299/9e901326-67f8-4c62-9a5f-572907fb70af)
寻找一个超平面 $f(w, b) = sign(w^Tx+b)$ 将样本正确分类，显然，在这样一个简单例子中，这样的平面有无数个，问题在于。哪一个最好？
如果只有正负样本各一个，直觉上这个超平面应该是应该是两个点的垂直平分线，同样的，在这个例子中，SVM给出了一个寻找方法：**最大化最小距离。**
如图，对于每一个可分割样本的超平面，可以计算距两端样本的最小距离d，svm的目标就是寻找能使d最大的超平面。
> 补充说明1：SVM是一个判别模型，而非概率模型。

### 2. SVM的数学表达
设样本点为 $\\{ (x_i, y_i) \\} _{i=1}^{N} ，x \propto R^P，y_i \propto \\{ -1, +1\\}$ ，超平面为 $f(w, b) = w^Tx+b$ 
超平面距离最近样本点的间距：
![image](https://github.com/iLovEing/notebook/assets/109459299/a1b4fc5f-3ad6-410e-86d8-0c1859bc7f1b)
----- ***记 式(a)***

则，SVM中”最大化最小距离“可以用数学表达出来：
![image](https://github.com/iLovEing/notebook/assets/109459299/ebc850bc-b15f-45cf-ad55-41b471513366)
----- ***记 式(b)***

对上式进一步解析：
1. 观察式(b)中的约束条件，可以改写为：
$\exists r>0，\min_i{y_i(w^Tx_i+b)}=r$
又因为超平面 $w^Tx_i+b$ 可以任意缩放，故该约束条件可以进一步改写为：
$\min_i{y_i(w^Tx_i+b)}=1$ ，即 $y_i(w^Tx_i+b) \ge 1$
2. 将1中的约束条件带入优化方程：
$\max_{w, b}{\min_{x_i}{\frac{ | w^Tx_i+b | }{ ||w|| }}} = \max_{w, b}{\min_{x_i}{\frac{ y_i(w^Tx_i+b) }{ ||w|| }}} = \max_{w, b}{\frac{ 1 }{ ||w|| }}\min_{x_i}{ y_i(w^Tx_i+b) } = \max_{w, b}{\frac{ 1 }{ ||w|| }} = \min_{w, b}{ ||w|| } = \min_{w, b}{\frac{1}{2} w^Tw}$
这里，第一个等号用 $y_i$ 相乘抵消绝对值，第三个等号是将约束条件代入的结果，最后一个等号中1/2是为了后面计算方便加上的。
3. 因此，SVM最终可以写成如下带约束优化问题：
![image](https://github.com/iLovEing/notebook/assets/109459299/8712f7c6-85e6-4160-a51e-ab5275cbc967)
----- ***记 式(c)***

### 3. SVM求解
1. 根据先前知识的优化理论，将式(c)转化为无约束优化问题，并写出对偶问题：
    - 拉格朗日函数： $L(w, b, λ) = \frac{1}{2}w^Tw + \sum_{i=1}^{N}\λ_i[1-y_i(w^Tx_i+b)]$
    - 无约束优化问题： $\min_{w, b}{\max_λ{L(w, b, λ)}}，s.t.λ_i \ge 0$
    - 对偶问题： $\max_λ{\min_{w, b}{L(w, b, λ)}}，s.t.λ_i \ge 0$

2. 代入KKT条件
    1. 根据梯度为0条件可得：
        - $\frac{\partial L}{\partial b} = 0 \Longrightarrow  \sum_{i}^{N}λ_iy_i = 0$  -----***记 式(d)***
        - $\frac{\partial L}{\partial w} = 0$ ，联立式（d）可解得 $\tilde{w} = \sum_{i}^{N}λ_iy_ix_i$ ----- ***记 式(e)***
    2. 根据互补松弛条件可得： $λ_i[1-y_i(w^Tx_i+b)] = 0$ ----- ***记 式(f)***
    观察式（f），结合优化问题本身的两个约束条件： $λ_i \ge 0，1-y_i(w^Tx_i+b) \le 0$ ，可推测大部分 $λ_i $ 值均为0，只对少部分满足 $1-y_i(w^Tx_i+b)=0$ 的样本点， $λ_i$ 有值，这些样本点就被称为**支持向量**。
    同理，选一支持向量 $(x_k, y_k)$ ，代入上式可解得： $\tilde{b} = y_k - w^Tx_k$ 。
3. 最终结果
对于本章提出的分类问题，SVM给出的超平面为 $f(w, b) = sign(w^Tx+b)$  ， $w, b$ 由下式解出：
![image](https://github.com/iLovEing/notebook/assets/109459299/9366668e-41cf-47c2-864a-9352790b3f0e)
----- ***记 式(f)***  

其中 $λ_i$ 的值由优化问题 $\max_λ{\min_{w, b}{L(w, b, λ)}}，s.t.λ_i \ge 0$ 解出，求解该优化问题有一些经典算法，比如。


### 4. soft-margin SVM

---


> 公式latex附录
> (a): margin(w, b) = \min_{w,b,x_i}{d(w, b, x_i)} = \min_{w,b,x_i}{\frac{\left | w^Tx_i+b \right | }{\left \| w \right \| } }
> (b): \left\{\begin{matrix} \max_{w, b}{\min_{x_i}{\frac{\left | w^Tx_i+b \right | }{\left \| w \right \| } }}
\\s.t. y_i(w^Tx_i + b) > 0，i=1,2,...,N\end{matrix}\right.
> (c): \left\{\begin{matrix} \min_{w, b}{\frac{1}{2} w^Tw}\\s.t. 1-y_i(w^Tx_i + b) \le 0，i=1,2,...,N\end{matrix}\right.
> (f): \left\{\begin{matrix}\tilde{w} = \sum_{i}^{N}\lambda_iy_ix_i\\\tilde{b} = y_k - w^Tx_k，(x_k,y_k)为支持向量\end{matrix}\right.

---

## kernel SVM

挖个坑，写完核方法填坑。
（其实解出w和b已经能看出来了）