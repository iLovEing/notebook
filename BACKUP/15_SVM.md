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

## 约束优化问题求解
先前知识，svm的约束优化问题求解思路：
1. 利用拉格朗日乘子法，将原问题转成无约束形式
2. 将原问题转化为对偶问题求解
3. 根据slater条件，原问题和对偶问题同解
4. 根据3，引入KKT条件
5. 已知算法求解最终问题

![image](https://user-images.githubusercontent.com/109459299/224753194-284f184a-b6f0-4b48-ade7-abd391c2a828.png)
![image](https://user-images.githubusercontent.com/109459299/224753398-4db6f84a-0ec4-427f-b4c6-0b13e3d42f4c.png)
![image](https://user-images.githubusercontent.com/109459299/224753526-a8e90401-d2f7-44ec-839e-d85038422fc6.png)
![image](https://user-images.githubusercontent.com/109459299/224753657-97a7e8fc-4044-433e-82b6-07e6022f6d85.png)
![image](https://user-images.githubusercontent.com/109459299/224753714-e390b5bc-0a03-451a-a99d-b884424aed43.png)




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