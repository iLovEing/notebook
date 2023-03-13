# [SVM](https://github.com/iLovEing/notebook/issues/15)

support vector machine-支持向量机
- 约束优化问题
- svm


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


