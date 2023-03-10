# [clustering](https://github.com/iLovEing/notebook/issues/9)

# 聚类算法

记录传统学习中常用聚类算法

---

## Kmeans

---

## DBSCAN
DBSCAN（Density-Based Spatial Clustering of Applications with Noise） 

___
### DBSCAN算法原理
1.**两个参数**：
- 领域半径：**Eps**；
- 成为核心对象的在领域半径内的最少点数：**MinPts**。

2.**几个重要概念**：
- Eps领域 (Eps-neighborhood of a point)：点p的**Eps**邻域，记为 $N_{Eps}(p)$，定义为 $N_{Eps}(p) = \lbrace q\in D | dist(p,q)≤Eps \rbrace$
- 核心对象 (core points)：如果给定对象**Eps**领域内的样本点数大于等于**MinPts**，则称该对象为核心对象。
- 直接密度可达 (directly density-reachable):
若：
    1. $p\in N_{Eps}(q)$
    2. $|N_{Eps}(q)| ≥ MinPts$ 
则称对象p从核心对象q是直接密度可达的。
- 密度可达 (density-reachable)：对于对象p1,p2, …, pn，令p1= q，pn = p。若pi+1是从pi直接密度可达的，则称p是从q密度可达的。
- 密度相连 (density-connected)：对于点p和点q，若点p点q都是从点o密度可达的，则称点p和点q密度相连。
- 簇 (cluster)：对于数据集D，若C是其中一个簇，C中的点需要满足以下两个条件：
1）∀ p, q，如果 p∈ C且q 是从p密度可达的，则 q∈ C。
2）∀ p, q ∈ C，p和q是密度相连的。
- 噪音(noise)：不属于任何簇的点为噪音数据。

3.**算法流程**
1. 给定领域半径：**Eps**和成为核心对象的在领域半径内的最少点数：**MinPts**。
2. 从任意点p开始，将其标记为”visited“，检查其是否为核心点（即p的**Eps**邻域至少有**MinPts**个对象），如果不是核心点，则将其标记为噪声点。否则为p创建一个新的簇C，并且把p的**Eps**邻域中的所有对象都放到候选集合N中。
3. 迭代地把N中不属于其它簇的对象添加至C中，在此过程中，对于N中标记为”unvisited"的对象p‘，把它标记为“visited”，并且检查它的**Eps**邻域，如果p’也为核心对象，则p’的**Eps**邻域中的对象都被添加至N中。继续添加对象至C中，直到C不能扩展，即直到N为空。此时，簇C完全生成。
4. 从剩余的对象中随机选择下一个未访问对象，重复3）的过程，直到所有对象都被访问。


算法比较简单，就是界定邻居+寻找邻居的过程，有点类似于晶格生长。给张结果图感受一下
![image](https://user-images.githubusercontent.com/109459299/224216265-6d53d88f-5dc4-45f7-9bd6-286af6cffc12.png)

___
### 参数选择
1. 领域半径：**Eps**的选取方法（k-distance函数）
    1. 选取k值，建议取k为2*维度-1。（其中维度为特征数）
    2. 计算并绘制k-distance图。（计算出每个点到距其第k近的点的距离，然后将这些距离从大到小排序后进行绘图。）
    3. 找到拐点位置的距离，即为**Eps**的值。
如下图所示：
![image](https://user-images.githubusercontent.com/109459299/224216528-23ddb6ad-5134-477e-a2c4-0a9fc7c33ef4.png)

2. **MinPts**的选取方法
**MinPts**的取值为上述k值加1，即： $MinPts = k + 1$

___
### 代码示例

	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.cluster import DBSCAN
	from sklearn.cluster import KMeans
	from sklearn.neighbors import NearestNeighbors

	X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
										  noise=.05)
	X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
				   random_state=9)

	X = np.concatenate((X1, X2))
	plt.scatter(X[:, 0], X[:, 1], marker='o')
	plt.show()

	nbrs = NearestNeighbors(n_neighbors=4).fit(X)
	distances, indices = nbrs.kneighbors(X)
	dis = distances[:, 3]
	dis = -np.sort(-dis)
	fig, ax = plt.subplots()
	ax.plot(np.array(range(len(dis))), dis, linewidth=2.0)
	plt.show()

	y_pred1 = DBSCAN(eps = 0.09, min_samples = 4).fit_predict(X)
	plt.scatter(X[:, 0], X[:, 1], c=y_pred1)
	plt.show()
	np.unique(y_pred1)


---

## 层次聚类