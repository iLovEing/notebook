# [Fourier Transform](https://github.com/iLovEing/notebook/issues/11)

傅里叶变换各种信号处理的基础，可以将信号从时域转换到频域。

![image](https://github.com/iLovEing/notebook/assets/109459299/c4554a38-ee1e-4af7-ae35-525dd33afe13)
## 章节
- [x] 傅里叶级数（FS, Fourier Series）
- [x] 傅里叶变换（FT, Fourier Transform）
- [ ] 离散傅里叶级数（DFS, Discrete Fourier Series）
- [ ] 离散傅里叶变换（DFT, Discrete Fourier Transform）
- [ ] 快速傅里叶变换（FFT, Fast Fourier Series）
- [ ] kiss_fft: 一个较高效的fft代码实现

---

## 傅里叶级数

级数可以理解为无穷求和，傅里叶级数就是将**周期函数**在三角函数系上展开。类比线性代数，三角函数系就是线性空间的一组正交基，该空间维度为无穷，函数在三角函数系上展开就是向量在基底上线性表出，两者过程基本一致。

### 1. 用三角函数系表示周期函数
假设一周期函数 $f(t)$ ，其周期为 $T$ ，将 $f(t)$ 变换到三角函数为基地的空间，得到傅里叶级数：
![image](https://github.com/iLovEing/notebook/assets/109459299/3724faf6-e2c9-4f5a-85a6-5531e8f19ef6)
这里可能会有两点疑惑：
    1. 有些地方写的是 $a_0 + ...$ ，求和从1开始；两者是一样的，这里求和从0开始，当 $n=0$ 时， $sin0$ 值为0。
    2. 为什么只有频率参数，没有相位？这里每一组频率有两个分量，包含 $\cos、\sin$ 其实已经包含相位信息，逆向运用三角函数展开式可以化成 $\sin(wt+\phi)$ 的形式。

### 2. 三角函数基底的正交性
两个周期函数之间的分量计算，取函数乘积在一个周期内的积分： 
$\int_{-\frac{T}{2} }^{\frac{T}{2}} \cos(nwt)\cos(mwt)dt$
$= \frac{1}{2} \int_{-\frac{T}{2} }^{\frac{T}{2}} [\cos(m+n)wt + \cos(m-n)wt]dt $
$=$
$(when: m=n)： \frac{1}{2} \int_{-\frac{T}{2} }^{\frac{T}{2}} \cos0 dt = \frac{T}{2}$
$(when: m\ne n)： \frac{1}{2} \int_{-\frac{T}{2} }^{\frac{T}{2}} \cos0 dt = \frac{T}{2}$
这里只证明了三角函数系的正交性，没有证明收敛性。










---

> latex附录
> 式(a)： f(t) = \sum_{0}^{+\infty }[a_n\cos(nwt)+b_n\sin(nwt)]，w=\frac{2\pi }{T} 

---

## 傅里叶变换

傅里叶级数的基底是无穷可列的，但是周期不连续，无法用于非周期函数。
将周期函数的周期延拓到无穷，使得三角函数基地周期变为连续，即傅里叶变换。而非周期函数可以看作周期无穷大的周期函数，使得傅里叶变换使用场景得到扩展。
![image](https://user-images.githubusercontent.com/109459299/224724402-6d5a40ff-6410-4a2c-a4bf-77712702c671.png)


---

## FFT

---

## KISS_FFT