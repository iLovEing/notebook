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

级数可以理解为无穷求和
将周期函数在三角函数系上展开，得到傅里叶级数：


1. 只有频率，没有相位？

这里只证明了三角函数系的正交性，没有证明收敛性。


---

## 傅里叶变换

傅里叶级数的基底是无穷可列的，但是周期不连续，无法用于非周期函数。
将周期函数的周期延拓到无穷，使得三角函数基地周期变为连续，即傅里叶变换。而非周期函数可以看作周期无穷大的周期函数，使得傅里叶变换使用场景得到扩展。
![image](https://user-images.githubusercontent.com/109459299/224724402-6d5a40ff-6410-4a2c-a4bf-77712702c671.png)


---

## FFT

---

## KISS_FFT