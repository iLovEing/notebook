# [Fourier Transform](https://github.com/iLovEing/notebook/issues/11)

从傅里叶级数到FFT

---

## 傅里叶级数

将周期函数在三角函数系上展开，得到傅里叶级数：
![image](https://user-images.githubusercontent.com/109459299/224723347-fd730310-bdc7-4a32-a688-a05c785441da.png)

**注意**
这里只证明了三角函数系的正交性，没有证明完备性。只是单纯使用可以不关注完备性证明，后续有机会填坑~


---

## 傅里叶变换

傅里叶级数的基底是无穷可列的，但是周期不连续，无法用于非周期函数。
将周期函数的周期延拓到无穷，使得三角函数基地周期变为连续，即傅里叶变换。而非周期函数可以看作周期无穷大的周期函数，使得傅里叶变换使用场景得到扩展。
![image](https://user-images.githubusercontent.com/109459299/224724402-6d5a40ff-6410-4a2c-a4bf-77712702c671.png)
