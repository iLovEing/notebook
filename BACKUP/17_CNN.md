# [CNN](https://github.com/iLovEing/notebook/issues/17)

卷积神经网络

---

## 基本结构
![image](https://user-images.githubusercontent.com/109459299/224759392-7b42c8a4-4f09-4441-a56b-362df452b08d.png)
![image](https://user-images.githubusercontent.com/109459299/224759642-958856d0-70db-4546-b277-5c4faefd6daf.png)
![image](https://user-images.githubusercontent.com/109459299/224759777-e97df8dd-1a3d-4aea-a54c-ed342062b7ba.png)


---

## 经典结构
- LeNet，激活函数为sigmod
![image](https://user-images.githubusercontent.com/109459299/225285118-1805f341-d18f-48f7-b340-c1084486b9e6.png)

- AlexNet，激活函数为ReLU, 在全连接层用了dropout
![image](https://user-images.githubusercontent.com/109459299/225285207-c110022f-4c8b-480e-8835-56d4cf196cad.png)

- VGG，使用VGG块，每个块中有若干个相同的卷积层和一个池化层
![image](https://user-images.githubusercontent.com/109459299/225285361-8de76aa0-79d0-40c7-b000-369287c291df.png)

- NiN，引入1*1卷积层，相当于像素全连接
![image](https://user-images.githubusercontent.com/109459299/225285514-abd99adc-7d49-4af9-9500-91e73943a77e.png)

- GoogLeNet
![image](https://user-images.githubusercontent.com/109459299/225285631-86f58c15-f49f-495d-9d7f-0b64f5450163.png)
![image](https://user-images.githubusercontent.com/109459299/225285659-7929b2c5-5cf2-404a-a8e4-3455fbcfd09f.png)

---

## Batch Normal