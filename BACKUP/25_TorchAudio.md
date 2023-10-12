# [TorchAudio](https://github.com/iLovEing/notebook/issues/25)

[torchaudio官方文档](https://pytorch.org/audio/stable/index.html)

---

## Audio I/O
> [示例代码](www.test)
### 查看信息
- metadata = torchaudio.info(file_path)：打印metadata，可以获取采样率，通道数等信息

### 读取和保存
- waveform, sample_rate = torchaudio.load(file_path)：读取文件，得到波形数据（张量，通道*帧）和采样率可以指定 frame_offset 和 num_frames。
- torchaudio.save(file, waveform, sample_rate)：保存文件，可以指定编码格式等