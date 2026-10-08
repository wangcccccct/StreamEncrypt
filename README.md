# 低空经济场景下的图传画面隐私保护实验

本项目探索低空经济相关场景中的图像画面隐私保护，重点面向无人机图传、低空巡检、航拍采集等应用设想。程序以 Python 和 FFmpeg 为基础，对视频文件逐帧进行频域视觉扰乱，并支持使用相同密钥尝试恢复画面。

本项目受 Shi 等人关于无人机安全视频通信的研究所启发，但没有复现论文中的算法。论文采用多轮混沌置乱与扩散，并报告了无人机平台上的实验；本项目使用独立的频域变换方法，目前仅为处理本地图片或视频文件的实验原型，未接入无人机设备、无线图传链路或实时传输系统。

> **适用范围说明：** 本程序处理视频画面，不处理音频、字幕或其他数据流。它用于探索低空经济场景下的画面隐私保护，不是经过完整密码分析和工程验证的生产级加密产品，也不提供认证、完整性保护或密钥协商。默认视频编码为有损编码，恢复画面可能与原始画面存在像素差异。

## 论文参考

Shi 等人，《基于异构并行计算的实时混沌视频加密及其在无人机安全通信中的应用》，《混沌、孤子与分形》，第 181 卷，文章编号 114681，2024 年。[论文页面与摘要](https://doi.org/10.1016/j.chaos.2024.114681)

引用该论文是为了说明本项目的应用方向受到相关无人机通信研究启发；两者的算法实现和实验条件不同。

## 环境要求

- Python 3.10 或更新版本
- FFmpeg（需包含 `ffmpeg` 和 `ffprobe`）
- Python 软件包：NumPy、SciPy

macOS 可使用 Homebrew 安装 FFmpeg：

```bash
brew install ffmpeg
```

## 安装

在项目目录中执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy scipy
```

## 处理视频

使用 V11 程序处理视频画面：

```bash
.venv/bin/python mspe_v11_media.py encrypt input.mp4 encrypted.mp4 "your-secret-key"
```

使用相同密钥尝试恢复画面：

```bash
.venv/bin/python mspe_v11_media.py decrypt encrypted.mp4 restored.mp4 "your-secret-key"
```

程序根据输出文件扩展名选择 FFmpeg 容器。常用扩展名包括 `.mp4`、`.mov` 和 `.mkv`。

## 处理单张图片

V11 支持处理静态图片。输入和输出可使用 PNG、JPEG 等支持的图片格式：

```bash
.venv/bin/python mspe_v11_media.py encrypt photo.png photo_encrypted.png "your-secret-key"
.venv/bin/python mspe_v11_media.py decrypt photo_encrypted.png photo_restored.png "your-secret-key"
```

## 可选参数

运行以下命令查看全部参数：

```bash
.venv/bin/python mspe_v11_media.py --help
```

常用参数包括帧流水线并行数、每帧傅里叶变换线程数、编码器、视频质量参数和编码预设。处理与恢复时应保持变换参数一致；默认参数已经匹配。

感谢 [Linux.do 社区](https://linux.do) 的支持与鼓励。
