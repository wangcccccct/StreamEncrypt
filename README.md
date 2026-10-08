# StreamEncrypt MSPE-V9

这个仓库提供一个基于 Python 和 FFmpeg 的视频画面加密/解密命令行程序：`mspe_v9_video.py`。V9 对画面做可逆的频域扰乱，并支持视频流式处理。

> **注意：** 这是实验性的感知扰乱，不是带认证的现代加密方案。程序只处理视频画面，不保留音频。默认 H.264 输出是有损编码；解密结果与输入画面会有少量像素差异。

## 环境要求

- Python 3.10 或更新版本
- FFmpeg（需包含 `ffmpeg` 和 `ffprobe`）
- Python 包：NumPy、SciPy

macOS 可用 Homebrew 安装 FFmpeg：

```bash
brew install ffmpeg
```

## 安装并启动

在仓库目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy scipy
```

加密视频：

```bash
.venv/bin/python mspe_v9_video.py encrypt input.mp4 encrypted.mp4 "your-secret-key"
```

解密视频时使用同一个密钥：

```bash
.venv/bin/python mspe_v9_video.py decrypt encrypted.mp4 restored.mp4 "your-secret-key"
```

程序会根据输出文件扩展名选择 FFmpeg 容器。常用输出扩展名包括 `.mp4`、`.mov` 和 `.mkv`。

## 处理单张图片

V9 的入口是视频处理器。FFmpeg 可将 PNG/JPEG 作为只有一帧的视频输入；输出使用视频容器（例如 MP4）：

```bash
.venv/bin/python mspe_v9_video.py encrypt photo.png photo_encrypted.mp4 "your-secret-key"
.venv/bin/python mspe_v9_video.py decrypt photo_encrypted.mp4 photo_restored.mp4 "your-secret-key"
```

如果需要恢复为 PNG，可从解密视频导出首帧：

```bash
ffmpeg -i photo_restored.mp4 -frames:v 1 photo_restored.png
```

## 可选参数

运行 `--help` 查看完整参数：

```bash
.venv/bin/python mspe_v9_video.py --help
```

常用参数包括 `--pipeline-workers`（并行处理的帧数）、`--workers`（每帧 FFT 线程数）、`--codec`、`--crf` 和 `--preset`。加密和解密时应保持变换参数一致；默认值已匹配。

## 本仓库运行示例

本仓库的截图已用 V9 完成单帧加密与解密，结果位于 `release/`：

- `release/cleanshot_v9_encrypted.mp4`
- `release/cleanshot_v9_decrypted.mp4`
- `release/cleanshot_v9_encrypted.png`（加密结果预览帧）
- `release/cleanshot_v9_decrypted.png`（解密结果预览帧）

重新处理时，将示例中的密钥字符串替换为你自己的密钥，并在解密时使用同一个值。
