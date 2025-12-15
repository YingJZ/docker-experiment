FROM docker.m.daocloud.io/library/python:3.9-slim

WORKDIR /app

# 设置国内源
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch CPU 版本 (使用官方 CPU 轮子源)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY . /app

CMD ["python", "infer.py"]
