FROM python:3.10-slim

WORKDIR /code
COPY requirements.txt .

# 🔧 使用 HTTP 版本避免 SSL 問題
RUN pip install --no-cache-dir --upgrade pip \
    -i http://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --timeout 300 --retries 10

RUN pip install --no-cache-dir -r requirements.txt \
    -i http://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --timeout 300 --retries 10

COPY ./app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]