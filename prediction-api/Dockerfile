# 建議使用 Python 3.11 slim 版
FROM python:3.11-slim

# 系統相依與時區（可選）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 預先放 requirements 以利快取
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 放入程式碼
COPY . /app

# 建模輸出與運行時會用到的資料夾
RUN mkdir -p /models
ENV PYTHONUNBUFFERED=1

# 預設啟動 prediction API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
