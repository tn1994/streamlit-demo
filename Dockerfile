FROM python:3.10.7

COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r requirements.txt