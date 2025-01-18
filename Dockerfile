FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workingspace

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-distutils python3-pip build-essential git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --config python3 && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shell/predict_independent.sh .
COPY shell/predict_multitask.sh .
COPY shell/train_independent.sh .
COPY shell/train_multitask.sh .

RUN chmod +x predict_independent.sh
RUN chmod +x predict_multitask.sh
RUN chmod +x train_independent.sh
RUN chmod +x train_multitask.sh

CMD ["bash"]
