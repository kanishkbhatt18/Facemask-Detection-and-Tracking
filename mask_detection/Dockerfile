FROM quay.io/jupyter/pytorch-notebook

WORKDIR /facemask

USER root

RUN apt-get update && apt-get install -y \ 
    libx11-6 libxext-dev libxrender-dev \
    libxinerama-dev libxrandr-dev libxcursor-dev libxtst-dev \
    libsm6 libxext6 \
    tk-dev && rm -rf var/lib/apt/lists/*

USER jovyan
    
ENV DISPLAY=:0

COPY detection.ipynb yolo.ipynb detection.py requirements.txt sort.py 2_class.pt best_7.pt /facemask/

RUN pip install -r requirements.txt







