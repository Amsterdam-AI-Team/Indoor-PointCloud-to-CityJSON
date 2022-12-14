FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Amsterdam"

RUN mkdir /usr/app && cd /usr/app
WORKDIR /usr/app
RUN apt-get update -y && apt-get upgrade -y && apt-get install --no-install-recommends -y  \ 
    glpk-utils \
    libglpk-dev \
    glpk-doc  \ 
    libmpfr-dev \
    tzdata \
    build-essential \
    python3-dev \
    glpk-utils \
    libglpk-dev \
    glpk-doc \
    libmpfr-dev \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade jupyter

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]