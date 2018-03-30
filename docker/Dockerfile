from ubuntu:16.04

RUN set -ex \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    software-properties-common \
    language-pack-ja-base language-pack-ja \
    sudo xvfb ffmpeg python3-opengl zlib1g-dev cmake \
  && add-apt-repository -y ppa:jonathonf/python-3.6 \
  && apt-get update \
  && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

RUN ln -sf /usr/bin/python3.6 /usr/bin/python3 \
  && ln -sf /usr/bin/python3.6 /usr/bin/python

RUN python -m pip install -U pip \
  && pip install -U \
    wheel setuptools \
    numpy \
    pillow \
    scipy \
    scikit-learn \
    tensorflow \
    gym \
    "gym[atari]"

WORKDIR /

ENV USERNAME developer
RUN useradd -m $USERNAME \
  && echo "$USERNAME:$USERNAME" | chpasswd \
  && usermod --shell /bin/bash $USERNAME \
  && usermod -aG sudo $USERNAME \
  && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && usermod  --uid 1000 $USERNAME \
  && groupmod --gid 1000 $USERNAME
# Replace 1000 with your user/group id

WORKDIR /work
