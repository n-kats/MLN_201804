from ubuntu:16.04

RUN set -ex \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    software-properties-common \
    language-pack-ja-base language-pack-ja \
    xvfb ffmpeg python3-opengl zlib1g-dev cmake \
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
  && pip install -U --no-cache-dir \
    wheel setuptools \
    numpy \
    gym \
    "gym[atari]" \
    chainerrl

WORKDIR /work
ENTRYPOINT xvfb-run -s "-screen 0 1400x900x24" bash
