ARG BASE_IMAGE
FROM ${BASE_IMAGE} as dev
############################################

SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

RUN pip3 install gtsam