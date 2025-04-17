# Builder Image
ARG JETPACK_VERSION="r36.4.0"

FROM pipeline-scheduler:${JETPACK_VERSION}

USER root
RUN pip install -U jetson-stats --force
WORKDIR /home/soulsaver/FCPO/build

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone