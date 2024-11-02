# Builder Image
ARG JETPACK_VERSION="r35.2.1-py3"

FROM pipeline-scheduler:${JETPACK_VERSION}

USER root
RUN pip install -U jetson-stats --force
WORKDIR /home/soulsaver/PipelineScheduler/build

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#RUN apt install libspdlog-dev libpqxx-dev -y
#RUN mkdir /app/build -p
#COPY ./cmake /app/cmake
#COPY ./src /app/src
#COPY ./libs /app/libs
#COPY ./CMakeLists.txt /app/CMakeLists.txt
#WORKDIR /app/build
#RUN cmake ..
#RUN make -j 8
#COPY ./jsons /app/jsons
