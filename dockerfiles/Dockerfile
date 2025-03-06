# Use the base image
FROM 143.248.55.42:5000/nvcr.io/nvidia/l4t-ml:r35.1.0-py3

# Install required packages
RUN apt update && apt install -y \
        build-essential \
        wget \
        vim \
        && rm -rf /var/llib/apt/list/*

RUN wget https://cmake.org/files/v3.25/cmake-3.25.2-linux-aarch64.sh -q -O /tmp/cmake-install.sh
RUN chmod u+x /tmp/cmake-install.sh
RUN mkdir /opt/cmake-3.25.1
RUN /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.25.1
RUN ln -s /opt/cmake-3.25.1/bin/* /usr/local/bin


# Install gRPC 1.62.0
ARG GRPC_INSTALL_DIR=/grpc
RUN mkdir -p ${GRPC_INSTALL_DIR}
RUN apt-get update
RUN apt-get -y install autoconf libtool
RUN git clone --recurse-submodules -b v1.62.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc /grpc-src
RUN cd /grpc-src && \
    mkdir -p third_party/abseil-cpp/cmake/build && \
    cd third_party/abseil-cpp/cmake/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ../.. && \
    make "-j${GRPC_CPP_DISTRIBTEST_BUILD_COMPILER_JOBS}" install

RUN cd /grpc-src && \
    mkdir -p "third_party/cares/cares/cmake/build" && \
    cd third_party/cares/cares/cmake/build && \
    cmake -DCMAKE_BUILD_TYPE=Release ../.. && \
    make "-j${GRPC_CPP_DISTRIBTEST_BUILD_COMPILER_JOBS}" install

RUN cd /grpc-src && \
    mkdir -p "third_party/protobuf/cmake/build" && \
    cd "third_party/protobuf/cmake/build" && \
    cmake -Dprotobuf_BUILD_SHARED_LIBS=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -Dprotobuf_ABSL_PROVIDER=package ../.. && \
    make "-j4" install


RUN cd /grpc-src && \
    mkdir -p "third_party/re2/cmake/build" && \
    cd "third_party/re2/cmake/build" && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ../.. && \
    make "-j${GRPC_CPP_DISTRIBTEST_BUILD_COMPILER_JOBS}" install

RUN cd /grpc-src && \
    mkdir -p "third_party/zlib/cmake/build" && \
    cd "third_party/zlib/cmake/build" && \
    cmake -DCMAKE_BUILD_TYPE=Release ../.. && \
    make "-j${GRPC_CPP_DISTRIBTEST_BUILD_COMPILER_JOBS}" install

RUN apt-get update && apt-get install -y libssl-dev

RUN cd /grpc-src && \
    git submodule update --init && \
    mkdir -p cmake/build && \
    cd cmake/build && \
    cmake -DgRPC_INSTALL=ON                \
              -DCMAKE_BUILD_TYPE=Release       \
              -DgRPC_ABSL_PROVIDER=package     \
              -DgRPC_CARES_PROVIDER=package    \
              -DgRPC_PROTOBUF_PROVIDER=package \
              -DgRPC_RE2_PROVIDER=package      \
              -DgRPC_SSL_PROVIDER=package      \
              -DgRPC_ZLIB_PROVIDER=package     \
              -DBUILD_DEPS=ON \
              -DCMAKE_INSTALL_PREFIX=$GRPC_INSTALL_DIR \
      ../.. && \
    make -j 8 && \
    make install
ENV PATH="${PATH}:${GRPC_INSTALL_DIR}/bin"

# Install Boost
RUN apt update
RUN apt-get install build-essential g++ python3-dev autotools-dev libicu-dev libbz2-dev libboost-all-dev -y

# Install spdlog
RUN apt install libspdlog-dev libpqxx-dev -y

# Install posgresql-client
RUN apt install postgresql-client -y

# Store load time model
ARG upload_model="upload_model_orn.txt"
ARG download_model="download_model_orn.txt"
COPY ${upload_model} /ltmodel
COPY ${download_model} /ltmodel

# Install LightGBM
ARG LIGHTGBM_SRC_DIR=/lightgbm-src
RUN mkdir -p ${LIGHTGBM_SRC_DIR}
RUN cd lightgbm-src
RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    cmake -B build -S . -DCMAKE_INSTALL_PREFIX=/usr && \
    cmake --build build -j4 && \
    cmake --install build


RUN apt install gdb -y

ARG user=soulsaver
ARG host_uid=1000
ARG host_gid=1000

RUN groupadd -g $host_gid ${user}
RUN useradd -u $host_uid -g ${user} ${user}
RUN usermod -aG video ${user}
RUN echo '%sudo ALL(ALL:ALL) ALL' >> /etc/sudoers
RUN mkdir /home/${user}
RUN chown -R ${user}:${user} /home/${user}

USER ${user}

ENV HOME=/home/${user}


CMD [ "/bin/bash" ]
