# PipelineScheduler

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14789255.svg)](https://doi.org/10.5281/zenodo.14789255)

PipelineScheduler is a system which enables the highest performance in terms of throughput and latency for can find the **optimal workload distribution** to split the pipelines between the server and the Edge devices, and set the **optimal batch sizes** for them to ensure the best throughput and latency against challenges such as *content dynamics* and *network instability*.
PipelineScheduler also considers *resource contention* and is equipped with **inference spatiotemporal scheduling** to mitigate the adverse effects of *co-location interference*.

The current stage of PipelineScheduler is the implementation for our [IEEE PerCom 2025](https://www.percom.org/) full paper, titled **"OCTOPINF: Workload-Aware Real-Time Inference Serving for Edge Video Analytics"**, available [here](). Future improvements will be continually updated. Architectural diagram:

![overall-arch](/assets/overall-arch.png)


This repo is contributed and maintained by Thanh-Tung Nguyen, Lucas Liebe (equally), and other colleges at [CDSN Lab](http://cds.kaist.ac.kr) at KAIST.

When using our Code please cite the following paper:
```
@inproceedings{nguyen2025,
    author={Thanh-Tung Nguyen and Lucas Liebe and Tau-Nhat Quang and Yuheng Wu and Jinghan Cheng and Dongman Lee}
    title = {{OCTOPINF: Workload-Aware Real-Time Inference Serving for Edge Video Analytics}},
    booktitle = {The 23rd International Conference on Pervasive Computing and Communications (PerCom 2025)},
    year = {2025},
    publisher = {IEEE},
    month = march,
}
```

# Table of Contents

1. [Overview](#pipelinescheduler)  
2. [Implementation Architecture](#implementation-architecture)  
   * [Controller](#controller)  
   * [Device Agent](#device-agent)  
   * [Inference Container](#inference-container)  
     * [Container Agent](#container-agent)  
     * [Configurations](#configurations)  
   * [Knowledge Base](#knowledge-base)  
3. [Running ***PipelineScheduler***](#running-pipelinescheduler)  
   * [Installation](#installation)  
     * [Prerequisites](#prerequisites)  
     * [Inference Platform](#inference-platform)  
     * [Build & Compile](#build--compile)  
       * [Controller](#the-controller)  
       * [Device Agent](#the-device-agent)  
       * [Microservices Inside Each Container](#the-microservices-inside-each-container)  
   * [Preparing Data](#preparing-data)  
   * [Preparing Models](#preparing-models)  
     * [Build](#build)  
     * [Run Conversion](#run-conversion)  
     * [Edit Model Configuration Templates](#edit-model-configuration-templates)  
     * [Running Model Profiling](#running-model-profiling)  
   * [Running](#running)  
     * [Step 1: Running the Controller](#step-1-running-the-controller)  
     * [Step 2: Running the Device Agent](#step-2-once-the-controller-is-running-run-a-device-agent-on-each-device)  
4. [Extending ***PipelineScheduler***](#extending-pipelinescheduler)  
   * [Adding Models](#adding-models)  
   * [Local Optimizations](#local-optimizations)  
5. [Misc](#misc)  
6. [Citing Our Works](#citing-our-works)

# Implementation Architecture
PipelineScheduler is composed of 4 main components:

## Controller
The **Controller** is run as a separate C++ process to oversee the whole system.
It queries operational statistics from the Knowledge Base via the PostGreSQL API and issues commands to the **Device Agents** via custom gRPC APIs.

## Device Agent
Each workload handling device (including the Edge server) runs a **Device Agent** as a separate C++ process to manage and monitor the containers using Docker and other APIs (e.g., NVIDIA Driver API).

## Inference Container
EVA pipelines are organized into DAGs with each node is a container, which packages a DNN model and its pre/postprocessing logics.
Particularly, each container is its own pipeline of microservices which typically follows a structure of `[Receiver -> Preprocessor -> Batcher -> Inferencer -> Postprocessor -> Sender]`. Each microservice runs as a thread to ensure high parallelism.

The `Preprocessor` and `Postprocessor` can be cloned (or *horizontally scaled up*), as well as the batch size of `Inferencer` can be dynamically set to ensure the highest throghput while maintain strict service-level objective (i.e., end-to-end latency) compliance.

This design allows microservice to be replaced easily in a plug-and-place manner. For instane, if a TensorRT inferencer (the current version) is not suitable for the hardware, a new inferencer (e.g., ONNX-based) can be whipped up with minimal adaptation.

But other designs (e.g., monolithic) works as well as long as the endpoints for sending/receiving data are specific correctly.

### Container Agent
The **Container Agent** is a light-weight thread in charge of creating/deleting the microservices according to the instructions of the **Device Agent** and **Controller**.
It also collects operational stats inside the container and published them to designated metrics endpoints.

### Configurations
The **Container Agent** relies on a json configuration file specifying the details of microservices inside each container.
The current container configurations are store [here](/jsons).
It is worth taking a look at their structures before proceeding to the next part.

Besides general metadata like:
```json
    "cont_experimentName": "prof",
    "cont_systemName": "ppp",
    "cont_pipeName": "traffic",
    "cont_taskName": "retina1face",
    "cont_hostDevice": "server",
    "cont_hostDeviceType": "server",
    "cont_name": "retinaface_0",
```

The microservice details are defined under `"cont_pipeline"`. This is what the example of the `Preprocessor` for model Retina Face.
```json
{
    "msvc_name": "preprocessor",
    "msvc_numInstances": 1,
    "msvc_concat": 1,
    "msvc_idealBatchSize": 1,
    "msvc_dnstreamMicroservices": [
        {
            "nb_classOfInterest": -1,
            "nb_commMethod": 3,
            "nb_link": [
                ""
            ],
            "nb_maxQueueSize": 100,
            "nb_name": "batcher",
            "nb_expectedShape": [
                [
                    3,
                    576,
                    640
                ]
            ]
        }
    ],
    "msvc_dataShape": [
        [
            -1,
            -1
        ]
    ],
    "msvc_pipelineSLO": 999999,
    "msvc_type": 1000,
    "msvc_upstreamMicroservices": [
        {
            "nb_classOfInterest": -2,
            "nb_commMethod": 4,
            "nb_link": [
                ""
            ],
            "nb_maxQueueSize": 100,
            "nb_name": "receiver",
            "nb_expectedShape": [
                [
                    -1,
                    -1
                ]
            ]
        }
    ],
    "msvc_maxQueueSize": 100,
    "msvc_imgType": 16,
    "msvc_colorCvtType": 4,
    "msvc_resizeInterpolType": 3,
    "msvc_imgNormScale": "1/1",
    "msvc_subVals": [
        104,
        117,
        123
    ],
    "msvc_divVals": [
        1,
        1,
        1
    ]

},
```
Details on how to set the configurations can be found [here](/jsons/README).

## Knowledge Base
The Knowledge Base is a PostgreSQL (14) database which contains all the operational statistics.

# Running ***PipelineScheduler***
## Installation
### Prerequisites
To run the system, this following software must be installed on the host machines.
* CMake (or newer)
* Docker
* OpenCV
* gRPC
* Protobuf
* PostgreSQL.

Inside the container, it is also necessary to install inference software platforms (e.g., TensorRT, ONNX).

The specific software versions and commands for installation can be found taken from the [dockerfiles](/dockerfiles/), which are written to build inference container images. Since the current version is run on NVIDIA hardware (i.e., GPU and Jetson devices), most of the images are built upon NVIDIA container images published [here](https://catalog.ngc.nvidia.com/containers).

The build instructions can be found [here](dockerfiles/README).

### Inference Platform
The current versions of `Preprocessors, Postprocessors and Inferencer` are written for NVIDIA hardware, especially the `Inferencer`. But custom microservices can be written based on these with minimal adaptation.

The system is designed to be deployed on a Edge cluster, but can also be run on a single machine.
The first step is to build the source code, here you can use multiple options for instance to change the scheduling system.

### Build & Compile
* The **Controller**
    ```bash
    mkdir build_host && cd build_host
    cmake -DSYSTEM_NAME=[PPP, DIS, JLF, RIM] -DON_HOST=True -DDEVICE_ARCH=platform_name
    # Ours is PPP (standing for PipePlusPlus)
    # Platform name is amd64 or Jetson.
    make -j 64 Controller
    ```

* The **Device Agent** 
    ```bash
    mkdir build_host && cd build_host
    cmake -DSYSTEM_NAME=[PPP, DIS, JLF, RIM] -DON_HOST=True -DDEVICE_ARCH=platform_name
    # Ours is PPP (standing for PipePlusPlus)
    # Platform name is amd64 or Jetson.
    make -j 64 Device_Agent
    ```

* The microservices **inside each container**
    ```bash
    mkdir build && cd build
    cmake -DSYSTEM_NAME=[PPP, DIS, JLF, RIM] -DON_HOST=False -DDEVICE_ARCH=platform_name
    # Ours is PPP (standing for PipePlusPlus)
    # Platform name is amd64 or Jetson.
    make -j 64 Container_[name]
    # Name of the model. YoloV5 for instance.
    ```

## Preparing Data
The data is collected from publicly available streams on website like [www.earthcam.com](https://www.earthcam.com). The script for pulling the data can be found [here](/scripts/collect_dataset.sh).

## Preparing Models
Models need to be prepared according to fit the current hardware and software inference platforms. For NVIDIA-TensorRT, please build and use following script.

* Build
    ```bash
    mkdir build && cd build
    cmake -DSYSTEM_NAME=[PPP, DIS, JLF, RIM] -DON_HOST=False -DDEVICE_ARCH=platform_name
    # Ours is PPP (standing for PipePlusPlus)
    # Platform name is amd64 or Jetson.
    make -j 64 convert_onnx2trt
    ```

* Run conversion.
    ```bash
    ./onnx2trt --onnx_path [path-to-onnx-model-file] --min_batch [batch_size] --max_batch [batch_size] --precision 4
    # Set [batch_size] to the maximum batch size you want the model to handle. The actually avaialble batch sizes during run time will range from [1, batch_size]
    ```

* Edit Model Configuration Templates
* Running Model Profiling
    * This is only necessary for scheduling, the inference works without profiling.

## Running
* Step 1: Running the **Controller**.
    ```bash
    ./Controller --ctrl_configPath ../jsons/experiments/full-run-ppp.json
    ```
    * The guideline to set configurations for controller run is available [here](/jsons/experiments/README).
* Step 2: Once the **Controller** is running, run a **Device Agent** on each device.
    ```bash    
    ./DeviceAgent --name [device_name] --device_type [server, agx, nx, orinano] --controller_url [controller_ip_address] --dev_port_offset 0 --dev_verbose 1 --deploy_mode 1
    ```

# Extending ***PipelineScheduler***
## Adding Models
New models can be easily introduced to the system using one of the following ways:
1. Blackbox Container
    * This doesn't work with scheduling since it requires a [**Container Agent**](#container-agent), but as long as endpoints are correctly specified and data types are correctly aligned, pipeline inference should still work perfectly.
2. Adding new code
    * New code for new types of `Inferencer`, `Preprocessor`, `Postprocessor` can be easily added by modifying the current code.
    * Instructions can be found [here](/libs/workloads/README)
## Local Optimizations
TBA

# Misc
* The main source code can be found within the `libs/` folder while `src/` contains the data sink and simulates the end-user receiving the data.
* Configurations for models and experiments can be found in `jsons/` while the directories `cmake`, `scripts/`, and `dockerfiles` show deployment related code and helpers.
* For analyzing the results we provide python scripts in `analyze`.

## Useful Scripts
### Bandwidth setting
For the purpose of running the experiments with real-world 5G traces, this [script](/scripts/set_bandwidth.sh) is provided, which is invoked from the Device Agent to set the network bandwidth using Linux's *Traffic Control* (tc).

### Stop all containers with a keyword
* If the experiment is not running as expected, we may want to force fully stop them.
* Otherwise, the containers should come to their natural termination eventually.
```bash
./stop_containers_with_keywords.sh KEY_WORD
```

### Change names/Delete of PostgreSQL tables en masse.
* If an experiment is not running as expected and we want to wipe out the old statistics table for a clean slate.

# Citing our works
If you find the repo useful, please cite the following works which have encompassed the development of this repo.

* **OCTOPINF: Workload-Aware Real-Time Inference Serving for Edge Video Analytics** 
    ```
    @inproceedings{nguyen2025,
        author={Thanh-Tung Nguyen and Lucas Liebe and Tau-Nhat Quang and Yuheng Wu and Jinghan Cheng and Dongman Lee}
        title = {{OCTOPINF: Workload-Aware Real-Time Inference Serving for Edge Video Analytics}},
        booktitle = {The 23rd International Conference on Pervasive Computing and Communications (PerCom 2025)},
        year = {2025},
        publisher = {IEEE},
        month = march,
    }
    ```
