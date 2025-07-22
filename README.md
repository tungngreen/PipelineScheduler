# PipelineScheduler

# OctoCross

OctoCross is a real‑time video analytics scheduling system for distributed camera networks. It learns and predicts fine‑grained spatiotemporal workload dynamics and proactively generates efficient task‑offloading strategies to:
![System Architecture](System_arch.png)
- **Boost throughput** by up to 5.8×  
- **Reduce end‑to‑end latency** by up to 3.2×  
- **Maintain high SLO compliance** in dynamic topologies of up to 100 nodes  

Key features include:

1. **Joint Spatiotemporal Learning**  
   - Graph‑based multi‑head attention encoder  
   - End‑to‑end learning of historical load and network topology  

2. **Predictive Scheduling Loop**  
   - Multi‑step iterative offloading based on predicted workloads  
   - Runtime masks skip low‑impact nodes to reduce compute overhead  

3. **Continuous Adaptation**  
   - Online masking to handle node churn  
   - Targeted fine‑tuning of only added/removed nodes (1000 epochs, ≤30 s convergence)

![overall-arch](/assets/System_arch.png)


# Table of Contents

1. [Overview](#pipelinescheduler)  
2. [Implementation Architecture](#implementation-architecture)  
   * [Controller](#controller)  
   * [Device Agent](#device-agent)  
   * [Inference Container](#inference-container)  
     * [Container Agent](#container-agent)  
     * [Configurations](#configurations)  
   * [Knowledge Base](#knowledge-base)
3. [Running ***OctoCross***](#running-pipelinescheduler)  
   * [Installation](#installation)  
     * [Prerequisites](#prerequisites)  
     * [Inference Platform](#inference-platform)  
     * [Build & Compile](#build--compile)  
   * [Preparing Data](#preparing-data)  
   * [Preparing Inference Models](#preparing-inference-models)  
     * [Model weights](#model-weights)  
     * [Model conversion](#model-conversion)
   * [Docker images](#docker-images)
   * [Running Our Experiments](#running-our-experiments)  
     * [ST Model Training](#st-model-training)  
     * [ST Model Testing](#st-model-testing)  
     * [ST Model Weights](#st-model-weights)
     * [Runtime config settings](#runtime-config-seetings)
     * [Running OctoCross](#running-octocross)
4. [Extending ***OctoCross***](#extending-pipelinescheduler)  
   * [Adding Models](#adding-models)  
   * [Local Optimizations](#local-optimizations)  
5. [Misc](#misc)  
6. [Citing Our Works](#citing-our-works)

# Implementation Architecture
PipelineScheduler is composed of 4 main components:

## Controller
The **Controller** is responsible for centralized scheduling. It collects workload traces from all devices, encodes temporal load trends and spatial topology using a spatiotemporal encoder, and predicts the current load on each node. 
Based on the prediction, it generates an offloading matrix that decides how tasks should be redistributed across the network.
The Controller also handles dynamic changes by selectively updating the model when nodes are added or removed.
It queries operational statistics from the Knowledge Base via the PostGreSQL API and issues commands to the **Device Agents** via custom gRPC APIs.

## Device Agent
Each workload handling device (including the Edge server) runs a **Device Agent** as a separate C++ process to manage and monitor the containers using Docker and other APIs (e.g., NVIDIA Driver API).
It records recent workload statistics and sends them to the Controller. Upon receiving scheduling decisions, it manages task migration via containerized modules. The agent can also report resource issues or temporarily opt out of scheduling when overloaded.

## Inference Container
EVA pipelines are organized into DAGs with each node is a container, which packages a DNN model and its pre/postprocessing logics.
Particularly, each container is its own pipeline of microservices which typically follows a structure of `[Receiver -> Preprocessor -> Batcher -> Inferencer -> Postprocessor -> Sender]`. Each microservice runs as a thread to ensure high parallelism.

This design allows microservice to be replaced easily in a plug-and-place manner. For instane, if a TensorRT inferencer (the current version) is not suitable for the hardware, a new inferencer (e.g., ONNX-based) can be whipped up with minimal adaptation.

But other designs (e.g., monolithic) works as well as long as the endpoints for sending/receiving data are specific correctly.

Details on how to set the configurations can be found [here](/jsons/README).

Each container exposes an gRPC API to the **Device Agent**. Two protobuf messages are used a ContainerConfig

      ```protobuf
      message ContainerConfig {
        string name           = 1;  // container identifier
        string json_config    = 2;  // serialized runtime config
        string executable     = 3;  // path to binary or script
        int32  device         = 4;  // device ID on which to run
        int32  control_port   = 5;  // gRPC port for this container
        int32  model_type     = 6;  // enum/model selector
        repeated int32 input_shape = 7;  // e.g. [batch, C, H, W]
      }
      
      message ContainerLink {
        int32   mode               = 1;  // link mode (e.g. pipeline vs. broadcast)
        string  name               = 2;  // upstream container name
        string  downstream_name    = 3;  // downstream container name
        string  ip                 = 4;  // downstream device IP
        int32   port               = 5;  // downstream gRPC port
        float   data_portion       = 6;  // fraction of tasks to offload
        string  old_link           = 7;  // previous link identifier (optional)
        int64   timestamp          = 8;  // link update time (ms since epoch)
        int32   offloading_duration = 9; // duration budget (ms)
      }'''

## Knowledge Base
The Knowledge Base is a PostgreSQL (14) database which contains all the operational statistics.

# Running ***PipelineScheduler***
## Installation
### Prerequisites
To run the system, this following software must be installed on the host machines.
* CUDA 11.4 - 12.6
* CMake (or newer)
* Docker
* OpenCV 4.8.1
* gRPC 1.62.0
* Protobuf 25.1
* PostgreSQL 14

Inside the container, it is also necessary to install inference software platforms (e.g., TensorRT, ONNX).

The specific software versions and commands for installation can be found taken from the [dockerfiles](/dockerfiles/), which are written to build inference container images. 
Since the current version is run on NVIDIA hardware (i.e., GPU and Jetson devices), most of the images are built upon NVIDIA container images published [here](https://catalog.ngc.nvidia.com/containers).

The build instructions can be found [here](dockerfiles/README) and base containers without data or models are available [here](https://hub.docker.com/r/anonymoussub/octocross).

### Inference Platform
The current versions of `Preprocessors, Postprocessors and Inferencer` are written for NVIDIA hardware, especially the `Inferencer`. But custom microservices can be written based on these with minimal adaptation.

The system is designed to be deployed on a Edge cluster, but can also be run on a single machine.
The first step is to build the source code, here you can use multiple options for instance to change the scheduling system.

### Build & Compile
* Controller
* The **Device Agent** 
    ```bash
    mkdir build_host && cd build_host
    cmake -DSYSTEM_NAME=PPP -DON_HOST=True -DDEVICE_ARCH=platform_name
    # Platform name is amd64 or Jetson.
    make -j 64 Device_Agent
    ```

* The microservices **inside each container**
    ```bash
    mkdir build && cd build
    cmake -DSYSTEM_NAME=PPP -DON_HOST=False -DDEVICE_ARCH=platform_name
    # Platform name is amd64 or Jetson.
    make -j 64 Container_[name]
    # Name of the model. YoloV5 for instance.
    ```

## Preparing Data
Our experiments are conducted on two datasets: [MTMMC](https://openaccess.thecvf.com/content/CVPR2024/papers/Woo_MTMMC_A_Large-Scale_Real-World_Multi-Modal_Camera_Tracking_Benchmark_CVPR_2024_paper.pdf) for *Campus* scenario and [AI City Challenge 2022 Track 2 - Multi-vehicle Tracking dataset](https://www.aicitychallenge.org/2022-ai-city-challenge/).

**Data** 
* Please download the data and place them as videos into `data/` in the devices where they are supposed to streamed from in a real-world setting.
* To reproduce our experiments, please placement configurations available in `SToffloading_*.json` in `jsons/`.

**Ground truth**
* We need to collect the number of bounding boxes in each frame to use as the ground truth.
* For the MTMMC dataset, the authors provided the corresponding annotated groundtruth, so we directly extract the number of pedestrians per frame from the GT file to build each node’s workload history.
* In the AIcity scenario, the tracking dataset only includes moving objects and may omit certain detections. We therefore use a YOLO model to detect all target objects in each frame and generate corresponding JSON files for workload history (e.g., data/s36_person.json).Place these files in the PipePlusPlus/data directory (or your custom subfolder) and reference them in your data loading scripts accordingly.



## Preparing Inference Models

### Model weights
ONNX weight files for models used for our experiments can be found [here](https://drive.google.com/drive/u/7/folders/1ir14TSpIRghK-w1nDaZ2Q7dd3O0Xqzbt)

### Model conversion
Models need to be prepared accordingly to fit the current hardware and software inference platforms. For NVIDIA-TensorRT, please build and use following script.

* Build
    ```bash
    mkdir build && cd build
    cmake -DSYSTEM_NAME=PPP -DON_HOST=False -DDEVICE_ARCH=platform_name
    # Platform name is amd64 or jetson.
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

## Docker images
The build instructions can be found [here](dockerfiles/README) and base containers without data or models are available [here](https://hub.docker.com/r/anonymoussub/octocross).

## Running Our Experiments

### ST model training
The following illustrates, using the code above as an example, how to prepare files, run training commands, and the core training workflow.


1. Files to prepare
- **Network topology file** `bandwidth.cites`  
  A three‑column, space‑separated file: source node ID, destination node ID, bandwidth/distance.
- **Time‑series data file** `s01_person.json`  
  A JSON object where each key is a node name and each value has a `"person"` array of per‑frame counts.
- **Training script** `train_st_model.py`  
  Implements `load_data`, the `Generator` model, `discriminator_loss`, and the main training loop.

2. Run commands
    ```bash
    python train_st_model.py --cites_file bandwidth.cites  --data_file s01_person.json  --epochs 50 --batch_size 16 --lr 0.001 --output_model  model_super_factory.pt
    ```
    Argument details

    | Argument | Type | Default | Required | Description |
    |---|---|---|---|---|
    | `--cites_file` | `str` | — | Yes | Path to the topology file (`bandwidth.cites`), which lists edges as “source target weight.” Used by `load_data` to build the adjacency matrix. |
    | `--data_file` | `str` | — | Yes | Path to the node‑time data file (e.g. `Data2.csv` or `s01_person.json`), containing per‑frame workload counts for each node. |
    | `--epochs` | `int` | `100` | No | Number of epochs (full passes over the data) to train the model. |
    | `--batch_size` | `int` | `16` | No | Number of samples per training batch. Larger values speed up training (up to GPU memory limits) but may affect convergence. |
    | `--lr` | `float` | `0.0001` | No | Learning rate for the optimizer; controls the step size on each gradient update. |
    | `--output_model` | `str` | `model_super_campus.pt` | No | File path to save the final trained model weights. |
    | `--mode` | `str` | `train` | No | Operation mode:<br>- `train`: run training loop and save a model<br>- `predict`: load a saved model and perform inference |
    | `--load_model` | `str` | `None` | No | Optional path to a pre‑trained model. If provided in `train` mode, training will resume from these weights; otherwise start from scratch. |
    | `--num_nodes` | `int` | `16` | No | Number of graph nodes; must match the unique node count in `--cites_file`. |
    | `--embed_dim` | `int` | `32` | No | Dimensionality of the node embedding and internal feature vectors. Higher values increase model capacity (and compute cost). |
    | `--threshold` | `float` | `10.0` | No | Workload‑balancing threshold: nodes with load > threshold are considered senders, < threshold are receivers. |
    | `--num_layers` | `int` | `3` | No | Number of `SimpleSTBlock` layers in the spatiotemporal encoder. More layers capture deeper spatiotemporal patterns. |
    | `--heads` | `int` | `4` | No | Number of attention heads in each `SimpleSTBlock`. |
    | `--forecast_horizon` | `int` | `1` | No | Number of future time steps to forecast and schedule at each forward pass. |
    | `--history_steps` | `int` | `16` | No | Length of the historical window (in seconds) fed into the model for forecasting. |

### ST model testing
* Evaluate your Spatio‑Temporal model’s performance with the `--predict` flag:
    ```bash
    cd Model
    python ST_train.py --predict

* This will compare your model’s predictions against the ground truth. Next, point your inference client at the trained .pt file.

### ST Model weights
* Trained weights of models used for our experiments are provided are provided [here](https://drive.google.com/drive/folders/1ir14TSpIRghK-w1nDaZ2Q7dd3O0Xqzbt?usp=sharing)

### Runtime config settings
- **Upstream model**  
  Always use `yolov5.json` for the initial object detection stage.

- **Downstream model**  
  Depending on your task, deploy either:
  - `platedet.json` for license‐plate detection  
  - `retinaface.json` for face detection  

- **Class filter**  
  Edit the `"nb_classOfInterest"` field in your JSON to select COCO class indices you care about (e.g. `[0]` for “person”, `[2,3,5]` for “car”, “motorbike”, “bus”).

- **Input video**  
  Point to the video you want to run inference on via the "input_video" field. Place your MP4 video in the PipePlusPlus/data directory. Update the ST_offloading.json file to specify the input_video and select the model you want to use, for example:
  ```json
  {
    "input_video": "data/your_video.mp4",
    "model": "your_engine_file"
    }
- **Example**  
  Here’s how you’d configure `ST_offloading.json` for one device to run face detection downstream:

     ```json
     {
       "initial_pipelines": [
         {
           "container_name": "cont_faces",
           "model_type": 3,
           "device": "server",
           "recv_port": 5010,
           "batch_size": 4,
           "slo": 200000,
           "pipeline": "face1",
           "model_path": "../models/retinaface_1080_fp32_64_1.engine",
           "input_shape": [3, 576, 640]
         }
       ]
     }
### Running OctoCross
* Step 1: Running the **Controller**.
    ```bash
    cd controller
    python server.py
    ```
    * The guideline to set configurations for controller run is available [here](/jsons/experiments/README).
* Step 2: Once the **Controller** is running, run a **Device Agent** on each device.
    ```bash    
    sudo ./DeviceAgent \
        --name <DEVICE_NAME> \           
        --device_type <DEVICE_TYPE> \    
        --controller_url CONTROLLER_IP 
        --dev_port_offset 0 \           
        --dev_verbose 1 \                
        --dev_bandwidthLimitID 2    
    ```
* Step 3: Run the inference client
    ```bash
    cd controller
    python client.py 
    ```

## Useful Scripts
### Bandwidth setting
For the purpose of running the experiments with real-world 5G traces, this [script](/scripts/set_bandwidth.sh) is provided, which is invoked from the Device Agent to set the network bandwidth using Linux's *Traffic Control* (tc).
The required json configurations can be found [here](/jsons/) or created from the original dataset using this [script](/scripts/create_json_from_5g_dataset.py). 

### Stop all containers with a keyword
* If the experiment is not running as expected, we may want to force fully stop them.
* Otherwise, the containers should come to their natural termination eventually.
```bash
./stop_containers_with_keywords.sh KEY_WORD
```

### Change names/Delete of PostgreSQL tables en masse.
* If an experiment is not running as expected and we want to wipe out the old statistics table for a clean slate.
