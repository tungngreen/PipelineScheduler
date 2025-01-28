# PipelineScheduler

PipelinePlusPlus (p++ or ppp) is the c++ implementation of the paper **"FCPO: Federated Continual Policy Optimization with Heterogeneous Action Spaces for Real-Time High-Throughput Edge Video Analytics"**.
When using our Code please cite the following paper:

**SPACE FOR PAPER CITATION**

## Branches

The repository has 2 main branches for the different devices used in the experiments: Edge server (this one) and [Jetsons](https://anonymous.4open.science/r/fcpo-jetson)
This is because of library conflicts and easier tracking of the configurations used during the runtime.
The sub-branches contain all changes from master, with additional device specific modifications e.g. in the profiler.

## Directory Structure

The main source code can be found within the `libs/` folder while `src/` contains the data sink and simulates the end-user receiving the data.
Configurations for models and experiments can be found in `jsons/` while the directories `cmake`, `scripts/`, and `dockerfiles` show deployment related code and helpers.
For analyzing the results we provide python scripts in `analyze`.
The Dockerfile and CmakeList in the root directory are the main entry points to deploy the system.

## Example Usage

### On host devices and server
To run the system, you will need at least the following dependencies: CMake, Docker, TensorRT (8.4.3.1), OpenCV (4.8.1), Grpc (1.6.2), Protobuf (25.1), and PostgreSQL 14 on your server and devices.
The system is designed to be deployed on a Edge cluster, but can also be run on a single machine.
The first step is to build the source code, here you can use multiple options for instance to change the scheduling system.

```bash
mkdir build && cd build
cmake -DSYSTEM_NAME=[FCPO, DIS, BCE] -DON_HOST=true ..
make -j 64
```

### Building docker images
* Use one of the docker files in `./dockerfiles` to build the base image.
* Use the `./Dockerfile` to build image to run as inference containers for experiments/production.

### Run steps
* Then the system can be started by running this command at the server:
    ```bash
    ./Controller --ctrl_configPath ../jsons/experiments/full-run-fcpo.json
    ```
* At each device, then run:
    ```bash
    ./DeviceAgent --name server --device_type input_device_type_here --controller_url localhost --dev_port_offset 0 --dev_verbose 1 --deploy_mode 1
    ```
* Return to the server terminal and type `yes`.

Please note that the experiment config json file needs to be adjusted to your local system and even if all dependencies are installed in the correct version.

## Notice
The code to draw the figures can be found in `./analyze`