# PipelineScheduler

PipelinePlusPlus (p++ or ppp) is the c++ implementation of the paper **"OCTOPINF: Workload-Aware Real-Time Inference Serving for Edge Video Analytics"**.
When using our Code please cite the following paper:

**SPACE FOR PAPER CITATION**

## Branches

The repository has 4 main branches for the different devices used in the experiments: Edge server, Xavier AGX, Xavier NX and Orin Nano.
This is because of library conflicts and easier tracking of the configurations used during the runtime.
The sub-branches contain all changes from master, with additional device specific modifications e.g. in the profiler.

## Directory Structure

The main source code can be found within the `libs/` folder while `src/` contains the data sink and simulates the end-user receiving the data.
Configurations for models and experiments can be found in `jsons/` while the directories `cmake`, `scripts/`, and `dockerfiles` show deployment related code and helpers.
For analyzing the results we provide python scripts in `analyze`.
The Dockerfile and CmakeList in the root directory are the main entry points to deploy the system.

## Example Usage

To run the system, you will need at least the following dependencies: CMake, Docker, TensorRT, OpenCV, Grpc, Protobuf, and PostgreSQL.
The system is designed to be deployed on a Edge cluster, but can also be run on a single machine.
The first step is to build the source code, here you can use multiple options for instance to change the scheduling system.

```bash
mkdir build && cd build
cmake -DSYSTEM_NAME=[PPP, DIS, JLF, RIM] ..
make -j 64
```

Next, the Dockerfile has to be run to create the containers for individual models / pipeline components.
Then the system can be started with the following two commands in separate terminals:

```bash
./Controller --ctrl_configPath ../jsons/experiments/full-run-ppp.json
./DeviceAgent --name server --device_type server --controller_url localhost --dev_port_offset 0 --dev_verbose 1 --deploy_mode 1
```

Please note that the experiment config json file needs to be adjusted to your local system and even if all dependencies are installed in the correct version, the system might not run out of the box.