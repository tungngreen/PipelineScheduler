# Experiment Configuration Guide

This guide explains how to write and configure an experiment JSON file used for running adaptive video analytics
experiments with PipelineScheduler

## 🔧 File Structure

Please not that most of the strings (e.g. , pipeline_name) will be abbreviated to 4 characters by the system for the
database.
If you require longer names, please use underscore `_` between words that can be abbreviated.
For more information on the abbreviations, please refer to misc.cpp and misc.h in libs/misc.
Here's a breakdown of the key fields:

### 🔹 General Experiment Settings

| Field         | Description                                                    |
|---------------|----------------------------------------------------------------|
| `expName`     | A name for the experiment (used for logging and output files). |
| `system_fps`  | Frames per second that should be read from the video files.    |
| `systemName`  | The system to use, [CHEIS, PPP, DIS, JLF, RIM, BCE].            |
| `runtime`     | Duration of the experiment in minutes.                         |
| `port_offset` | Optional port number offset (e.g., for development setups).    |
| `sink_ip`     | IP address of the result SinkAgent.                            |

### 🔹 Inference Batch Sizes

These Batch Sizes are required for the SOTAS with static batch sizes (DIS, RIM) and as initialization for some of the
adaptive batch size algorithms (e.g. BCE).
They still need to be set, not to cause errors but serve no real purpose in our system.

| Field               | Description                                         |
|---------------------|-----------------------------------------------------|
| `yolov5_batch_size` | Batch size for the object detection model (YOLOv5). |
| `edge_batch_size`   | Inference batch size on edge devices.               |
| `server_batch_size` | Inference batch size on the central server.         |

### 🔹 Scheduling & Rescaling Intervals

| Field                               | Description                                                |
|-------------------------------------|------------------------------------------------------------|
| `scheduling_interval_sec`           | How often full system-level scheduling decisions are made. |
| `rescaling_interval_sec`            | How often the system considers only scaling resources.     |
| `scale_up_interval_threshold_sec`   | Minimum time before scaling up is allowed.                 |
| `scale_down_interval_threshold_sec` | Minimum time before scaling down is allowed.               |

---

## 🎥 Initial Pipelines

The `initial_pipelines` field defines a list of video processing pipelines. Each entry requires:

| Field                    | Description                                                           |
|--------------------------|-----------------------------------------------------------------------|
| `pipeline_name`          | A unique name, usually the video_source name without file ending.     |
| `pipeline_target_slo`    | Target Service Level Objective (in microseconds).                     |
| `pipeline_type`          | Type of pipeline (`0` = traffic, '1' = audience, `2` = surveillance). |
| `video_source`           | Path to the video file from inside the container.                     |
| `pipeline_source_device` | The edge device that hosts the datasource.                            |

### Example:

```json
{
  "pipeline_name": "traffic1",
  "pipeline_target_slo": 200000,
  "pipeline_type": 0,
  "video_source": "../data/short/traffic1.mp4",
  "pipeline_source_device": "nxavier1"
}
```

## CHEIS Hyperparameters

This is only required for CHEIS and can be omitted for other systems.

| Parameter               | Description                                          |
|-------------------------|------------------------------------------------------|
| `theta`, `sigma`, `phi` | Control the weights of the reward function.          |
| `lambda`, `gamma`       | Control the temporal importance in PPO GAE and loss. |
| `clip_epsilon`          | PPO clipping range.                                  |
| `penalty_weight`        | Weight for loss penalty.                             |
| `update_steps`          | Initial local steps between RL updates.              |
| `update_step_incs`      | Optional increment local steps per RL episode.       |
| `federated_steps`       | Number of local updates before FL aggregation.       |
| `seed`                  | Random seed for reproducibility.                     |

