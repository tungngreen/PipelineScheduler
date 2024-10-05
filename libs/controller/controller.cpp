#include "controller.h"

ABSL_FLAG(std::string, ctrl_configPath, "../jsons/experiments/base-experiment.json",
          "Path to the configuration file for this experiment.");
ABSL_FLAG(uint16_t, ctrl_verbose, 0, "Verbosity level of the controller.");
ABSL_FLAG(uint16_t, ctrl_loggingMode, 0, "Logging mode of the controller. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, ctrl_logPath, "../logs", "Path to the log dir for the controller.");

const int DATA_BASE_PORT = 55001;
const int CONTROLLER_BASE_PORT = 60001;
const int DEVICE_CONTROL_PORT = 60002;

// ======================================================================================================================================== //
// ======================================================================================================================================== //
// ======================================================================================================================================== //

GPULane::GPULane(GPUHandle *gpu, NodeHandle *device, uint16_t laneNum) : laneNum(laneNum), gpuHandle(gpu), node(device) {
    dutyCycle = 0;
    portionList.head = nullptr;
    portionList.list = {};
}

bool GPUPortion::assignContainer(ContainerHandle *container) {
    if (this->container != nullptr) {
        spdlog::get("console")->error("Portion already assigned to container {0:s}", this->container->name);
        return false;
    }
    container->executionPortion = this;
    this->container = container;
    start = container->startTime;
    end = container->endTime;

    spdlog::get("container_agent")->info("Portion assigned to container {0:s}", container->name);
    return true;
}

// GPUPortion::~GPUPortion() {
//     if (container != nullptr) {
//         throw std::runtime_error("Portion cannot be destroyed while it is still assigned to a container %s" + container->name);
//     }
//     lane->removePortion(this);
//     // TODO: remove from the list of free portions
//     spdlog::get("container_agent")->info("Portion is destroyed.");
// }

bool GPULane::removePortion(GPUPortion *portion) {
    if (portion->lane != this) {
        throw std::runtime_error("Lane %d cannot remove portion %s, which does not belong to it." + portion->container->name + std::to_string(laneNum));
        return false;
    }
    if (portion->prevInLane != nullptr) {
        portion->prevInLane->nextInLane = portion->nextInLane;
        
    }
    if (portion->nextInLane != nullptr) {
        portion->nextInLane->prevInLane = portion->prevInLane;
    }

    if (portion == portionList.head) {
        portionList.head = portion->nextInLane;
    }
    portion->prevInLane = nullptr;
    portion->nextInLane = nullptr;

    auto it = std::find(portionList.list.begin(), portionList.list.end(), portion);
    portionList.list.erase(it);
    return true;
}

// ======================================================================================================================================== //
// ======================================================================================================================================== //
// ======================================================================================================================================== //


// ============================================================ Configurations ============================================================ //
// ======================================================================================================================================== //
// ======================================================================================================================================== //
// ======================================================================================================================================== //

void Controller::readInitialObjectCount(const std::string &path) {
    std::ifstream file(path);
    json j = json::parse(file);
    std::map<std::string, std::map<std::string, std::map<int, float>>> initialPerSecondRate;
    for (auto &item: j.items()) {
        std::string streamName = item.key();
        initialPerSecondRate[streamName] = {};
        for (auto &object: item.value().items()) {
            std::string objectName = object.key();
            initialPerSecondRate[streamName][objectName] = {};
            std::vector<int> perFrameObjCount = object.value().get<std::vector<int>>();
            int numFrames = perFrameObjCount.size();
            int totalNumObjs = 0;
            for (auto i = 0; i < numFrames; i++) {
                totalNumObjs += perFrameObjCount[i];
                if ((i + 1) % 30 != 0) {
                    continue;
                }
                int seconds = (i + 1) / 30;
                initialPerSecondRate[streamName][objectName][seconds] = totalNumObjs * 1.f / seconds;
            }
        }
        float skipRate = ctrl_systemFPS / 30.f;
        std::map<std::string, float> *stream = &(ctrl_initialRequestRates[streamName]);
        float maxPersonRate = 1.2 * std::max_element(
                    initialPerSecondRate[streamName]["person"].begin(),
                    initialPerSecondRate[streamName]["person"].end()
            )->second * skipRate;
        maxPersonRate = std::max(maxPersonRate, ctrl_systemFPS * 1.f);
        float maxCarRate = 1.2 * std::max_element(
                    initialPerSecondRate[streamName]["car"].begin(),
                    initialPerSecondRate[streamName]["car"].end()
            )->second * skipRate;
        maxCarRate = std::max(maxCarRate, ctrl_systemFPS * 1.f);
        if (streamName.find("traffic") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});

            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"arcface", std::ceil(maxPersonRate * 0.6)});
            stream->insert({"carbrand", std::ceil(maxCarRate)});
            stream->insert({"platedet", std::ceil(maxCarRate)});
        } else if (streamName.find("people") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"age", std::ceil(maxPersonRate) * 0.6});
            stream->insert({"gender", std::ceil(maxPersonRate) * 0.6});
            stream->insert({"movenet", std::ceil(maxPersonRate)});
        } else if (streamName.find("zoom") != std::string::npos) {
            stream->insert({"retinaface", ctrl_systemFPS});
            stream->insert({"arcface", std::ceil(maxPersonRate)});
            stream->insert({"age", std::ceil(maxPersonRate)});
            stream->insert({"gender", std::ceil(maxPersonRate)});
            stream->insert({"emotionnet", std::ceil(maxPersonRate)});
        }
    }
}

void Controller::readConfigFile(const std::string &path) {
    std::ifstream file(path);
    json j = json::parse(file);

    ctrl_experimentName = j["expName"];
    ctrl_systemName = j["systemName"];
    ctrl_runtime = j["runtime"];
    ctrl_port_offset = j["port_offset"];
    ctrl_systemFPS = j["system_fps"];
    ctrl_sinkNodeIP = j["sink_ip"];
    ctrl_initialBatchSizes["yolov5"] = j["yolov5_batch_size"];
    ctrl_initialBatchSizes["edge"] = j["edge_batch_size"];
    ctrl_initialBatchSizes["server"] = j["server_batch_size"];
    ctrl_controlTimings.schedulingIntervalSec = j["scheduling_interval_sec"];
    ctrl_controlTimings.rescalingIntervalSec = j["rescaling_interval_sec"];
    ctrl_controlTimings.scaleUpIntervalThresholdSec = j["scale_up_interval_threshold_sec"];
    ctrl_controlTimings.scaleDownIntervalThresholdSec = j["scale_down_interval_threshold_sec"];
    initialTasks = j["initial_pipelines"];
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val) {
    j.at("pipeline_name").get_to(val.name);
    j.at("pipeline_target_slo").get_to(val.slo);
    j.at("pipeline_type").get_to(val.type);
    j.at("video_source").get_to(val.source);
    j.at("pipeline_source_device").get_to(val.device);
    val.fullName = val.name + "_" + val.device;
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

bool GPUHandle::addContainer(ContainerHandle *container) {
    if (container->name.find("datasource") != std::string::npos ||
        container->name.find("sink") != std::string::npos) {
        containers.insert({container->name, container});
        container->gpuHandle = this;
        spdlog::get("container_agent")->info("Container {} successfully added to GPU {} of {}", container->name, number, hostName);
        return true;
    }
    MemUsageType potentialMemUsage;
    potentialMemUsage = currentMemUsage + container->getExpectedTotalMemUsage();
    
    if (currentMemUsage > memLimit) {
        spdlog::get("container_agent")->error("Container {} cannot be assigned to GPU {} of {}"
                                            "due to memory limit", container->name, number, hostName);
        return false;
    }
    containers.insert({container->name, container});
    container->gpuHandle = this;
    currentMemUsage = potentialMemUsage;
    spdlog::get("container_agent")->info("Container {} successfully added to GPU {} of {}", container->name, number, hostName);
    return true;
}

bool GPUHandle::removeContainer(ContainerHandle *container) {
    if (containers.find(container->name) == containers.end()) {
        spdlog::get("container_agent")->error("Container {} not found in GPU {} of {}", container->name, number, hostName);
        return false;
    }
    containers.erase(container->name);
    container->gpuHandle = nullptr;
    currentMemUsage -= container->getExpectedTotalMemUsage();

    spdlog::get("container_agent")->info("Container {} successfully removed from GPU {} of {}", container->name, number, hostName);
    return true;

}


// ============================================================= Con/Desstructors ============================================================= //
// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //


Controller::Controller(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    readConfigFile(absl::GetFlag(FLAGS_ctrl_configPath));
    readInitialObjectCount("../jsons/object_count.json");

    ctrl_logPath = absl::GetFlag(FLAGS_ctrl_logPath);
    ctrl_logPath += "/" + ctrl_experimentName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_logPath += "/" + ctrl_systemName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_verbose = absl::GetFlag(FLAGS_ctrl_verbose);
    ctrl_loggingMode = absl::GetFlag(FLAGS_ctrl_loggingMode);

    setupLogger(
            ctrl_logPath,
            "controller",
            ctrl_loggingMode,
            ctrl_verbose,
            ctrl_loggerSinks,
            ctrl_logger
    );

    ctrl_containerLib = getContainerLib("all");

    json metricsCfgs = json::parse(std::ifstream("../jsons/metricsserver.json"));
    ctrl_metricsServerConfigs.from_json(metricsCfgs);
    ctrl_metricsServerConfigs.schema = abbreviate(ctrl_experimentName + "_" + ctrl_systemName);
    ctrl_metricsServerConfigs.user = "controller";
    ctrl_metricsServerConfigs.password = "agent";
    ctrl_metricsServerConn = connectToMetricsServer(ctrl_metricsServerConfigs, "controller");

    // Check if schema exists
    std::string sql = "SELECT schema_name FROM information_schema.schemata WHERE schema_name = '" + ctrl_metricsServerConfigs.schema + "';";
    pqxx::result res = pullSQL(*ctrl_metricsServerConn, sql);
    if (res.empty()) {
        std::string sql = "CREATE SCHEMA IF NOT EXISTS " + ctrl_metricsServerConfigs.schema + ";";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + 
              " GRANT ALL PRIVILEGES ON TABLES TO controller;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "GRANT USAGE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "GRANT CREATE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " GRANT SELECT, INSERT ON TABLES TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
    }


    std::thread networkCheckThread(&Controller::checkNetworkConditions, this);
    networkCheckThread.detach();

    running = true;

    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", CONTROLLER_BASE_PORT + ctrl_port_offset);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    // append one device for sink of type server
    NodeHandle *sink_node = new NodeHandle("sink", ctrl_sinkNodeIP,
                                           ControlCommands::NewStub(grpc::CreateChannel(
                                      absl::StrFormat("%s:%d", ctrl_sinkNodeIP, DEVICE_CONTROL_PORT + ctrl_port_offset), grpc::InsecureChannelCredentials())),
                      new CompletionQueue(), SystemDeviceType::Server,
                      DATA_BASE_PORT + ctrl_port_offset, {});
    devices.addDevice("sink", sink_node);

    ctrl_nextSchedulingTime = std::chrono::system_clock::now();
}

Controller::~Controller() {
    for (auto msvc: containers.getList()) {
        StopContainer(msvc, msvc->device_agent, true);
    }

    for (auto &device: devices.getList()) {
        EmptyMessage request;
        EmptyMessage reply;
        ClientContext context;
        Status status;
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                device->stub->AsyncShutdown(&context, request, device->cq));
        rpc->Finish(&reply, &status, (void *)1);
        void *got_tag;
        bool ok = false;
        if (device->cq != nullptr) GPR_ASSERT(device->cq->Next(&got_tag, &ok));
        device->cq->Shutdown();
    }
    server->Shutdown();
    cq->Shutdown();
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //







// ============================================================ Excutor/Maintainers ============================================================ //
// ============================================================================================================================================= //
// ============================================================================================================================================= //
// ============================================================================================================================================= //

MemUsageType ContainerHandle::getExpectedTotalMemUsage() const {
    if (device_agent->type == SystemDeviceType::Server) {
        return pipelineModel->processProfiles.at("server").batchInfer[pipelineModel->batchSize].gpuMemUsage;
    }
    std::string deviceTypeName = getDeviceTypeName(device_agent->type);
    return (pipelineModel->processProfiles.at(deviceTypeName).batchInfer[pipelineModel->batchSize].gpuMemUsage +
            pipelineModel->processProfiles.at(deviceTypeName).batchInfer[pipelineModel->batchSize].rssMemUsage) / 1000;
}

bool Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    TaskHandle *task = new TaskHandle{t.name, t.type, t.source, t.device, t.slo, {}, 0};

    std::map<std::string, NodeHandle*> deviceList = devices.getMap();

    if (deviceList.find(t.device) == deviceList.end()) {
        spdlog::error("Device {0:s} is not connected", t.device);
        return false;
    }

    while (!deviceList.at(t.device)->initialNetworkCheck) {
        spdlog::get("container_agent")->info("Waiting for device {0:s} to finish network check", t.device);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    task->tk_src_device = t.device;

    task->tk_pipelineModels = getModelsByPipelineType(t.type, t.device, t.name, t.source);
    for (auto &model: task->tk_pipelineModels) {
        model->datasourceName = t.source;
        model->task = task;
    }

    ctrl_savedUnscheduledPipelines.addTask(task->tk_name, task);
    return true;
}

void Controller::initialiseGPU(NodeHandle *node, int numGPUs, std::vector<int> memLimits) {
    if (node->type == SystemDeviceType::Server) {
        for (uint8_t gpuIndex = 0; gpuIndex < numGPUs; gpuIndex++) {
            std::string gpuName = "gpu" + std::to_string(gpuIndex);
            GPUHandle *gpuNode = new GPUHandle{"3090", "server", gpuIndex, memLimits[gpuIndex] - 2000, NUM_LANES_PER_GPU, node};
            node->gpuHandles.emplace_back(gpuNode);
        }
    } else {
        //MemUsageType memSize = node->type == SystemDeviceType::AGXXavier ? 30000 : 5000;
        GPUHandle *gpuNode = new GPUHandle{node->name, node->name, 0, memLimits[0] - 1500, 1, node};
        node->gpuHandles.emplace_back(gpuNode);
    }
}

void Controller::basicGPUScheduling(std::vector<ContainerHandle *> new_containers) {
    std::map<std::string, std::vector<ContainerHandle *>> scheduledContainers;
    for (auto device: devices.getMap()) {
        for (auto &container: new_containers) {
            if (container->device_agent->name != device.first) {
                continue;
            }
            if (container->name.find("datasource") != std::string::npos ||
                container->name.find("sink") != std::string::npos) {
                continue;
            }
            scheduledContainers[device.first].push_back(container);
        }
        std::sort(scheduledContainers[device.first].begin(), scheduledContainers[device.first].end(),
                [](ContainerHandle *a, ContainerHandle *b) {
                    auto aMemUsage = a->getExpectedTotalMemUsage();
                    auto bMemUsage = b->getExpectedTotalMemUsage();
                    return aMemUsage > bMemUsage;
                });
    }
    for (auto device: devices.getMap()) {
        std::vector<GPUHandle *> gpus = device.second->gpuHandles;
        for (auto &container: scheduledContainers[device.first]) {
            MemUsageType containerMemUsage = container->getExpectedTotalMemUsage();
            MemUsageType smallestGap  = std::numeric_limits<MemUsageType>::max();
            int8_t smallestGapIndex = -1;
            for (auto &gpu: gpus) {
                MemUsageType gap = gpu->memLimit - gpu->currentMemUsage - containerMemUsage;
                if (gap < 0) {
                    continue;
                }
                if (gap < smallestGap) {
                    smallestGap = gap;
                    smallestGapIndex = gpu->number;
                }
            }
            if (smallestGapIndex == -1) {
                spdlog::get("container_agent")->error("No GPU available for container {}", container->name);
                continue;
            }
            gpus[smallestGapIndex]->addContainer(container);
        }

    }
}

/**
 * @brief call this method after the pipeline models have been added to scheduled
 *
 */
void Controller::ApplyScheduling() {
    std::vector<ContainerHandle *> new_containers;

    // designate all current models no longer valid to run
    // after scheduling some of them will be designated as valid
    // All the invalid will be stopped and removed.
    for (auto device: devices.getList()) {
        std::unique_lock lock_device(device->nodeHandleMutex);
        for (auto &[modelName, model] : device->modelList) {
            model->toBeRun = false;
        }
    }

    /**
     * @brief // Turn schedule tasks/pipelines into containers
     * Containers that are already running may be kept running if they are still valid
     */
    for (auto &[pipeName, pipe]: ctrl_scheduledPipelines.getMap()) {
        for (auto &model: pipe->tk_pipelineModels) {
            if (ctrl_systemName != "ppp" && ctrl_systemName != "jlf") {
                model->cudaDevices.emplace_back(0);
                model->numReplicas = 1;
            }
            bool upstreamIsDatasource = (std::find_if(model->upstreams.begin(), model->upstreams.end(),
                                                      [](const std::pair<PipelineModel *, int> &upstream) {
                                                          return upstream.first->name.find("datasource") != std::string::npos;
                                                      }) != model->upstreams.end());
            if (model->name.find("yolov5n") != std::string::npos && model->device != "server" && upstreamIsDatasource) {
                if (model->name.find("yolov5ndsrc") == std::string::npos) {
                    model->name = replaceSubstring(model->name, "yolov5n", "yolov5ndsrc");
                }

            } else if (model->name.find("retina1face") != std::string::npos && model->device != "server" && upstreamIsDatasource) {
                if (model->name.find("retina1facedsrc") == std::string::npos) {
                    model->name = replaceSubstring(model->name, "retina1face", "retina1facedsrc");
                }
            }

            std::unique_lock lock_model(model->pipelineModelMutex);
            // look for the model full name 
            std::string modelFullName = model->name;

            // check if the pipeline already been scheduled once before
            PipelineModel* pastModel = nullptr;
            std::map<std::string, TaskHandle*> pastScheduledPipelines = ctrl_pastScheduledPipelines.getMap();
            if (pastScheduledPipelines.find(pipeName) != pastScheduledPipelines.end()) {

                auto it = std::find_if(pastScheduledPipelines[pipeName]->tk_pipelineModels.begin(),
                                       pastScheduledPipelines[pipeName]->tk_pipelineModels.end(),
                                              [&modelFullName](PipelineModel *m) {
                                                  return m->name == modelFullName;
                                              });
                // if the model is found in the past scheduled pipelines, its containers can be reused
                if (it != pastScheduledPipelines[pipeName]->tk_pipelineModels.end()) {
                    pastModel = *it;
                    std::vector<ContainerHandle*> pastModelContainers = pastModel->task->tk_subTasks[model->name];
                    for (auto container: pastModelContainers) {
                        if (container->device_agent->name == model->device) {
                            model->task->tk_subTasks[model->name].push_back(container);
                        }
                    }
                    pastModel->toBeRun = true;
                }
            }
            std::vector<ContainerHandle *> candidates = model->task->tk_subTasks[model->name];
            int candidate_size = candidates.size();
            // make sure enough containers are running with the right configurations
            if (candidate_size < model->numReplicas) {
                // start additional containers
                for (unsigned int i = candidate_size; i < model->numReplicas; i++) {
                    ContainerHandle *container = TranslateToContainer(model, devices.getDevice(model->device), i);
                    if (container == nullptr) {
                        continue;
                    }
                    new_containers.push_back(container);
                    new_containers.back()->pipelineModel = model;
                }
            } else if (candidate_size > model->numReplicas) {
                // remove the extra containers
                for (int i = model->numReplicas; i < candidate_size; i++) {
                    StopContainer(candidates[i], candidates[i]->device_agent);
                    model->task->tk_subTasks[model->name].erase(
                            std::remove(model->task->tk_subTasks[model->name].begin(),
                                        model->task->tk_subTasks[model->name].end(), candidates[i]),
                            model->task->tk_subTasks[model->name].end());
                    candidates.erase(candidates.begin() + i);
                }
            }
        }
    }
    // Rearranging the upstreams and downstreams for containers;
    for (auto pipe: ctrl_scheduledPipelines.getList()) {
        for (auto &model: pipe->tk_pipelineModels) {
            // If its a datasource, we dont have to do it now
            // datasource doesnt have upstreams
            // and the downstreams will be set later
            if (model->name.find("datasource") != std::string::npos) {
                continue;
            }

            for (auto &container: model->task->tk_subTasks[model->name]) {
                container->upstreams = {};
                for (auto &[upstream, coi]: model->upstreams) {
                    for (auto &upstreamContainer: upstream->task->tk_subTasks[upstream->name]) {
                        container->upstreams.push_back(upstreamContainer);
                        upstreamContainer->downstreams.push_back(container);
                    }
                }
            }

        }
    }

    if (ctrl_systemName != "ppp") {
        basicGPUScheduling(new_containers);
    } else {
        colocationTemporalScheduling();
    }

    // // Testing gpu portion reclaiming
    // uint8_t numReclaims = 0, numSinks = 0;

    // for (auto container: new_containers) {
    //     if (container->model == Sink) {
    //         numSinks++;
    //     }
    // }

    // std::mt19937 gen(3000);

    // std::uniform_int_distribution<> dis(1, 100);
    // while (numReclaims < new_containers.size() - numSinks) {
        
    //     for (auto container : new_containers) {
    //         if (container->model == Sink || container->executionPortion == nullptr) {
    //             continue;
    //         }
    //         int random = dis(gen);
    //         if (random <= 33) {
    //             std::cout << "Reclaiming GPU Portion for container: " << container->name << std::endl;
    //             reclaimGPUPortion(container->executionPortion);
    //             container->executionPortion = nullptr;
    //             numReclaims++;
    //         }
    //     }
    // }
    // // done testing

    for (auto pipe: ctrl_scheduledPipelines.getList()) {
        for (auto &model: pipe->tk_pipelineModels) {
            //int i = 0;
            std::vector<ContainerHandle *> candidates = model->task->tk_subTasks[model->name];
            for (auto *candidate: candidates) {
                if (std::find(new_containers.begin(), new_containers.end(), candidate) != new_containers.end() || candidate->model == Sink) {
                    continue;
                }
                if (candidate->device_agent->name != model->device) {
                    candidate->batch_size = model->batchSize;
                    //candidate->cuda_device = model->cudaDevices[i++];
                    MoveContainer(candidate, devices.getDevice(model->device));
                    continue;
                }
                if (candidate->batch_size != model->batchSize)
                    AdjustBatchSize(candidate, model->batchSize);
                AdjustTiming(candidate);
                //if (candidate->cuda_device != model->cudaDevices[i++])
                //    AdjustCudaDevice(candidate, model->cudaDevices[i - 1]);
            }
        }
    }

    for (auto container: new_containers) {
        StartContainer(container);
        containers.addContainer(container->name, container);
    }

    ctrl_pastScheduledPipelines = ctrl_scheduledPipelines;

    spdlog::get("container_agent")->info("SCHEDULING DONE! SEE YOU NEXT TIME!");
}

bool CheckMergable(const std::string &m) {
    return m == "datasource" || m == "yolov5n" || m == "retina1face" || m == "yolov5ndsrc" || m == "retina1facedsrc";
}

ContainerHandle *Controller::TranslateToContainer(PipelineModel *model, NodeHandle *device, unsigned int i) {
    if (model->name.find("datasource") != std::string::npos) {
        for (auto &[downstream, coi] : model->downstreams) {
            if ((downstream->name.find("yolov5n") != std::string::npos ||
                 downstream->name.find("retina1face") != std::string::npos) &&
                downstream->device != "server") {
                return nullptr;
            }
        }
    }
    std::string modelName = splitString(model->name, "_").back();

    int class_of_interest = -1;
    if (!model->upstreams.empty() && model->name.find("datasource") == std::string::npos &&
        model->name.find("dsrc") == std::string::npos) {
        class_of_interest = model->upstreams[0].second;
    }

    std::string subTaskName = model->name;
    std::string containerName = ctrl_experimentName + "_" + ctrl_systemName + "_" + model->task->tk_name + "_" +
            modelName + "_" + std::to_string(i);
    // the name of the container type to look it up in the container library
    std::string containerTypeName = modelName + "_" + getDeviceTypeName(device->type);
    
    auto *container = new ContainerHandle{containerName, i,
                                          class_of_interest,
                                          ModelTypeReverseList[modelName],
                                          CheckMergable(modelName),
                                          {0},
                                          model->task->tk_slo,
                                          0.0,
                                          model->batchSize,
                                          device->next_free_port++,
                                          ctrl_containerLib[containerTypeName].modelPath,
                                          device,
                                          model->task,
                                          model};
    
    if (model->name.find("datasource") != std::string::npos) {
        container->dimensions = ctrl_containerLib[containerTypeName].templateConfig["container"]["cont_pipeline"][0]["msvc_dataShape"][0].get<std::vector<int>>();
    } else if (model->name.find("320") != std::string::npos) {
        container->dimensions = {3, 320, 320};
    } else if (model->name.find("512") != std::string::npos) {
        container->dimensions = {3, 512, 512};
    } else if (model->name.find("sink") == std::string::npos) {
        container->dimensions = ctrl_containerLib[containerTypeName].templateConfig["container"]["cont_pipeline"][1]["msvc_dnstreamMicroservices"][0]["nb_expectedShape"][0].get<std::vector<int>>();
    }

    // container->timeBudgetLeft for lazy dropping
    container->timeBudgetLeft = container->pipelineModel->timeBudgetLeft;
    // container start time
    container->startTime = container->pipelineModel->startTime;
    // container end time
    container->endTime = container->pipelineModel->endTime;
    // container SLO
    container->localDutyCycle = container->pipelineModel->localDutyCycle;
    // 
    container->cycleStartTime = ctrl_currSchedulingTime;

    model->task->tk_subTasks[subTaskName].push_back(container);

    // for (auto &upstream: model->upstreams) {
    //     std::string upstreamSubTaskName = upstream.first->name;
    //     for (auto &upstreamContainer: upstream.first->task->tk_subTasks[upstreamSubTaskName]) {
    //         container->upstreams.push_back(upstreamContainer);
    //         upstreamContainer->downstreams.push_back(container);
    //     }
    // }

    // for (auto &downstream: model->downstreams) {
    //     std::string downstreamSubTaskName = downstream.first->name;
    //     for (auto &downstreamContainer: downstream.first->task->tk_subTasks[downstreamSubTaskName]) {
    //         container->downstreams.push_back(downstreamContainer);
    //         downstreamContainer->upstreams.push_back(container);
    //     }
    // }
    model->manifestations.push_back(container);
    return container;
}

void Controller::AdjustTiming(ContainerHandle *container) {
    // container->timeBudgetLeft for lazy dropping
    container->timeBudgetLeft = container->pipelineModel->timeBudgetLeft;
    // container->start_time
    container->startTime = container->pipelineModel->startTime;
    // container->end_time
    container->endTime = container->pipelineModel->endTime;
    // duty cycle of the lane where the container is assigned
    container->localDutyCycle = container->pipelineModel->localDutyCycle;
    // `container->task->tk_slo` for the total SLO of the pipeline
    container->cycleStartTime = ctrl_currSchedulingTime;

    TimeKeeping request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(container->name);
    request.set_slo(container->pipelineSLO);
    request.set_time_budget(container->timeBudgetLeft);
    request.set_start_time(container->startTime);
    request.set_end_time(container->endTime);
    request.set_local_duty_cycle(container->localDutyCycle);
    request.set_cycle_start_time(std::chrono::duration_cast<TimePrecisionType>(container->cycleStartTime.time_since_epoch()).count());
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container->device_agent->stub->AsyncUpdateTimeKeeping(&context, request,
                                                               container->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (container->device_agent->cq != nullptr) GPR_ASSERT(container->device_agent->cq->Next(&got_tag, &ok));
    if (!status.ok()) {
        spdlog::get("container_agent")->error("Failed to update TimeKeeping for container: {0:s}", container->name);
        return;
    }
}

void Controller::StartContainer(ContainerHandle *container, bool easy_allocation) {
    spdlog::get("container_agent")->info("Starting container: {0:s}", container->name);
    ContainerConfig request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    json start_config;
    unsigned int control_port;
    std::string pipelineName = splitString(container->name, "_")[2];
    ModelType model = static_cast<ModelType>(container->model);
    std::string modelName = getContainerName(container->device_agent->type, model);
    std::cout << "Creating container: " << container->name << std::endl;
    if (model == ModelType::Sink) {
        start_config["experimentName"] = ctrl_experimentName;
        start_config["systemName"] = ctrl_systemName;
        start_config["pipelineName"] = pipelineName;
        control_port = container->recv_port;
    } else {
        start_config = ctrl_containerLib[modelName].templateConfig;

        // adjust container configs
        start_config["container"]["cont_experimentName"] = ctrl_experimentName;
        start_config["container"]["cont_systemName"] = ctrl_systemName;
        start_config["container"]["cont_pipeName"] = pipelineName;
        start_config["container"]["cont_hostDevice"] = container->device_agent->name;
        start_config["container"]["cont_hostDeviceType"] = ctrl_sysDeviceInfo[container->device_agent->type];
        start_config["container"]["cont_name"] = container->name;
        start_config["container"]["cont_allocationMode"] = easy_allocation ? 1 : 0;
        if (ctrl_systemName == "ppp") {
            start_config["container"]["cont_batchMode"] = 2;
        }
        if (ctrl_systemName == "ppp" || ctrl_systemName == "jlf") {
            start_config["container"]["cont_dropMode"] = 1;
        }
        start_config["container"]["cont_pipelineSLO"] = container->task->tk_slo;
        start_config["container"]["cont_timeBudgetLeft"] = container->timeBudgetLeft;
        start_config["container"]["cont_startTime"] = container->startTime;
        start_config["container"]["cont_endTime"] = container->endTime;
        start_config["container"]["cont_localDutyCycle"] = container->localDutyCycle;
        start_config["container"]["cont_cycleStartTime"] = std::chrono::duration_cast<TimePrecisionType>(container->cycleStartTime.time_since_epoch()).count();

        if (container->model != DataSource &&
            container->model != Sink) {
            std::vector<uint32_t> modelProfile;
            for (auto &[batchSize, profile]: container->pipelineModel->processProfiles.at(ctrl_sysDeviceInfo[container->device_agent->type]).batchInfer) {
                modelProfile.push_back(batchSize);
                modelProfile.push_back(profile.p95prepLat);
                modelProfile.push_back(profile.p95inferLat);
                modelProfile.push_back(profile.p95postLat);
            }

            if (modelProfile.empty()) {
                spdlog::get("container_agent")->warn("Model profile not found for container: {0:s}", container->name);
            }
            start_config["container"]["cont_modelProfile"] = modelProfile;
        }

        json base_config = start_config["container"]["cont_pipeline"];

        // adjust pipeline configs
        for (auto &j: base_config) {
            j["msvc_idealBatchSize"] = container->batch_size;
            j["msvc_pipelineSLO"] = container->pipelineSLO;
        }
        if (model == ModelType::DataSource) {
            base_config[0]["msvc_dataShape"] = {container->dimensions};
            base_config[0]["msvc_idealBatchSize"] = ctrl_systemFPS;
        } else {
            if (model == ModelType::Yolov5nDsrc || model == ModelType::RetinafaceDsrc) {
                base_config[0]["msvc_dataShape"] = {container->dimensions};
                base_config[0]["msvc_type"] = 500;
                base_config[0]["msvc_idealBatchSize"] = ctrl_systemFPS;
            }
            base_config[1]["msvc_dnstreamMicroservices"][0]["nb_expectedShape"] = {container->dimensions};
            base_config[2]["path"] = container->model_file;
        }

        // adjust receiver upstreams
        base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"] = {};
        if (container->model == DataSource || container->model == Yolov5nDsrc || container->model == RetinafaceDsrc) {
            base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = "video_source";
            base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"].push_back(container->pipelineModel->datasourceName);
        } else {
            if (!container->pipelineModel->upstreams.empty()) {
                base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = container->pipelineModel->upstreams[0].first->name;
            } else {
                base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = "empty";
            }
            base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"].push_back(absl::StrFormat("0.0.0.0:%d", container->recv_port));
        }
//        if ((container->device_agent == container->upstreams[0]->device_agent) && (container->gpuHandle == container->upstreams[0]->gpuHandle)) {
//            base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::localGPU;
//        } else {
//            base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
//        }
        //TODO: REMOVE THIS IF WE EVER DECIDE TO USE GPU COMM AGAIN
        base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;

        // adjust sender downstreams
        json sender = base_config.back();
        uint16_t postprocessorIndex = base_config.size() - 2;
        json post_down = base_config[base_config.size() - 2]["msvc_dnstreamMicroservices"][0];
        base_config[base_config.size() - 2]["msvc_dnstreamMicroservices"] = json::array();
        base_config.erase(base_config.size() - 1);
        int i = 1;
        for (auto [dwnstr, coi]: container->pipelineModel->downstreams) {
            json *postprocessor = &base_config[postprocessorIndex];
            sender["msvc_name"] = "sender" + std::to_string(i++);
            sender["msvc_dnstreamMicroservices"][0]["nb_name"] = dwnstr->name;
            sender["msvc_dnstreamMicroservices"][0]["nb_link"] = {};
            for (auto *replica: container->downstreams) {
                if (replica->pipelineModel->name == dwnstr->name) {
                    sender["msvc_dnstreamMicroservices"][0]["nb_link"].push_back(
                            absl::StrFormat("%s:%d", replica->device_agent->ip, replica->recv_port));
                }
            }
            post_down["nb_name"] = sender["msvc_name"];
            if (container->device_agent != dwnstr->deviceAgent) {
                post_down["nb_commMethod"] = CommMethod::encodedCPU;
                sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
            } else {
                //TODO: REMOVE AND FIX THIS IF WE EVER DECIDE TO USE GPU COMM AGAIN
//                if ((container->gpuHandle == dwnstr->gpuHandle)) {
//                    post_down["nb_commMethod"] = CommMethod::localGPU;
//                    sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::localGPU;
//                } else {
                post_down["nb_commMethod"] = CommMethod::localCPU;
                sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
//                }
            }
            post_down["nb_classOfInterest"] = coi;

            postprocessor->at("msvc_dnstreamMicroservices").push_back(post_down);
            base_config.push_back(sender);
        }

        start_config["container"]["cont_pipeline"] = base_config;
        control_port = container->recv_port - 5000;
    }

    request.set_name(container->name);
    request.set_json_config(start_config.dump());
    std::cout << start_config.dump() << std::endl;
    request.set_executable(ctrl_containerLib[modelName].runCommand);
    if (container->model == DataSource || container->model == Sink) {
        request.set_device(-1);
    } else if (container->device_agent->name == "server") {
        if (container->gpuHandle == nullptr) {
            container->gpuHandle = container->device_agent->gpuHandles[3];
        }
        request.set_device(container->gpuHandle->number);
    } else {
        request.set_device(0);
    }
    request.set_control_port(control_port);


    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container->device_agent->stub->AsyncStartContainer(&context, request,
                                                                      container->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (container->device_agent->cq != nullptr) GPR_ASSERT(container->device_agent->cq->Next(&got_tag, &ok));
    if (!status.ok()) {
        spdlog::get("container_agent")->error("Failed to start container: {0:s}", container->name);
        return;
    }
    spdlog::get("container_agent")->info("Container {0:s} started", container->name);
}

void Controller::MoveContainer(ContainerHandle *container, NodeHandle *device) {
    NodeHandle *old_device = container->device_agent;
    bool start_dsrc = false, merge_dsrc = false;
    if (device->name != "server") {
        if (container->mergable) {
            merge_dsrc = true;
            if (container->model == Yolov5n) {
                container->model = Yolov5nDsrc;
            } else if (container->model == Retinaface) {
                container->model = RetinafaceDsrc;
            }
        }
    } else {
        if (container->mergable) {
            start_dsrc = true;
            if (container->model == Yolov5nDsrc) {
                container->model = Yolov5n;
            } else if (container->model == RetinafaceDsrc) {
                container->model = Retinaface;
            }
        }
    }
    container->device_agent = device;
    container->recv_port = device->next_free_port++;
    device->containers.insert({container->name, container});
    container->gpuHandle = container->gpuHandle;
    StartContainer(container, !(start_dsrc || merge_dsrc));
    for (auto upstr: container->upstreams) {
        if (start_dsrc) {
            StartContainer(upstr, false);
            SyncDatasource(container, upstr);
        } else if (merge_dsrc) {
            SyncDatasource(upstr, container);
            StopContainer(upstr, old_device);
        } else {
            AdjustUpstream(container->recv_port, upstr, device, container->pipelineModel->name);
        }
    }
    StopContainer(container, old_device);
    spdlog::get("container_agent")->info("Container {0:s} moved to device {1:s}", container->name, device->name);
    old_device->containers.erase(container->name);
}

void Controller::AdjustUpstream(int port, ContainerHandle *upstr, NodeHandle *new_device,
                                const std::string &dwnstr, AdjustUpstreamMode mode) {
    ContainerLink request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_mode(mode);
    request.set_name(upstr->name);
    request.set_downstream_name(dwnstr);
    request.set_ip(new_device->ip);
    request.set_port(port);

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            upstr->device_agent->stub->AsyncUpdateDownstream(&context, request, upstr->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (upstr->device_agent->cq != nullptr) GPR_ASSERT(upstr->device_agent->cq->Next(&got_tag, &ok));
    spdlog::get("container_agent")->info("Upstream of {0:s} adjusted to container {1:s}", dwnstr, upstr->name);
}

void Controller::SyncDatasource(ContainerHandle *prev, ContainerHandle *curr) {
    ContainerLink request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(prev->name);
    request.set_downstream_name(curr->name);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            curr->device_agent->stub->AsyncSyncDatasource(&context, request, curr->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (curr->device_agent->cq != nullptr) GPR_ASSERT(curr->device_agent->cq->Next(&got_tag, &ok));
}

void Controller::AdjustBatchSize(ContainerHandle *msvc, int new_bs) {
    msvc->batch_size = new_bs;
    ContainerInts request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.add_value(new_bs);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (msvc->device_agent->cq != nullptr) GPR_ASSERT(msvc->device_agent->cq->Next(&got_tag, &ok));
    spdlog::get("container_agent")->info("Batch size of {0:s} adjusted to {1:d}", msvc->name, new_bs);
}

void Controller::AdjustCudaDevice(ContainerHandle *msvc, GPUHandle *new_device) {
    msvc->gpuHandle = new_device;
    // TODO: also adjust actual running container
}

void Controller::AdjustResolution(ContainerHandle *msvc, std::vector<int> new_resolution) {
    msvc->dimensions = new_resolution;
    ContainerInts request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.add_value(new_resolution[0]);
    request.add_value(new_resolution[1]);
    request.add_value(new_resolution[2]);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateResolution(&context, request, msvc->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (msvc->device_agent->cq != nullptr) GPR_ASSERT(msvc->device_agent->cq->Next(&got_tag, &ok));
}

void Controller::StopContainer(ContainerHandle *container, NodeHandle *device, bool forced) {
    spdlog::get("container_agent")->info("Stopping container: {0:s}", container->name);
    ContainerSignal request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(container->name);
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            device->stub->AsyncStopContainer(&context, request, container->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (container->device_agent->cq != nullptr) GPR_ASSERT(container->device_agent->cq->Next(&got_tag, &ok));
    if (container->gpuHandle != nullptr)
        container->gpuHandle->removeContainer(container);
    if (!forced) { //not forced means the container is stopped during scheduling and should be removed
        containers.removeContainer(container->name);
        container->device_agent->containers.erase(container->name);
    }
    for (auto upstr: container->upstreams) {
        upstr->downstreams.erase(std::remove(upstr->downstreams.begin(), upstr->downstreams.end(), container), upstr->downstreams.end());
    }
    for (auto dwnstr: container->downstreams) {
        dwnstr->upstreams.erase(std::remove(dwnstr->upstreams.begin(), dwnstr->upstreams.end(), container), dwnstr->upstreams.end());
    }
    spdlog::get("container_agent")->info("Container {0:s} stopped", container->name);
}

/**
 * @brief 
 * 
 * @param node 
 */
void Controller::queryInDeviceNetworkEntries(NodeHandle *node) {
    std::string deviceTypeName = SystemDeviceTypeList[node->type];
    std::string deviceTypeNameAbbr = abbreviate(deviceTypeName);
    if (ctrl_inDeviceNetworkEntries.find(deviceTypeName) == ctrl_inDeviceNetworkEntries.end()) {
        std::string tableName = "prof_" + deviceTypeNameAbbr + "_netw";
        std::string sql = absl::StrFormat("SELECT p95_transfer_duration_us, p95_total_package_size_b "
                                    "FROM %s ", tableName);
        pqxx::result res = pullSQL(*ctrl_metricsServerConn, sql);
        if (res.empty()) {
            spdlog::get("container_agent")->error("No in-device network entries found for device type {}.", deviceTypeName);
            return;
        }
        for (pqxx::result::const_iterator row = res.begin(); row != res.end(); ++row) {
            std::pair<uint32_t, uint64_t> entry = {row["p95_total_package_size_b"].as<uint32_t>(), row["p95_transfer_duration_us"].as<uint64_t>()};
            ctrl_inDeviceNetworkEntries[deviceTypeName].emplace_back(entry);
        }
        spdlog::get("container_agent")->info("Finished querying in-device network entries for device type {}.", deviceTypeName);
    }
    std::unique_lock lock(node->nodeHandleMutex);
    node->latestNetworkEntries[deviceTypeName] = aggregateNetworkEntries(ctrl_inDeviceNetworkEntries[deviceTypeName]);
    std::cout << node->latestNetworkEntries[deviceTypeName].size() << std::endl;
}

/**
 * @brief 
 * 
 * @param container calculating queue sizes for the container before its official deployment.
 * @param modelType 
 */
void Controller::calculateQueueSizes(ContainerHandle &container, const ModelType modelType) {
    float preprocessRate = 1000000.f / container.expectedPreprocessLatency; // queries per second
    float postprocessRate = 1000000.f / container.expectedPostprocessLatency; // qps
    float inferRate = 1000000.f / (container.expectedInferLatency * container.batch_size); // batch per second

    QueueLengthType minimumQueueSize = 30;

    // Receiver to Preprocessor
    // Utilization of preprocessor
    float preprocess_rho = container.arrival_rate / preprocessRate;
    QueueLengthType preprocess_inQueueSize = std::max((QueueLengthType) std::ceil(preprocess_rho * preprocess_rho / (2 * (1 - preprocess_rho))), minimumQueueSize);
    float preprocess_thrpt = std::min(preprocessRate, container.arrival_rate);

    // Preprocessor to Inferencer
    // Utilization of inferencer
    float infer_rho = preprocess_thrpt / container.batch_size / inferRate;
    QueueLengthType infer_inQueueSize = std::max((QueueLengthType) std::ceil(infer_rho * infer_rho / (2 * (1 - infer_rho))), minimumQueueSize);
    float infer_thrpt = std::min(inferRate, preprocess_thrpt / container.batch_size); // batch per second

    float postprocess_rho = (infer_thrpt * container.batch_size) / postprocessRate;
    QueueLengthType postprocess_inQueueSize = std::max((QueueLengthType) std::ceil(postprocess_rho * postprocess_rho / (2 * (1 - postprocess_rho))), minimumQueueSize);
    float postprocess_thrpt = std::min(postprocessRate, infer_thrpt * container.batch_size);

    QueueLengthType sender_inQueueSize = postprocess_inQueueSize * container.batch_size;

    container.queueSizes = {preprocess_inQueueSize, infer_inQueueSize, postprocess_inQueueSize, sender_inQueueSize};

    container.expectedThroughput = postprocess_thrpt;
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //








// ============================================================ Communication Handlers ============================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //

void Controller::HandleRecvRpcs() {
    new DeviseAdvertisementHandler(&service, cq.get(), this);
    new DummyDataRequestHandler(&service, cq.get(), this);
    void *tag;
    bool ok;
    while (running) {
        if (!cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::DeviseAdvertisementHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestAdvertiseToController(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new DeviseAdvertisementHandler(service, cq, controller);
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), DEVICE_CONTROL_PORT + controller->ctrl_port_offset);
        std::string deviceName = request.device_name();
        NodeHandle *node = new NodeHandle{deviceName,
                                     request.ip_address(),
                                          ControlCommands::NewStub(
                                             grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<SystemDeviceType>(request.device_type()),
                                     DATA_BASE_PORT + controller->ctrl_port_offset, {}};
        reply.set_name(controller->ctrl_systemName);
        reply.set_experiment(controller->ctrl_experimentName);
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
        controller->initialiseGPU(node, request.processors(), std::vector<int>(request.memory().begin(), request.memory().end()));
        controller->devices.addDevice(deviceName, node);
        spdlog::get("container_agent")->info("Device {} is connected to the system", request.device_name());
        controller->queryInDeviceNetworkEntries(controller->devices.getDevice(deviceName));

        if (node->type != SystemDeviceType::Server) {
            std::thread networkCheck(&Controller::initNetworkCheck, controller, std::ref(*(controller->devices.getDevice(deviceName))), 1000, 300000, 30);
            networkCheck.detach();
        }
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::DummyDataRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendDummyData(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new DummyDataRequestHandler(service, cq, controller);
        ClockType now = std::chrono::system_clock::now();
        unsigned long diff = std::chrono::duration_cast<TimePrecisionType>(
                now - std::chrono::time_point<std::chrono::system_clock>(TimePrecisionType(request.gen_time()))).count();
        unsigned int size = request.data().size();
        controller->network_check_buffer[request.origin_name()].push_back({size, diff});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

std::string DeviceNameToType(std::string name) {
    if (name == "server" || name == "sink") {
        return "server";
    } else {
        return name.substr(0, name.size() - 1);
    }
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //














// ============================================================ Network Conditions ============================================================ //

// void Controller::optimizeBatchSizeStep(
//         const Pipeline &models,
//         std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects) {
//     ModelType candidate;
//     int max_saving = 0;
//     std::vector<ModelType> blacklist;
//     for (const auto &m: models) {
//         int saving;
//         if (max_saving == 0) {
//             saving =
//                     estimated_infer_times[m.first] - InferTimeEstimator(m.first, batch_sizes[m.first] * 2);
//         } else {
//             if (batch_sizes[m.first] == 64 ||
//                 std::find(blacklist.begin(), blacklist.end(), m.first) != blacklist.end()) {
//                 continue;
//             }
//             for (const auto &d: m.second) {
//                 if (batch_sizes[d.first] > batch_sizes[m.first]) {
//                     blacklist.push_back(d.first);
//                 }
//             }
//             saving = estimated_infer_times[m.first] -
//                      (InferTimeEstimator(m.first, batch_sizes[m.first] * 2) * (nObjects / batch_sizes[m.first] * 2));
//         }
//         if (saving > max_saving) {
//             max_saving = saving;
//             candidate = m.first;
//         }
//     }
//     batch_sizes[candidate] *= 2;
//     estimated_infer_times[candidate] -= max_saving;
// }

// double Controller::LoadTimeEstimator(const char *model_path, double input_mem_size) {
//     // Load the pre-trained model
//     BoosterHandle booster;
//     int num_iterations = 1;
//     int ret = LGBM_BoosterCreateFromModelfile(model_path, &num_iterations, &booster);

//     // Prepare the input data
//     std::vector<double> input_data = {input_mem_size};

//     // Perform inference
//     int64_t out_len;
//     std::vector<double> out_result(1);
//     ret = LGBM_BoosterPredictForMat(booster,
//                                     input_data.data(),
//                                     C_API_DTYPE_FLOAT64,
//                                     1,  // Number of rows
//                                     1,  // Number of columns
//                                     1,  // Is row major
//                                     C_API_PREDICT_NORMAL,  // Predict type
//                                     0,  // Start iteration
//                                     -1,  // Number of iterations, -1 means use all
//                                     "",  // Parameter
//                                     &out_len,
//                                     out_result.data());
//     if (ret != 0) {
//         std::cout << "Failed to perform inference!" << std::endl;
//         exit(ret);
//     }

//     // Print the predicted value
//     std::cout << "Predicted value: " << out_result[0] << std::endl;

//     // Free the booster handle
//     LGBM_BoosterFree(booster);

//     return out_result[0];
// }


/**
 * @brief
 *
 * @param model to specify model
 * @param batch_size for targeted batch size (binary)
 * @return int for inference time per full batch in nanoseconds
 */
int Controller::InferTimeEstimator(ModelType model, int batch_size) {
    return 0;
}

// std::map<ModelType, std::vector<int>> Controller::InitialRequestCount(const std::string &input, const Pipeline &models,
//                                                                       int fps) {
//     std::map<ModelType, std::vector<int>> request_counts = {};
//     std::vector<int> fps_values = {fps, fps * 3, fps * 7, fps * 15, fps * 30, fps * 60};

//     request_counts[models[0].first] = fps_values;
//     json objectCount = json::parse(std::ifstream("../jsons/object_count.json"))[input];

//     for (const auto &m: models) {
//         if (m.first == ModelType::Sink) {
//             request_counts[m.first] = std::vector<int>(6, 0);
//             continue;
//         }

//         for (const auto &d: m.second) {
//             if (d.second == -1) {
//                 request_counts[d.first] = request_counts[m.first];
//             } else {
//                 std::vector<int> objects = (d.second == 0 ? objectCount["person"]
//                                                           : objectCount["car"]).get<std::vector<int>>();

//                 for (int j: fps_values) {
//                     int count = std::accumulate(objects.begin(), objects.begin() + j, 0);
//                     request_counts[d.first].push_back(request_counts[m.first][0] * count);
//                 }
//             }
//         }
//     }
//     return request_counts;
// }

/**
 * @brief '
 * 
 * @param node 
 * @param minPacketSize bytes
 * @param maxPacketSize bytes
 * @param numLoops 
 * @return NetworkEntryType 
 */
NetworkEntryType Controller::initNetworkCheck(NodeHandle &node, uint32_t minPacketSize, uint32_t maxPacketSize, uint32_t numLoops) {
    if (!node.networkCheckMutex.try_lock()) {
        return {};
    }
    LoopRange request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_min(minPacketSize);
    request.set_max(maxPacketSize);
    request.set_repetitions(numLoops);
    try {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                node.stub->AsyncExecuteNetworkTest(&context, request, node.cq));
        rpc->Finish(&reply, &status, (void *)1);
        void *got_tag;
        bool ok = false;
        if (node.cq != nullptr) GPR_ASSERT(node.cq->Next(&got_tag, &ok));
        spdlog::get("container_agent")->info("Successfully started network check for device {}.", node.name);
    } catch (const std::exception &e) {
        spdlog::get("container_agent")->error("Error while starting network check for device {}.", node.name);
    }

    while (network_check_buffer[node.name].size() < numLoops) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    NetworkEntryType entries = network_check_buffer[node.name];
    entries = aggregateNetworkEntries(entries);
    network_check_buffer[node.name].clear();
    spdlog::get("container_agent")->info("Finished network check for device {}.", node.name);
    std::lock_guard lock(node.nodeHandleMutex);
    node.initialNetworkCheck = true;
    node.latestNetworkEntries["server"] = entries;
    node.lastNetworkCheckTime = std::chrono::system_clock::now();
    node.networkCheckMutex.unlock();
    return entries;
};

/**
 * @brief Query the latest network entries for each device to determine the network conditions.
 * If no such entries exists, send to each device a request for network testing.
 * 
 */
void Controller::checkNetworkConditions() {
    std::this_thread::sleep_for(TimePrecisionType(5 * 1000000));
    while (running) {
        Stopwatch stopwatch;
        stopwatch.start();
        std::map<std::string, NetworkEntryType> networkEntries = {};

        
        for (auto [deviceName, nodeHandle] : devices.getMap()) {
            std::unique_lock<std::mutex> lock(nodeHandle->nodeHandleMutex);
            bool initialNetworkCheck = nodeHandle->initialNetworkCheck;
            uint64_t timeSinceLastCheck = std::chrono::duration_cast<TimePrecisionType>(
                    std::chrono::system_clock::now() - nodeHandle->lastNetworkCheckTime).count() / 1000000;
            lock.unlock();
            if (nodeHandle->type == SystemDeviceType::Server || (initialNetworkCheck && timeSinceLastCheck < 60)) {
                spdlog::get("container_agent")->info("Skipping network check for device {}.", deviceName);
                continue;
            }
            initNetworkCheck(*nodeHandle, 1000, 300000, 30);
        }

        stopwatch.stop();
        uint64_t sleepTimeUs = 60 * 1000000 - stopwatch.elapsed_microseconds();
        std::this_thread::sleep_for(TimePrecisionType(sleepTimeUs));
    }
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice, const std::string &pipelineName, const std::string &streamName) {
    std::string sourceName = streamName;
    if (ctrl_initialRequestRates.find(sourceName) == ctrl_initialRequestRates.end()) {
        for (auto [key, rates]: ctrl_initialRequestRates) {
            if (key.find(pipelineName) != std::string::npos) {
                sourceName = key;
                break;
            }
        }
    }
    switch (type) {
        case PipelineType::Traffic: {
            auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s= new PipelineModel{
                        "server",
                        "yolov5s",
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            if (ctrl_systemName == "jlf") {
                retina1face->possibleDevices = {"server"};
                retina1face->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({retina1face, 0});
                yolov5n512->downstreams.push_back({retina1face, 0});
                yolov5s->downstreams.push_back({retina1face, 0});
            }

            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            arcface->possibleDevices = {"server"};
            retina1face->downstreams.push_back({arcface, -1});

            auto *carbrand = new PipelineModel{
                    "server",
                    "carbrand",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}
            };
            carbrand->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({carbrand, 2});

            if (ctrl_systemName == "jlf") {
                carbrand->possibleDevices = {"server"};
                carbrand->upstreams = {{yolov5n, 2}, {yolov5n320, 2}, {yolov5n512, 2}, {yolov5s, 2}};
                yolov5n320->downstreams.push_back({carbrand, 2});
                yolov5n512->downstreams.push_back({carbrand, 2});
                yolov5s->downstreams.push_back({carbrand, 2});
            }

            auto *platedet = new PipelineModel{
                    "server",
                    "platedet",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}
            };
            platedet->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({platedet, 2});

            if (ctrl_systemName == "jlf") {
                platedet->possibleDevices = {"server"};
                platedet->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({platedet, 2});
                yolov5n512->downstreams.push_back({platedet, 2});
                yolov5s->downstreams.push_back({platedet, 2});
            }

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{arcface, -1}, {carbrand, -1}, {platedet, -1}}
            };
            sink->possibleDevices = {"sink"};
            arcface->downstreams.push_back({sink, -1});
            carbrand->downstreams.push_back({sink, -1});
            platedet->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
                carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carbrand->name];
                platedet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][platedet->name];
            }

            if (ctrl_systemName == "jlf") {
                arcface->possibleDevices = {"server"};
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, arcface, carbrand, platedet, sink};
            }
            return {datasource, yolov5n, retina1face, arcface, carbrand, platedet, sink};
        }
        case PipelineType::Building_Security: {
            auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s= new PipelineModel{
                        "server",
                        "yolov5s",
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            if (ctrl_systemName == "jlf") {
                retina1face->possibleDevices = {"server"};
                retina1face->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({retina1face, 0});
                yolov5n512->downstreams.push_back({retina1face, 0});
                yolov5s->downstreams.push_back({retina1face, 0});
            }

            auto *movenet = new PipelineModel{
                    "server",
                    "movenet",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            movenet->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({movenet, 0});

            if (ctrl_systemName == "jlf") {
                movenet->possibleDevices = {"server"};
                movenet->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({movenet, 0});
                yolov5n512->downstreams.push_back({movenet, 0});
                yolov5s->downstreams.push_back({movenet, 0});
            }

            auto *gender = new PipelineModel{
                    "server",
                    "gender",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            gender->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({gender, -1});

            auto *age = new PipelineModel{
                    "server",
                    "age",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            age->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({age, -1});

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{gender, -1}, {age, -1}, {movenet, -1}}
            };
            sink->possibleDevices = {"sink"};
            gender->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            movenet->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                movenet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][movenet->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][movenet->name];
                age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][age->name];
            }

            if (ctrl_systemName == "jlf") {
                gender->possibleDevices = {"server"};
                age->possibleDevices = {"server"};
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, movenet, gender, age, sink};
            }
            return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
        }
        case PipelineType::Video_Call: {
            auto *datasource = new PipelineModel{startDevice, "datasource", {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};
            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            retina1face->possibleDevices = {"server"};
            datasource->downstreams.push_back({retina1face, -1});

            auto *emotionnet = new PipelineModel{
                    "server",
                    "emotionnet",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            emotionnet->possibleDevices = {"server"};
            retina1face->downstreams.push_back({emotionnet, -1});

            auto *age = new PipelineModel{
                    "server",
                    "age",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            age->possibleDevices = {"server"};
            retina1face->downstreams.push_back({age, -1});

            auto *gender = new PipelineModel{
                    "server",
                    "gender",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            gender->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({gender, -1});

            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            arcface->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({arcface, -1});

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{emotionnet, -1}, {age, -1}, {gender, -1}, {arcface, -1}}
            };
            sink->possibleDevices = {"sink"};
            emotionnet->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            gender->downstreams.push_back({sink, -1});
            arcface->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                emotionnet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][emotionnet->name];
                age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][age->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][gender->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
            }

            return {datasource, retina1face, emotionnet, age, gender, arcface, sink};
        }
        default:
            return {};
    }
}

PipelineModelListType deepCopyPipelineModelList(const PipelineModelListType& original) {
    PipelineModelListType newList;
    newList.reserve(original.size());
    for (const auto* model : original) {
        newList.push_back(new PipelineModel(*model));
    }
    return newList;
}
