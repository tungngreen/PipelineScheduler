#include "controller.h"

ABSL_FLAG(std::string, ctrl_configPath, "../jsons/experiments/base-experiment.json",
          "Path to the configuration file for this experiment.");
ABSL_FLAG(uint16_t, ctrl_verbose, 0, "Verbosity level of the controller.");
ABSL_FLAG(uint16_t, ctrl_loggingMode, 0, "Logging mode of the controller. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, ctrl_logPath, "../logs", "Path to the log dir for the controller.");


// =========================================================GPU Lanes/Portions Control===========================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

void Controller::initiateGPULanes(NodeHandle &node) {
    // Currently only support powerful GPUs capable of running multiple models in parallel
    if (node.name == "sink") {
        return;
    }
    auto deviceList = devices.getMap();

    if (deviceList.find(node.name) == deviceList.end()) {
        spdlog::get("container_agent")->error("Device {0:s} is not found in the device list", node.name);
        return;
    }

    // TODO: Check if we can remove this if
    if (node.type == Server || node.type == Virtual) {
        node.numGPULanes = NUM_LANES_PER_GPU * NUM_GPUS;
    } else {
        node.numGPULanes = 1;
    }
    node.gpuHandles.clear();
    node.freeGPUPortions.list.clear();

    for (unsigned short i = 0; i < node.numGPULanes; i++) {
        GPULane *gpuLane = new GPULane{node.gpuHandles[i / NUM_LANES_PER_GPU], &node, i};
        node.gpuLanes.push_back(gpuLane);
        // Initially the number of portions is the number of lanes'
        GPUPortion *portion = new GPUPortion{gpuLane};
        node.freeGPUPortions.list.push_back(portion);
        // This is currently the only portion in a lane, later when it is divided
        // we need to keep track of the portions in the lane to be able to recover the free portions
        // when the container is removed.
        portion->nextInLane = nullptr;
        portion->prevInLane = nullptr;

        gpuLane->portionList.list.push_back(portion);
        gpuLane->portionList.head = portion;

        // TODO: HANDLE FREE PORTIONS WITHIN THE GPU
        // gpuLane->gpuHandle->freeGPUPortions.push_back(portion);

        if (i == 0) {
            node.freeGPUPortions.head = portion;
            portion->prev = nullptr;
        } else {
            node.freeGPUPortions.list[i - 1]->next = portion;
            portion->prev = node.freeGPUPortions.list[i - 1];
        }
        portion->next = nullptr;
    }
}

GPULane::GPULane(GPUHandle *gpu, NodeHandle *device, uint16_t laneNum) : gpuHandle(gpu), node(device), laneNum(laneNum) {
    dutyCycle = 0;
    portionList.head = nullptr;
    portionList.list = {};
}

bool GPUPortion::assignContainer(ContainerHandle *cont) {
    if (this->container != nullptr) {
        spdlog::get("console")->error("Portion already assigned to container {0:s}", this->container->name);
        return false;
    }
    cont->executionPortion = this;
    this->container = cont;
    start = cont->startTime;
    end = cont->endTime;

    spdlog::get("container_agent")->info("Portion assigned to container {0:s}", cont->name);
    return true;
}

/**
 * @brief insert the newly created free portion into the sorted list of free portions
 * Since the list is sorted, we can insert the new portion by traversing the list from the head
 * Complexity: O(n)
 *
 * @param head
 * @param freePortion
 */
void Controller::insertFreeGPUPortion(GPUPortionList &portionList, GPUPortion *freePortion) {
    auto &head = portionList.head;
    if (head == nullptr) {
        head = freePortion;
        return;
    }
    GPUPortion *curr = head;
    auto it = portionList.list.begin();
    while (true) {
        if ((curr->end - curr->start) >= (freePortion->end - freePortion->start)) {
            if (curr == head) {
                freePortion->next = curr;
                curr->prev = freePortion;
                head = freePortion;
                portionList.list.insert(it, freePortion);
                return;
            } else if ((curr->prev->end - curr->prev->start) < (freePortion->end - freePortion->start)) {
                freePortion->next = curr;
                freePortion->prev = curr->prev;
                curr->prev = freePortion;
                freePortion->prev->next = freePortion;
                portionList.list.insert(it, freePortion);
                return;
            }
        } else {
            if (curr->next == nullptr) {
                curr->next = freePortion;
                freePortion->prev = curr;
                portionList.list.push_back(freePortion);
                return;
            } else if ((curr->next->end - curr->next->start) > (freePortion->end - freePortion->start)) {
                freePortion->next = curr->next;
                freePortion->prev = curr;
                curr->next = freePortion;
                freePortion->next->prev = freePortion;
                portionList.list.insert(it + 1, freePortion);
                return;
            } else {
                curr = curr->next;
            }
        }
        it++;
    }
}

GPUPortion* Controller::findFreePortionForInsertion(GPUPortionList &portionList, ContainerHandle *container) {
    auto &head = portionList.head;
    GPUPortion *curr = head;
    while (true) {
        auto laneDutyCycle = curr->lane->dutyCycle;
        if (curr->start <= container->startTime &&
            curr->end >= container->endTime &&
            container->pipelineModel->localDutyCycle >= laneDutyCycle) {
            return curr;
        }
        if (curr->next == nullptr) {
            return nullptr;
        }
        curr = curr->next;
    }
}

/**
 * @brief
 *
 * @param node
 * @param scheduledPortion
 * @param toBeDividedFreePortion
 */
std::pair<GPUPortion *, GPUPortion *> Controller::insertUsedGPUPortion(GPUPortionList &portionList, ContainerHandle *container, GPUPortion *toBeDividedFreePortion) {
    auto gpuLane = toBeDividedFreePortion->lane;
    GPUPortion *usedPortion = new GPUPortion{gpuLane};
    usedPortion->assignContainer(container);
    gpuLane->portionList.list.push_back(usedPortion);

    usedPortion->nextInLane = toBeDividedFreePortion->nextInLane;
    usedPortion->prevInLane = toBeDividedFreePortion->prevInLane;
    if (toBeDividedFreePortion->prevInLane != nullptr) {
        toBeDividedFreePortion->prevInLane->nextInLane = usedPortion;
    }
    if (toBeDividedFreePortion->nextInLane != nullptr) {
        toBeDividedFreePortion->nextInLane->prevInLane = usedPortion;
    }

    auto &head = portionList.head;
    // new portion on the left
    uint64_t newStart = toBeDividedFreePortion->start;
    uint64_t newEnd = container->startTime;

    GPUPortion* leftPortion = nullptr;
    bool goodLeft = false;
    GPUPortion* rightPortion = nullptr;
    bool goodRight = false;
    // Create a new portion on the left only if it is large enough
    if (newEnd - newStart > 0) {
        leftPortion = new GPUPortion{};
        leftPortion->start = newStart;
        leftPortion->end = newEnd;
        leftPortion->lane = gpuLane;
        gpuLane->portionList.list.push_back(leftPortion);
        leftPortion->prevInLane = toBeDividedFreePortion->prevInLane;
        leftPortion->nextInLane = usedPortion;
        usedPortion->prevInLane = leftPortion;
        if (toBeDividedFreePortion == gpuLane->portionList.head) {
            gpuLane->portionList.head = leftPortion;
        }
        if (newEnd - newStart >= MINIMUM_PORTION_SIZE) {
            goodLeft = true;
            // TODO: HANDLE FREE PORTIONS WITHIN THE GPU
            // gpu->freeGPUPortions.push_back(leftPortion);
        }
    }
    if (toBeDividedFreePortion == gpuLane->portionList.head && !goodLeft) {
        gpuLane->portionList.head = usedPortion;
    }

    // new portion on the right
    newStart = container->endTime;
    auto laneDutyCycle = gpuLane->dutyCycle;
    if (laneDutyCycle == 0) {
        if (container->pipelineModel->localDutyCycle == 0) {
            throw std::runtime_error("Duty cycle of the container 0");
        }
        int64_t slack = container->pipelineModel->task->tk_slo - container->pipelineModel->localDutyCycle * 2;
        if (slack < 0) {
            throw std::runtime_error("Slack is negative. Duty cycle is larger than the SLO");
        }
        laneDutyCycle = container->pipelineModel->localDutyCycle;
        newEnd = container->pipelineModel->localDutyCycle;
    } else {
        newEnd = toBeDividedFreePortion->end;
    }
    // Create a new portion on the right only if it is large enough
    if (newEnd - newStart > 0) {
        rightPortion = new GPUPortion{};
        rightPortion->start = newStart;
        rightPortion->end = newEnd;
        rightPortion->lane = gpuLane;
        gpuLane->portionList.list.push_back(rightPortion);
        rightPortion->nextInLane = toBeDividedFreePortion->nextInLane;
        rightPortion->prevInLane = usedPortion;
        usedPortion->nextInLane = rightPortion;
        if (newEnd - newStart >= MINIMUM_PORTION_SIZE) {
            goodRight = true;
            // TODO: HANDLE FREE PORTIONS WITHIN THE GPU
            // gpu->freeGPUPortions.push_back(rightPortion);
        }
    }

    gpuLane->dutyCycle = laneDutyCycle;

    auto it = std::find(portionList.list.begin(), portionList.list.end(), toBeDividedFreePortion);
    portionList.list.erase(it);
    // TODO: HANDLE FREE PORTIONS WITHIN THE GPU
    // it = std::find(gpu->freeGPUPortions.begin(), gpu->freeGPUPortions.end(), toBeDividedFreePortion);
    // gpu->freeGPUPortions.erase(it);
    it = std::find(gpuLane->portionList.list.begin(), gpuLane->portionList.list.end(), toBeDividedFreePortion);
    gpuLane->portionList.list.erase(it);



    // Delete the old portion as it has been divided into two new free portions and an occupied portion
    if (toBeDividedFreePortion->prev != nullptr) {
        toBeDividedFreePortion->prev->next = toBeDividedFreePortion->next;
    } else {
        head = toBeDividedFreePortion->next;
    }
    if (toBeDividedFreePortion->next != nullptr) {
        toBeDividedFreePortion->next->prev = toBeDividedFreePortion->prev;
    }
    delete toBeDividedFreePortion;

    if (goodLeft) {
        insertFreeGPUPortion(portionList, leftPortion);
    }

    if (goodRight) {
        insertFreeGPUPortion(portionList, rightPortion);
    }

    return {leftPortion, rightPortion};
}

/**
 * @brief Remove a free GPU portion from the list of free portions
 * This happens when a container is removed from the system and its portion is reclaimed
 * and merged with the free portions on the left and right.
 * These left and right portions are to be removed from the list of free portions.
 *
 * @param portionList
 * @param toBeRemovedPortion
 * @return true
 * @return false
 */
bool Controller::removeFreeGPUPortion(GPUPortionList &portionList, GPUPortion *toBeRemovedPortion) {
    if (toBeRemovedPortion == nullptr) {
        spdlog::get("container_agent")->error("Portion to be removed doesn't exist");
        return false;
    }
    auto container = toBeRemovedPortion->container;
    if (container != nullptr) {
        spdlog::get("container_agent")->error("Portion to be removed is being used by container {0:s}", container->name);
        return false;
    }
    auto &head = portionList.head;
    auto it = std::find(portionList.list.begin(), portionList.list.end(), toBeRemovedPortion);
    if (it == portionList.list.end()) {
        spdlog::get("container_agent")->error("Portion to be removed not found in the list of free portions");
        return false;
    }
    portionList.list.erase(it);

    if (toBeRemovedPortion->prev != nullptr) {
        toBeRemovedPortion->prev->next = toBeRemovedPortion->next;
    } else {
        if (toBeRemovedPortion != head) {
            throw std::runtime_error("Portion is not the head of the list but its previous is null");
        }
        head = toBeRemovedPortion->next;
    }
    if (toBeRemovedPortion->next != nullptr) {
        toBeRemovedPortion->next->prev = toBeRemovedPortion->prev;
    }

    // auto gpuHandle = toBeRemovedPortion->lane->gpuHandle;
    // it = std::find(gpuHandle->freeGPUPortions.begin(), gpuHandle->freeGPUPortions.end(), toBeRemovedPortion);
    // gpuHandle->freeGPUPortions.erase(it);
    spdlog::get("container_agent")->info("Portion from {0:d} to {1:d} removed from the list of free portions of lane {2:d}",
                                         toBeRemovedPortion->start,
                                         toBeRemovedPortion->end,
                                         toBeRemovedPortion->lane->laneNum);
    delete toBeRemovedPortion;
    return true;
}

/**
 * @brief
 *
 * @param toBeReclaimedPortion
 * @return true
 * @return false
 */
bool Controller::reclaimGPUPortion(GPUPortion *toBeReclaimedPortion) {
    if (toBeReclaimedPortion == nullptr) {
        throw std::runtime_error("Portion to be reclaimed is null");
    }

    spdlog::get("container_agent")->info("Reclaiming portion from {0:d} to {1:d} in lane {2:d}",
                                         toBeReclaimedPortion->start,
                                         toBeReclaimedPortion->end,
                                         toBeReclaimedPortion->lane->laneNum);
    if (toBeReclaimedPortion->container != nullptr) {
        spdlog::get("container_agent")->warn("Portion is being used by container {0:s}", toBeReclaimedPortion->container->name);
    }

    GPULane *gpuLane = toBeReclaimedPortion->lane;
    NodeHandle *node = gpuLane->node;

    /**
     * @brief Organizing the lsit of portions in the lane the container is currently using

     *
     */
    GPUPortion *leftInLanePortion = toBeReclaimedPortion->prevInLane;
    GPUPortion *rightInLanePortion = toBeReclaimedPortion->nextInLane;

    // No container is using the portion now
    toBeReclaimedPortion->container = nullptr;

    // Resetting its left boundary by merging it with the left portion if it is free
    if (leftInLanePortion == nullptr) {
        toBeReclaimedPortion->start = 0;
        spdlog::get("container_agent")->trace("The portion to be reclaimed is the head of the list of portions in the lane.");
        if (gpuLane->portionList.head != toBeReclaimedPortion) {
            throw std::runtime_error("Left portion is null but the portion is not the head of the list");
        }
    } else {
        if (leftInLanePortion->container != nullptr) {
            spdlog::get("container_agent")->trace("Left portion is occupied.");
        } else {
            spdlog::get("container_agent")->trace("Left portion was free and is merged with the reclaimed portion.");
            /**
             * @brief Merging the left portion with the portion to be reclaimed in a lane context
             * Removing the left portion from the list of portions in the lane
             *
             */

            // Whatever was on the left of the left portion will now be on the left of the portion to be reclaimed
            toBeReclaimedPortion->prevInLane = leftInLanePortion->prevInLane;
            // AFter merging, the portion to be reclaimed will have the start of the left portion
            toBeReclaimedPortion->start = leftInLanePortion->start;
            // If the left portion was the head of the list, the portion to be reclaimed will be the new head
            if (leftInLanePortion == gpuLane->portionList.head) {
                gpuLane->portionList.head = toBeReclaimedPortion;
            }
            auto it = std::find(gpuLane->portionList.list.begin(), gpuLane->portionList.list.end(), leftInLanePortion);
            gpuLane->portionList.list.erase(it);

            /**
             * @brief Removing the left portion from the list of free portions as it is now merged with the portion to be reclaimed
             * to create a bigger free portion
             *
             */

            removeFreeGPUPortion(node->freeGPUPortions, leftInLanePortion);
        }
    }

    // Resetting its right boundary by merging it with the right portion if it is free

    if (rightInLanePortion == nullptr) {
    } else {
        if (rightInLanePortion->container != nullptr) {
            spdlog::get("container_agent")->trace("Right portion is occupied.");
        } else {
            spdlog::get("container_agent")->trace("Right portion was free and is merged with the reclaimed portion.");
            /**
             * @brief Merging the right portion with the portion to be reclaimed in a lane context
             * Removing the right portion from the list of portions in the lane
             *
             */

            // Whatever was on the right of the right portion will now be on the right of the portion to be reclaimed
            toBeReclaimedPortion->nextInLane = rightInLanePortion->nextInLane;
            // AFter merging, the portion to be reclaimed will have the end of the right portion
            toBeReclaimedPortion->end = rightInLanePortion->end;

            if (rightInLanePortion == gpuLane->portionList.head) {
                gpuLane->portionList.head = rightInLanePortion->next;
            }
            auto it = std::find(gpuLane->portionList.list.begin(), gpuLane->portionList.list.end(), rightInLanePortion);
            gpuLane->portionList.list.erase(it);

            /**
             * @brief Removing the right portion from the list of free portions as it is now merged with the portion to be reclaimed
             * to create a bigger free portion
             *
             */
            removeFreeGPUPortion(node->freeGPUPortions, rightInLanePortion);
        }
    }

    if (toBeReclaimedPortion->prevInLane == nullptr) {
        toBeReclaimedPortion->start = 0;
    }
    // Recover the lane's original structure if the portion to be reclaimed is the only portion in the lane
    if (toBeReclaimedPortion->nextInLane == nullptr && toBeReclaimedPortion->start == 0) {
        toBeReclaimedPortion->end = MAX_PORTION_SIZE;
        gpuLane->dutyCycle = 0;
    }

    // Insert the reclaimed portion into the free portion list
    insertFreeGPUPortion(node->freeGPUPortions, toBeReclaimedPortion);

    return true;
}

bool GPULane::removePortion(GPUPortion *portion) {
    if (portion->lane != this) {
        throw std::runtime_error("Lane %d cannot remove portion %s, which does not belong to it." + portion->container->name + std::to_string(laneNum));
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

// ============================================================ Configurations ============================================================ //
// ======================================================================================================================================== //
// ======================================================================================================================================== //
// ======================================================================================================================================== //

void Controller::readConfigFile(const std::string &path) {
    std::ifstream file(path);
    json j = json::parse(file);

    ctrl_experimentName = j["expName"];
    ctrl_systemName = j["systemName"];
    ctrl_runtime = j["runtime"];
    ctrl_clusterCount = j["cluster_count"];
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
    if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") ctrl_fcpo_config = j["fcpo_parameters"];
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val) {
    j.at("pipeline_name").get_to(val.name);
    j.at("pipeline_target_slo").get_to(val.slo);
    j.at("pipeline_type").get_to(val.type);
    j.at("video_source").get_to(val.stream);
    j.at("pipeline_source_device").get_to(val.srcDevice);
    if (j.contains("pipeline_edge_node"))
        j.at("pipeline_edge_node").get_to(val.edgeNode);
    else
        val.edgeNode = "server";
    val.fullName = val.name + "_" + val.srcDevice;
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


// ============================================================= Con/Destructors ============================================================= //
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
        sql = "CREATE SCHEMA IF NOT EXISTS " + ctrl_metricsServerConfigs.schema + ";";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + 
              " GRANT ALL PRIVILEGES ON TABLES TO controller;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " GRANT SELECT, INSERT ON TABLES TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "GRANT USAGE, CREATE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
    }


    if (ctrl_systemName != "fcpo" && ctrl_systemName != "bce") {
        std::thread networkCheckThread(&Controller::checkNetworkConditions, this);
        networkCheckThread.detach();
    }

    running = true;
    ctrl_clusterID = 0;

    std::string server_address = absl::StrFormat("tcp://*:%d", CONTROLLER_API_PORT);
    api_ctx = context_t(1);
    api_socket = socket_t(api_ctx, ZMQ_REP);
    api_socket.bind(server_address);
    api_socket.set(zmq::sockopt::rcvtimeo, 1000);

    server_address = absl::StrFormat("tcp://*:%d", CONTROLLER_RECEIVE_PORT + ctrl_port_offset);
    system_ctx = context_t(2);
    server_socket = socket_t(system_ctx, ZMQ_REP);
    server_socket.bind(server_address);
    server_socket.set(zmq::sockopt::rcvtimeo, 1000);
    system_handlers = {
        {MSG_TYPE[DEVICE_ADVERTISEMENT], std::bind(&Controller::handleDeviseAdvertisement, this, std::placeholders::_1)},
        {MSG_TYPE[DUMMY_DATA], std::bind(&Controller::handleDummyDataRequest, this, std::placeholders::_1)},
        {MSG_TYPE[START_FL], std::bind(&Controller::handleForwardFLRequest, this, std::placeholders::_1)},
        {MSG_TYPE[SINK_METRICS], std::bind(&Controller::handleSinkMetrics, this, std::placeholders::_1)}
    };

    server_address = absl::StrFormat("tcp://*:%d", CONTROLLER_MESSAGE_QUEUE_PORT + ctrl_port_offset);
    message_queue = socket_t(system_ctx, ZMQ_PUB);
    message_queue.bind(server_address);
    message_queue.set(zmq::sockopt::sndtimeo, 100);

    // append one device for sink of type server
    NodeHandle *sink_node = new NodeHandle("sink", ctrl_sinkNodeIP,  SystemDeviceType::Server,
                                            DATA_BASE_PORT + ctrl_port_offset, {});
    devices.addDevice("sink", sink_node);

    if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
        ctrl_fcpo_server = new FCPOServer(ctrl_systemName + "_" + ctrl_experimentName, ctrl_fcpo_config, ctrl_clusterCount, &message_queue);
    }

    ctrl_nextSchedulingTime = std::chrono::system_clock::now();
}

Controller::~Controller() {
    running = false;
    if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") ctrl_fcpo_server->stop();
    for (auto msvc: containers.getList()) {
        StopContainer(msvc, msvc->device_agent, true);
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    for (auto &device: devices.getList()) {
        if (device->name == "sink") { continue; }
        sendMessageToDevice(device->name, MSG_TYPE[DEVICE_SHUTDOWN], "");
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
}

// =========================================================== Executor/Maintainers ============================================================ //
// ============================================================================================================================================= //
// ============================================================================================================================================= //
// ============================================================================================================================================= //

MemUsageType ContainerHandle::getExpectedTotalMemUsage() const {
    std::string deviceTypeName = getDeviceTypeName(device_agent->type);
    if (device_agent->type == SystemDeviceType::Virtual || device_agent->type == SystemDeviceType::Server
                || device_agent->type == SystemDeviceType::OnPremise) {
        return pipelineModel->processProfiles.at(deviceTypeName).batchInfer[pipelineModel->batchSize].gpuMemUsage;
    }
    return (pipelineModel->processProfiles.at(deviceTypeName).batchInfer[pipelineModel->batchSize].gpuMemUsage +
            pipelineModel->processProfiles.at(deviceTypeName).batchInfer[pipelineModel->batchSize].rssMemUsage) / 1000;
}

bool Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    TaskHandle *task = new TaskHandle{t.name, t.type, t.stream, t.srcDevice, t.slo, {}};

    std::map<std::string, NodeHandle*> deviceList = devices.getMap();

    if (deviceList.find(t.srcDevice) == deviceList.end()) {
        spdlog::error("Device {0:s} is not connected", t.srcDevice);
        return false;
    }

    while (!deviceList.at(t.srcDevice)->initialNetworkCheck) {
        spdlog::get("container_agent")->info("Waiting for device {0:s} to finish network check", t.srcDevice);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    task->tk_src_device = t.srcDevice;

    task->tk_pipelineModels = getModelsByPipelineType(t.type, t.srcDevice, t.name, t.stream, t.edgeNode);
    for (auto &model: task->tk_pipelineModels) {
        model->datasourceName = {t.stream};
        model->task = task;
    }

    ctrl_savedUnscheduledPipelines.addTask(task->tk_name, task);
    return true;
}

void Controller::initialiseGPU(NodeHandle *node, int numGPUs, std::vector<int> memLimits) {
    if (node->type == SystemDeviceType::Virtual) {
        GPUHandle *gpuNode = new GPUHandle{"3090", node->name, 0, memLimits[0] - 2000, NUM_LANES_PER_GPU, node};
        // TODO: differentiate between virtualEdge and virtualServer
        //GPUHandle *gpuNode = new GPUHandle{node->name, node->name, 0, memLimits[0] / 5, 1, node};
        node->gpuHandles.emplace_back(gpuNode);
    } else if (node->type == SystemDeviceType::Server) {
        for (uint8_t gpuIndex = 0; gpuIndex < numGPUs; gpuIndex++) {
            std::string gpuName = "gpu" + std::to_string(gpuIndex);
            GPUHandle *gpuNode = new GPUHandle{"3090", "server", gpuIndex, memLimits[gpuIndex] - 2000, NUM_LANES_PER_GPU, node};
            node->gpuHandles.emplace_back(gpuNode);
        }
    } else {
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
            MemUsageType biggestGap  = std::numeric_limits<MemUsageType>::min();
            int8_t targetGapIndex = -1;
            for (auto &gpu: gpus) {
                MemUsageType gap = gpu->memLimit - gpu->currentMemUsage - containerMemUsage;
                if (gap < 500) {
                    continue;
                }
                if (gap > biggestGap) {
                    biggestGap = gap;
                    targetGapIndex = gpu->number;
                }
            }
            if (targetGapIndex == -1) {
                spdlog::get("container_agent")->error("No GPU available for container {}", container->name);
                continue;
            }
            gpus[targetGapIndex]->addContainer(container);
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
            if (ctrl_systemName == "tuti" && model->name.find("datasource") == std::string::npos && model->name.find("sink") == std::string::npos) {
                model->numReplicas = 3;
            } else if (ctrl_systemName == "rim" && ctrl_systemName == "dis") {
                model->cudaDevices.emplace_back(0);
                model->numReplicas = 1;
            }
            bool upstreamIsDatasource = (std::find_if(model->upstreams.begin(), model->upstreams.end(),
                                                      [](const std::pair<PipelineModel *, int> &upstream) {
                                                          return upstream.first->canBeCombined && (upstream.first->name.find("datasource") != std::string::npos);
                                                      }) != model->upstreams.end());
            if (model->name.find("yolov5n") != std::string::npos && model->device != "server" && upstreamIsDatasource) {
                if (model->name.find("yolov5ndsrc") == std::string::npos) {
                    model->name = replaceSubstring(model->name, "yolov5n", "yolov5ndsrc");
                }

            } else if (model->name.find("retinamtface") != std::string::npos && model->device != "server" && upstreamIsDatasource) {
                if (model->name.find("retinamtfacedsrc") == std::string::npos) {
                    model->name = replaceSubstring(model->name, "retinamtface", "retinamtfacedsrc");
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
                }
            }
        }
    }
    // Rearranging the upstreams and downstreams for containers;
    for (auto pipe: ctrl_scheduledPipelines.getList()) {
        for (auto &model: pipe->tk_pipelineModels) {
            // If it's a datasource, we don't have to do it now
            // datasource doesn't have upstreams
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

    if (ctrl_systemName != "ppp" && ctrl_systemName != "fcpo" && ctrl_systemName != "bce") {
        basicGPUScheduling(new_containers);
    } else {
        colocationTemporalScheduling();
    }

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
    return m == "datasource" || m == "yolov5n" || m == "retinamtface" || m == "yolov5ndsrc" || m == "retinamtfacedsrc";
}

ContainerHandle *Controller::TranslateToContainer(PipelineModel *model, NodeHandle *device, unsigned int i) {
    if (model->name.find("datasource") != std::string::npos && model->canBeCombined) {
        for (auto &[downstream, coi] : model->downstreams) {
            if (CheckMergable(downstream->name) && downstream->device != "server") {
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
    std::string containerName = ctrl_experimentName + "_" + ctrl_systemName + "_" + device->name + "_" + model->task->tk_name + "_" +
            modelName + "_" + std::to_string(i);
    // the name of the container type to look it up in the container library
    std::string containerTypeName = modelName + "_" + getDeviceTypeName(device->type);
    if (getDeviceTypeName(device->type) == "virtual")
        containerTypeName = modelName + "_server";

    if (ctrl_systemName == "ppp" || ctrl_systemName == "fcpo" || ctrl_systemName== "apis" || ctrl_systemName == "bce") {
        if (model->batchSize < model->datasourceName.size()) model->batchSize = model->datasourceName.size();
    } // ensure minimum global batch size setting for these configurations for a good comparison

    auto *container = new ContainerHandle{containerName, model->position_in_pipeline, i,
                                          class_of_interest,
                                          ModelTypeReverseList[modelName],
                                          CheckMergable(modelName) && model->canBeCombined,
                                          {0},
                                          static_cast<uint64_t>(model->task->tk_slo),
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
    request.set_name(container->name);
    request.set_slo(container->pipelineSLO);
    request.set_time_budget(container->timeBudgetLeft);
    request.set_start_time(container->startTime);
    request.set_end_time(container->endTime);
    request.set_local_duty_cycle(container->localDutyCycle);
    request.set_cycle_start_time(std::chrono::duration_cast<TimePrecisionType>(container->cycleStartTime.time_since_epoch()).count());
    sendMessageToDevice(container->device_agent->name, MSG_TYPE[TIME_KEEPING_UPDATE], request.SerializeAsString());
    spdlog::get("container_agent")->info("Requested container {0:s} to update time keeping!", container->name);
}

void Controller::StartContainer(ContainerHandle *container, bool easy_allocation) {
    spdlog::get("container_agent")->info("Starting container: {0:s}", container->name);
    ContainerConfig request;
    json start_config;
    unsigned int control_port;
    std::string pipelineName = container->task->tk_name;
    ModelType model = static_cast<ModelType>(container->model);
    std::string modelName = getContainerName(container->device_agent->type, model);
    std::cout << "Creating container: " << container->name << std::endl;
    if (model == ModelType::Sink) {
        start_config["experimentName"] = ctrl_experimentName;
        start_config["systemName"] = ctrl_systemName;
        start_config["pipelineName"] = pipelineName;
        start_config["controllerIP"] = "<IP>";
        control_port = container->recv_port;
    } else {
        start_config = ctrl_containerLib[modelName].templateConfig;

        // adjust container configs
        start_config["container"]["cont_experimentName"] = ctrl_experimentName;
        start_config["container"]["cont_systemName"] = ctrl_systemName;
        start_config["container"]["cont_pipeName"] = pipelineName;
        start_config["container"]["cont_hostDevice"] = container->device_agent->name;
        start_config["container"]["cont_hostDeviceType"] = SystemDeviceTypeList[container->device_agent->type];
        start_config["container"]["cont_name"] = container->name;
        start_config["container"]["cont_allocationMode"] = easy_allocation ? 1 : 0;
        if (ctrl_systemName == "ppp" || ctrl_systemName == "bce") {
            //TODO: set back to 2 after OURs working again with batcher
            start_config["container"]["cont_batchMode"] = 0;
        } if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
            start_config["container"]["cont_batchMode"] = 1;
        } if (ctrl_systemName == "ppp" || ctrl_systemName == "jlf") {
            start_config["container"]["cont_dropMode"] = 1;
        }
        start_config["container"]["cont_pipelineSLO"] = container->task->tk_slo;
        start_config["container"]["cont_timeBudgetLeft"] = container->timeBudgetLeft;
        start_config["container"]["cont_startTime"] = container->startTime;
        start_config["container"]["cont_endTime"] = container->endTime;
        start_config["container"]["cont_localDutyCycle"] = container->localDutyCycle;
        start_config["container"]["cont_cycleStartTime"] = std::chrono::duration_cast<TimePrecisionType>(container->cycleStartTime.time_since_epoch()).count();

        if (container->model != DataSource) {
            std::vector<uint32_t> modelProfile;
            for (auto &[batchSize, profile]: container->pipelineModel->processProfiles.at(SystemDeviceTypeList[container->device_agent->type]).batchInfer) {
                modelProfile.push_back(batchSize);
                modelProfile.push_back(profile.p95prepLat);
                modelProfile.push_back(profile.p95inferLat);
                modelProfile.push_back(profile.p95postLat);
            }

            if (modelProfile.empty()) {
                spdlog::get("container_agent")->warn("Model profile not found for container: {0:s}", container->name);
            }
            start_config["container"]["cont_modelProfile"] = modelProfile;
            if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
                ctrl_fcpo_server->incrementClientCounter();
            }
        }

        json base_config = start_config["container"]["cont_pipeline"];

        // adjust pipeline configs
        for (auto &j: base_config) {
            j["msvc_idealBatchSize"] = container->batch_size;
            j["msvc_pipelineSLO"] = container->pipelineSLO;
        }
        if (model == ModelType::DataSource) {
            base_config[0]["msvc_dataShape"] = {container->dimensions};
            if (container->pipelineModel->datasourceName[0].find("spot") != std::string::npos) {
                base_config[0]["msvc_idealBatchSize"] = 7; // spot data only available in 7 fps
            } else {
                base_config[0]["msvc_idealBatchSize"] = ctrl_systemFPS;
            }
        } else {
            if (model == ModelType::Yolov5nDsrc || model == ModelType::RetinaMtfaceDsrc) {
                base_config[0]["msvc_dataShape"] = {container->dimensions};
                base_config[0]["msvc_type"] = 500;
                base_config[0]["msvc_idealBatchSize"] = ctrl_systemFPS;
            }
            base_config[1]["msvc_dnstreamMicroservices"][0]["nb_expectedShape"] = {container->dimensions};
            base_config[3]["path"] = container->model_file;
        }

        // adjust receiver upstreams
        base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"] = {};
        if (container->model == DataSource || container->model == Yolov5nDsrc || container->model == RetinaMtfaceDsrc) {
            base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = "video_source";
            for (auto &source: container->pipelineModel->datasourceName) {
                base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"].push_back(source);
            }
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
            if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
                start_config["fcpo"] = ctrl_fcpo_server->getConfig();
                std::string deviceTypeName = getDeviceTypeName(container->device_agent->type);
                start_config["fcpo"]["timeout_size"] = (deviceTypeName == "server") ? 3 : 2;
                start_config["fcpo"]["batch_size"] = container->pipelineModel->processProfiles[deviceTypeName].maxBatchSize;
                start_config["fcpo"]["threads_size"] = (deviceTypeName == "server") ? 4 : 2;
            }
        }

        start_config["container"]["cont_pipeline"] = base_config;
        control_port = container->recv_port - 5000;
    }
    container->fcpo_conf = start_config["fcpo"];

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
    request.set_model_type(container->model);
    for (auto &dim: container->dimensions) {
        request.add_input_shape(dim);
    }

    sendMessageToDevice(container->device_agent->name, MSG_TYPE[CONTAINER_START], request.SerializeAsString());
    spdlog::get("container_agent")->info("Requested container {0:s} to start!", container->name);
}

void Controller::MoveContainer(ContainerHandle *container, NodeHandle *device) {
    NodeHandle *old_device = container->device_agent;
    bool start_dsrc = false, merge_dsrc = false;
    if (device->name != "server") {
        if (container->mergable) {
            merge_dsrc = true;
            if (container->model == Yolov5n) {
                container->model = Yolov5nDsrc;
            } else if (container->model == RetinaMtface) {
                container->model = RetinaMtfaceDsrc;
            }
        }
    } else {
        if (container->mergable) {
            start_dsrc = true;
            if (container->model == Yolov5nDsrc) {
                container->model = Yolov5n;
            } else if (container->model == RetinaMtfaceDsrc) {
                container->model = RetinaMtface;
            }
        }
    }
    std::string old_link = absl::StrFormat("%s:%d", container->device_agent->ip, container->recv_port);
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
            AdjustUpstream(container->recv_port, upstr, device, container->pipelineModel->name, AdjustUpstreamMode::Overwrite, old_link);
        }
    }
    StopContainer(container, old_device);
    spdlog::get("container_agent")->info("Container {0:s} moved to device {1:s}", container->name, device->name);
    old_device->containers.erase(container->name);
}

void Controller::AdjustUpstream(int port, ContainerHandle *upstr, NodeHandle *new_device,
                                const std::string &dwnstr, AdjustUpstreamMode mode, const std::string &old_link) {
    ContainerLink request;
    request.set_mode(mode);
    request.set_name(upstr->name);
    request.set_downstream_name(dwnstr);
    request.set_ip(new_device->ip);
    request.set_port(port);
    request.set_data_portion(1.0);
    request.set_old_link(old_link);
    request.set_offloading_duration(0);

    sendMessageToDevice(upstr->device_agent->name, MSG_TYPE[ADJUST_UPSTREAM], request.SerializeAsString());
    spdlog::get("container_agent")->info("Upstream of {0:s} adjusted to container {1:s}", dwnstr, upstr->name);
}

void Controller::SyncDatasource(ContainerHandle *prev, ContainerHandle *curr) {
    ContainerLink request;
    request.set_name(prev->name);
    request.set_downstream_name(curr->name);

    sendMessageToDevice(curr->device_agent->name, MSG_TYPE[SYNC_DATASOURCES], request.SerializeAsString());
    spdlog::get("container_agent")->info("Datasource {0:s} synced with {1:s}", prev->name, curr->name);
}

void Controller::AdjustBatchSize(ContainerHandle *msvc, int new_bs) {
    msvc->batch_size = new_bs;
    ContainerInts request;
    request.set_name(msvc->name);
    request.add_value(new_bs);

    sendMessageToDevice(msvc->device_agent->name, MSG_TYPE[BATCH_SIZE_UPDATE], request.SerializeAsString());
    spdlog::get("container_agent")->info("Batch size of {0:s} adjusted to {1:d}", msvc->name, new_bs);
}

void Controller::AdjustCudaDevice(ContainerHandle *msvc, GPUHandle *new_device) {
    msvc->gpuHandle = new_device;
    // TODO: also adjust actual running container
}

void Controller::AdjustResolution(ContainerHandle *msvc, std::vector<int> new_resolution) {
    msvc->dimensions = new_resolution;
    ContainerInts request;
    request.set_name(msvc->name);
    request.add_value(new_resolution[0]);
    request.add_value(new_resolution[1]);
    request.add_value(new_resolution[2]);

    sendMessageToDevice(msvc->device_agent->name, MSG_TYPE[RESOLUTION_UPDATE], request.SerializeAsString());
    spdlog::get("container_agent")->info("Resolution of {0:s} adjusted to {1:d}x{2:d}x{3:d}",
                                      msvc->name, new_resolution[0], new_resolution[1], new_resolution[2]);
}

void Controller::StopContainer(ContainerHandle *container, NodeHandle *device, bool forced) {
    spdlog::get("container_agent")->info("Stopping container: {0:s}", container->name);
    ContainerSignal request;
    request.set_name(container->name);
    request.set_forced(forced);
    sendMessageToDevice(device->name, MSG_TYPE[CONTAINER_STOP], request.SerializeAsString());

    if (container->gpuHandle != nullptr)
        container->gpuHandle->removeContainer(container);
    if ((ctrl_systemName == "fcpo" || ctrl_systemName== "apis") && container->model != DataSource && container->model != Sink) {
        ctrl_fcpo_server->decrementClientCounter();
    }
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

    // Preprocessor to Inference
    // Utilization of inference
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

// ============================================================ Communication Handlers ============================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //

void Controller::HandleControlMessages() {
    while (running) {
        message_t message;
        if (server_socket.recv(message, recv_flags::none)) {
            std::string raw = message.to_string();
            std::istringstream iss(raw);
            std::string topic;
            iss >> topic;
            iss.get(); // skip the space after the topic
            std::string payload((std::istreambuf_iterator<char>(iss)),
                                std::istreambuf_iterator<char>());
            if (system_handlers.count(topic)) {
                system_handlers[topic](payload);
            } else {
                spdlog::get("container_agent")->error("Received unknown topic: {}", topic);
            }
//        } else {
//            spdlog::get("container_agent")->trace("Communication Receive Timeout");
        }
    }
}

void Controller::handleDeviseAdvertisement(const std::string& msg) {
    ConnectionConfigs request;
    SystemInfo reply;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed to connect device with msg: {}", msg);
        return;
    }
    std::string deviceName = request.device_name();
    NodeHandle *node = new NodeHandle{deviceName, request.ip_address(), static_cast<SystemDeviceType>(request.device_type()),
                                      DATA_BASE_PORT + ctrl_port_offset + request.agent_port_offset(), {}};
    reply.set_name(ctrl_systemName);
    reply.set_experiment(ctrl_experimentName);
    server_socket.send(message_t(reply.SerializeAsString()), send_flags::dontwait);
    initialiseGPU(node, request.processors(), std::vector<int>(request.memory().begin(), request.memory().end()));
    devices.addDevice(deviceName, node);
    spdlog::get("container_agent")->info("Device {} is connected to the system", request.device_name());
    queryInDeviceNetworkEntries(devices.getDevice(deviceName));

    if (node->type != SystemDeviceType::Server) {
        std::thread networkCheck(&Controller::initNetworkCheck, this, std::ref(*(devices.getDevice(deviceName))), 1000, 300000, 30);
        networkCheck.detach();
    } else {
        node->initialNetworkCheck = true;
    }
}

void Controller::handleDummyDataRequest(const std::string& msg) {
    DummyMessage request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed handle dummy data: {}", msg);
        return;
    }
    ClockType now = std::chrono::system_clock::now();
    unsigned long diff = std::chrono::duration_cast<TimePrecisionType>(
            now - std::chrono::time_point<std::chrono::system_clock>(TimePrecisionType(request.gen_time()))).count();
    unsigned int size = request.data().size();
    network_check_buffer[request.origin_name()].push_back({size, diff});
    server_socket.send(message_t("success"), send_flags::dontwait);
}

void Controller::handleForwardFLRequest(const std::string& msg) {
    FlData request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed adding FCPO client with msg: {}", msg);
        return;
    }
    for (auto &dev: devices.getMap()) {
        if (dev.first == request.device_name()) {
            if (ctrl_fcpo_server->addClient(request)) {
                spdlog::get("container_agent")->info("Successfully added client {} to FCPO Aggregation.", request.device_name());
                server_socket.send(message_t("success"), send_flags::dontwait);
            } else {
                spdlog::get("container_agent")->error("Failed adding client {} to FCPO Aggregation.", request.device_name());
                server_socket.send(message_t("error"), send_flags::dontwait);
            }
            break;
        }
    }
}

void Controller::handleSinkMetrics(const std::string& msg) {
    SinkMetrics request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse sink metrics with msg: {}", msg);
        return;
    }
    TaskHandle *task = ctrl_scheduledPipelines.getTask(request.name());
    task->tk_lastLatency = request.avg_latency();
    task->tk_lastThroughput = request.throughput();
    server_socket.send(message_t("success"), send_flags::dontwait);
}

void Controller::sendMessageToDevice(const std::string &topik, const std::string &type, const std::string &content) {
    std::string msg = absl::StrFormat("%s| %s %s", topik, type, content);
    message_t zmq_msg(msg.size());
    memcpy(zmq_msg.data(), msg.data(), msg.size());
    message_queue.send(zmq_msg, send_flags::none);
}

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
    request.set_min(minPacketSize);
    request.set_max(maxPacketSize);
    request.set_repetitions(numLoops);
    try {
        sendMessageToDevice(node.name, MSG_TYPE[NETWORK_CHECK], request.SerializeAsString());
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
    // find the closest latency between min and max packet size
    float latency = 0.0f;
    for (auto &entry: entries) {
        if (entry.first >= (minPacketSize + maxPacketSize) / 2) {
            latency = entry.second;
            break;
        }
    }
    std::lock_guard lock(node.nodeHandleMutex);
    node.initialNetworkCheck = true;
    if (entries.empty()) entries = {std::pair<uint32_t, uint64_t>{1, 1}};
    node.latestNetworkEntries["server"] = entries;
    node.lastNetworkCheckTime = std::chrono::system_clock::now();
    if (node.transmissionLatencyHistory.size() > ctrl_bandwidth_predictor.getWindowSize()) {
        node.transmissionLatencyHistory.erase(node.transmissionLatencyHistory.begin());
    }
    node.transmissionLatencyHistory.push_back(latency / 1000.0f); // convert to ms
    node.transmissionLatencyPrediction = ctrl_bandwidth_predictor.predict(node.transmissionLatencyHistory);
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

        
        for (const auto& [deviceName, nodeHandle] : devices.getMap()) {
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