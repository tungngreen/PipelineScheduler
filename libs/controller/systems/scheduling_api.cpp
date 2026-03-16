#include "scheduling_api.h"
#include "controller.h"
#include "misc.h"

// ==================================================================Scheduling==================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

void Controller::queryingProfiles(TaskHandle *task) {
    if (!task) return;

    // FIX: Update map to hold shared_ptrs
    std::map<std::string, std::shared_ptr<NodeHandle>> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;

    //if task->tk_name ends with number, remove it
    std::string sanitizedTaskName = task->tk_name;
    if (!task->tk_name.empty() && task->tk_name.back() >= '0' && task->tk_name.back() <= '9') {
        sanitizedTaskName = task->tk_name.substr(0, task->tk_name.size() - 1);
    }
    std::string source = task->tk_source.substr(task->tk_source.find_last_of('/') + 1);

    for (auto &model: *pipelineModels) {
        if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos) {
            continue;
        }
        
        // Ensure the device exists before accessing to prevent out-of-bounds crash
        if (deviceList.find(model->device) == deviceList.end()) {
            spdlog::get("container_agent")->error("Device {} not found in deviceList", model->device);
            continue;
        }
        model->deviceTypeName = getDeviceTypeName(deviceList.at(model->device)->type);
        
        // Safely lock the upstream weak_ptr to access its possibleDevices
        std::vector<std::string> upstreamPossibleDeviceList = model->possibleDevices;
        if (!model->upstreams.empty()) {
            if (auto upModel = model->upstreams.front().targetNode.lock()) {
                upstreamPossibleDeviceList = upModel->possibleDevices;
            }
        }
        
        std::vector<std::string> thisPossibleDeviceList = model->possibleDevices;
        std::vector<std::pair<std::string, std::string>> possibleDevicePairList;
        for (const auto &deviceName : upstreamPossibleDeviceList) {
            for (const auto &deviceName2 : thisPossibleDeviceList) {
                if ((deviceName == "server" && deviceName2 != deviceName) ||
                    (std::find(model->possibleDevices.begin(), model->possibleDevices.end(), "server") == model->possibleDevices.end() &&
                     deviceName.find("virt") != std::string::npos && deviceName2 != deviceName)) {
                    continue;
                }
                possibleDevicePairList.push_back({deviceName, deviceName2});
            }
        }
        std::string containerName = model->name + "_" + model->deviceTypeName;
        
        if (!task->tk_newlyAdded) {
            if (ctrl_containerLib.find(containerName) == ctrl_containerLib.end()) {
                spdlog::get("container_agent")->error("Container type {} not found in library during profile query.", containerName);
            } else {
                auto rateAndCoeffVar = queryArrivalRateAndCoeffVar(
                        *ctrl_metricsServerConn,
                        ctrl_experimentName,
                        ctrl_systemName,
                        sanitizedTaskName,
                        source,
                        ctrl_containerLib[containerName].taskName,
                        ctrl_containerLib[containerName].modelName,
                        // TODO: Change back once we have profilings in every fps
                        //ctrl_systemFPS
                        15
                );
                model->arrivalProfiles.arrivalRates = rateAndCoeffVar.first;
                model->arrivalProfiles.coeffVar = rateAndCoeffVar.second;
            }
        }

        for (const auto &pair : possibleDevicePairList) {
            // Safely ensure both devices in the pair actually exist
            if (deviceList.find(pair.first) == deviceList.end() || deviceList.find(pair.second) == deviceList.end()) continue;
            
            std::string senderDeviceType = getDeviceTypeName(deviceList.at(pair.first)->type);
            std::string receiverDeviceType = getDeviceTypeName(deviceList.at(pair.second)->type);
            containerName = model->name + "_" + receiverDeviceType;
            if (receiverDeviceType == "virtual") {
                containerName = model->name + "_server";
            }
            
            // Use shared_ptr to lock the specific device, preventing dangling pointer segfaults
            auto devFirst = devices.getDevice(pair.first);
            if (!devFirst) continue;
            
            std::unique_lock lock(devFirst->nodeHandleMutex);

            NetworkEntryType entry;
            if (receiverDeviceType == "virtual")
                entry = devFirst->latestNetworkEntries["server"];
            else
                entry = devFirst->latestNetworkEntries[receiverDeviceType];
            lock.unlock();

            if (ctrl_containerLib.find(containerName) == ctrl_containerLib.end()) {
                spdlog::get("container_agent")->error("Container type {} not found in library during D2D network query.", containerName);
                continue;
            }
            
            NetworkProfile test = queryNetworkProfile(
                    *ctrl_metricsServerConn,
                    ctrl_experimentName,
                    ctrl_systemName,
                    sanitizedTaskName,
                    source,
                    ctrl_containerLib[containerName].taskName,
                    ctrl_containerLib[containerName].modelName,
                    pair.first,
                    senderDeviceType,
                    pair.second,
                    receiverDeviceType,
                    entry,
                    // TODO: Change back once we have profilings in every fps
                    //ctrl_systemFPS
                    15
            );
            model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
        }

        for (auto &deviceName : model->possibleDevices) {
            // Safely ensure device exists
            if (deviceList.find(deviceName) == deviceList.end()) continue;
            
            std::string deviceTypeName = getDeviceTypeName(deviceList.at(deviceName)->type);
            containerName = model->name + "_" + deviceTypeName;
            if (deviceTypeName == "virtual") {
                containerName = model->name + "_server";
            }

            if (ctrl_containerLib.find(containerName) == ctrl_containerLib.end()) {
                spdlog::get("container_agent")->error("Container type {} not found in library during Model Profile query.", containerName);
                continue;
            }
            
            ModelProfile profile = queryModelProfile(
                    *ctrl_metricsServerConn,
                    ctrl_experimentName,
                    ctrl_systemName,
                    task->tk_name,
                    source,
                    deviceName,
                    deviceTypeName,
                    ctrl_containerLib[containerName].modelName,
                    // TODO: Change back once we have profilings in every fps
                    //ctrl_systemFPS
                    15
            );
            model->processProfiles[deviceTypeName] = profile;
            
            // Prevent segfault when std::max_element tries to dereference .end() on an empty SQL result
            if (!profile.batchInfer.empty()) {
                model->processProfiles[deviceTypeName].maxBatchSize = std::max_element(
                        profile.batchInfer.begin(),
                        profile.batchInfer.end(),
                        [](const auto &p1, const auto &p2) {
                            return p1.first < p2.first;
                        }
                )->first;
            } else {
                model->processProfiles[deviceTypeName].maxBatchSize = 1; // Safe fallback
            }
        }
    }
}

void Controller::Scheduling() {
    // Map network messages to handler functions
    api_handlers = {
        {MSG_TYPE[START_TASK], std::bind(&Controller::HandleStartTask, this, std::placeholders::_1)},
        {MSG_TYPE[STOP_TASK], std::bind(&Controller::StopSingleTask, this, std::placeholders::_1)}
    };

    while (running) {
        Stopwatch schedulingSW;
        schedulingSW.start();
        
        auto now = std::chrono::system_clock::now();
        ctrl_controlTimings.currSchedulingTime = now;
        
        /**
         * @brief Block 1: Calculate the timeout for the ZeroMQ socket based on the NEXT scheduled task
         * We default to a 1-second maximum wait if no tasks are scheduled yet
         * 
         */
        int timeout_ms = 1000; 
        auto next_wake_time = now + std::chrono::seconds(1);
        
        // Iterate through the nextSchedulingtime map to find the earliest scheduled task time
        for (const auto& [tName, tTime] : ctrl_controlTimings.nextSchedulingtime) {
            if (tTime < next_wake_time) {
                next_wake_time = tTime;
            }
        }
        
        // Calculate the timeout in milliseconds until the next wake time
        timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(next_wake_time - now).count();
        if (timeout_ms < 0) timeout_ms = 0; // Prevent negative timeouts if we are lagging

        // Set the socket timeout. It will block until a message arrives OR the timeout is hit
        api_socket.set(zmq::sockopt::rcvtimeo, timeout_ms);

        /****************************************************************************************************************************************/

        /**
         * @brief Block 2: Event-Driven Execution
         * Listen for incoming messages and handle them immediately. This allows us to react to new tasks or updates without waiting for the
         * periodic timer. If a message is received, we set triggered_by_event to true, which will cause the scheduling logic to run immediately 
         * after this block, even if the periodic timer hasn't expired yet.
         * 
         */
        bool triggered_by_event = false;
        message_t message;

        try {
            if (api_socket.recv(message, recv_flags::none)) {
                std::string raw = message.to_string();
                std::istringstream iss(raw);
                std::string topic;
                iss >> topic;
                iss.get(); // skip the space after the topic
                std::string payload((std::istreambuf_iterator<char>(iss)),
                                    std::istreambuf_iterator<char>());
                if (api_handlers.count(topic)) {
                    // This will call HandleStartTask, which handles the msg and calls ScheduleSingleTask
                    api_handlers[topic](payload);
                    triggered_by_event = true;
                } else {
                    spdlog::get("container_agent")->error("Received unknown topic: {}", topic);
                }
            }
        } catch (const zmq::error_t& e) {
            // Timeout reached, or socket interrupted. Proceed to periodic checks.
        }
        /****************************************************************************************************************************************/

        /**
         * @brief Block 3: Periodic Execution
         * Scheduling all pipelines periodically to account for changes in the workloads
         * 
         */
        now = std::chrono::system_clock::now();
        bool periodic_update_needed = false;
        
        if (isPipelineInitialised) {
            auto taskList = ctrl_savedUnscheduledPipelines.getMap();
            for (auto &[taskName, taskHandle] : taskList) {
                // If the task's individual timer is up, schedule it
                if (now >= ctrl_controlTimings.nextSchedulingtime[taskName]) {
                    ScheduleSingleTask(taskHandle);
                    
                    // Reset the timer for this specific task
                    ctrl_controlTimings.nextSchedulingtime[taskName] = now + std::chrono::seconds(ctrl_controlTimings.schedulingIntervalSec);
                    periodic_update_needed = true;
                }
            }
        }
        /****************************************************************************************************************************************/

        /**
         * @brief Block 4: Global Merge & Apply\
         * After handling any event-driven updates and checking for periodic scheduling needs, we perform a global merge of pipelines and 
         * apply the new schedule.
         * 
         */
        if (triggered_by_event || periodic_update_needed) {
            
            mergePipelines();

            auto mergedTasks = ctrl_mergedPipelines.getMap();
            for (auto &[taskName, taskHandle]: mergedTasks) {
                for (auto &model: taskHandle->tk_pipelineModels) {
                    for (auto i = 0; i < model->numReplicas; i++) {
                        model->cudaDevices.push_back(0); // Add dummy cuda device value to create a container manifestation
                    }
                    if (model->name.find("sink") != std::string::npos) {
                        model->device = "sink";
                    }
                }
            }

            estimatePipelineTiming();
            ctrl_scheduledPipelines = ctrl_mergedPipelines;
            ApplyScheduling();
        }

        schedulingSW.stop();
        
        // Hard runtime limit check
        if (startTime == std::chrono::system_clock::time_point()) startTime = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(std::chrono::system_clock::now() - startTime).count() > ctrl_runtime) {
            running = false;
            break;
        }
    }
    delete this;
}

/**
 * @brief Handles the network payload, creates/updates the task, and passes it to the mathematical scheduler.
 * * @param msg The raw protobuf payload
 */
void Controller::HandleStartTask(const std::string &msg) {
    TaskDesc request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed handle task request: {}", msg);
        return;
    }

    std::shared_ptr<TaskHandle> task;
    std::string taskName = request.name();
    
    if (ctrl_savedUnscheduledPipelines.hasTask(taskName)) {
        task = ctrl_savedUnscheduledPipelines.getTask(taskName);
    }
    *task = *CreatePipelineFromMessage(request);

    // Initial mathematical scheduling
    ScheduleSingleTask(task);

    // Set the individual timer for future periodic updates (e.g., 1 minute from now)
    ctrl_controlTimings.nextSchedulingtime[taskName] = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_controlTimings.schedulingIntervalSec);

    api_socket.send(message_t("success"), send_flags::dontwait);
}

/**
 * @brief Performs the mathematical operations (profiling, CWD, edge shifting) on a single pipeline.
 * * @param task The task to schedule
 */
void Controller::ScheduleSingleTask(std::shared_ptr<TaskHandle> task) {
    if (!task || task->tk_pipelineModels.empty()) return;
    
    // Pass raw pointer via .get() to the underlying math functions
    queryingProfiles(task.get());
    crossDeviceWorkloadDistributor(task.get(), task->tk_slo / 2);
    
    //Find the datasource model in the pipeline to use as the starting point for edge shifting
    std::shared_ptr<PipelineModel> datasource = nullptr;
    for (auto &model: task->tk_pipelineModels) {
        if (model->name.find("datasource") != std::string::npos) {
            datasource = model;
            break;
        }
    }

    shiftModelToEdge(task->tk_pipelineModels, datasource.get(),
                     task->tk_slo, datasource->device);
                     
    for (auto &model: task->tk_pipelineModels) {
        if (model->name.find("_") == std::string::npos) {
            model->name = task->tk_name + "_" + model->name;
        }
    }
    
    estimateModelTiming(datasource.get(), 0);
    task->tk_newlyAdded = false;
}

void Controller::StopSingleTask(const std::string &msg) {
    // TaskCommand request;
    // if (!request.ParseFromString(msg)) {
    //     spdlog::get("container_agent")->error("Failed handle task request: {}", msg);
    //     return;
    // }
    // TaskHandle *task = nullptr;
    // std::string taskName = request.task_name();
    // if (ctrl_scheduledPipelines.hasTask(taskName)) task = ctrl_scheduledPipelines.getTask(taskName);

    // if (task == nullptr) {
    //     api_socket.send(message_t("error: task not found"), send_flags::dontwait);
    //     return;
    // } else if (task->tk_src_device != request.task_srcdevice()) {
    //     api_socket.send(message_t("error: src doesnt match"), send_flags::dontwait);
    //     return;
    // }

    // for (auto &model: task->tk_pipelineModels)
    //     for (auto &cont: model->manifestations)
    //         StopContainer(cont, cont->device_agent, true);

    // ctrl_scheduledPipelines.removeTask(taskName);
    // server_socket.send(message_t("success"), send_flags::dontwait);
}

void Controller::ScaleUp(PipelineModel *model, uint8_t numIncReps) {}

void Controller::ScaleDown(PipelineModel *model, uint8_t numDecReps) {}

void Controller::Rescaling() {}

/**
 * @brief colocationTemporalScheduler (CORAL) for container instances
 *
 * @param container
 * @return true
 * @return false
 */
bool Controller::containerColocationTemporalScheduling(std::shared_ptr<ContainerHandle> container) {
    if (!container) return false;

    auto agent = container->device_agent.lock();
    if (!agent) {
        spdlog::get("container_agent")->error("Device agent is null for container {0:s}", container->name);
        return false;
    }

    std::string deviceName = agent->name;
    
    auto targetDevice = devices.getDevice(deviceName);
    if (!targetDevice) {
        spdlog::get("container_agent")->error("Target device {0:s} not found in registry", deviceName);
        return false;
    }

    // Pass the securely locked shared_ptr to the memory reservation algorithms
    auto portion = findFreePortionForInsertion(targetDevice->freeGPUPortions, container.get());

    if (portion == nullptr) {
        spdlog::get("container_agent")->error("No free portion found for container {0:s}", container->name);
        return false;
    }
    
    container->executionPortion = portion;
    container->gpuHandle = portion->lane->gpuHandle;
    container->gpuHandle->addContainer(container);
    insertUsedGPUPortion(targetDevice->freeGPUPortions, container, portion);
    return true;
}

/**
 * @brief colocationTemporalScheduler (CORAL) for models
 *
 * @param pipelineModel
 * @param replica_id
 * @return true
 * @return false
 */
bool Controller::modelColocationTemporalScheduling(PipelineModel *pipelineModel, int replica_id) {
    if (!pipelineModel) return false; // Safety guard
    
    if (pipelineModel->gpuScheduled) { return true; }
    
    if (pipelineModel->name.find("datasource") == std::string::npos &&
        pipelineModel->name.find("sink") == std::string::npos) {
        
        // FIX: Safely lock the task weak_ptr to access physical containers
        if (auto task = pipelineModel->task.lock()) {
            for (auto &contWk : task->tk_subTasks[pipelineModel->name]) {
                // FIX: Lock the container weak_ptr
                if (auto container = contWk.lock()) {
                    if (container->replica_id == replica_id) {
                        container->startTime = pipelineModel->startTime;
                        container->endTime = pipelineModel->endTime;
                        
                        // Because containerColocationTemporalScheduling now accepts a shared_ptr, we pass it safely!
                        containerColocationTemporalScheduling(container);
                    }
                }
            }
        }
    }
    
    bool allScheduled = true;
    
    for (auto &downstreamWk : pipelineModel->downstreams) {
        if (auto downstream = downstreamWk.targetNode.lock()) {
            // Because this is a synchronous traversal, passing the raw pointer downstream (.get()) is ok
            if (!modelColocationTemporalScheduling(downstream.get(), replica_id)) allScheduled = false;
        }
    }
    
    if (!allScheduled) return false;
    
    // If numReplicas is 0, replica_id (0) + 1 >= 0 evaluates
    // Using addition instead of subtraction completely neutralizes the unsigned integer underflow risk
    if (replica_id + 1 >= pipelineModel->numReplicas) {
        pipelineModel->gpuScheduled = true;
        return true;
    }
    return false;
}

/**
 * @brief colocationTemporalScheduler (CORAL)
 *
 */
void Controller::colocationTemporalScheduling() {
    // FIX: devices.getMap() returns shared_ptrs. 
    auto deviceList = devices.getMap();
    for (auto &[deviceName, deviceHandle]: deviceList) {
        if (deviceHandle) {
            // Safely dereferences the shared_ptr to pass the NodeHandle reference
            initiateGPULanes(*deviceHandle); 
        }
    }
    
    bool process_flag = true;
    int replica_id = 0;
    
    while (process_flag) {
        process_flag = false;
        for (auto &[taskName, taskHandle]: ctrl_scheduledPipelines.getMap()) {
            
            // Guard against empty pipeline vectors to prevent segfaults
            if (!taskHandle || taskHandle->tk_pipelineModels.empty()) continue; 
            
            for (auto &model: taskHandle->tk_pipelineModels) {
                if ((model->name.find("datasource") != std::string::npos || model->name.find("dsrc") != std::string::npos) \
                    && !model->gpuScheduled) {
                        process_flag = process_flag || !modelColocationTemporalScheduling(model.get(), replica_id);
                }
            }
        }
        replica_id++;
    }
}

bool Controller::mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile, const std::string& device, const std::string& upstreamDevice) {
    // Calculate coefficients BEFORE modifying mergedProfile.arrivalRates to prevent mathematical skew
    float totalRate = mergedProfile.arrivalRates + toBeMergedProfile.arrivalRates;
    float coefficient1 = 0.5f;
    float coefficient2 = 0.5f;
    
    // Prevent Division by Zero if both arrival rates are 0
    if (totalRate > 0.0001f) {
        coefficient1 = mergedProfile.arrivalRates / totalRate;
        coefficient2 = toBeMergedProfile.arrivalRates / totalRate;
    }
    
    // Now it is safe to modify the arrival rate
    mergedProfile.arrivalRates += toBeMergedProfile.arrivalRates;
    
    auto mergedD2DProfile = &mergedProfile.d2dNetworkProfile;
    auto toBeMergedD2DProfile = &toBeMergedProfile.d2dNetworkProfile;

    // There should be only 1 pair in the d2dNetworkProfile with key {"merged-...", device}
    D2DNetworkProfile newProfile = {};
    for (const auto &[pair1, profile2] : mergedProfile.d2dNetworkProfile) {
        for (const auto &[pair2, profile1] : toBeMergedProfile.d2dNetworkProfile) {
            if (pair2.first != upstreamDevice || pair2.second != device || pair2.second != pair1.second) {
                continue;
            }
            std::pair<std::string, std::string> newPair = {pair1.first + "_" + upstreamDevice, pair1.second};
            newProfile.insert({newPair, {}});
            newProfile[newPair].p95TransferDuration =
                    mergedD2DProfile->at(pair1).p95TransferDuration * coefficient1 +
                    toBeMergedD2DProfile->at(pair2).p95TransferDuration * coefficient2;
            newProfile[newPair].p95PackageSize =
                    mergedD2DProfile->at(pair1).p95PackageSize * coefficient1 +
                    toBeMergedD2DProfile->at(pair2).p95PackageSize * coefficient2;
        }
    }
    mergedProfile.d2dNetworkProfile = newProfile;
    return true;
}

bool Controller::mergeProcessProfiles(
        PerDeviceModelProfileType &mergedProfile,
        float arrivalRate1,
        const PerDeviceModelProfileType &toBeMergedProfile,
        float arrivalRate2,
        const std::string& device
) {
    // Calculate coefficients BEFORE modifying mergedProfile.arrivalRates to prevent mathematical skew
    float totalRate = arrivalRate1 + arrivalRate2;
    float coefficient1 = 0.5f;
    float coefficient2 = 0.5f;

    // Prevent Division by Zero if both arrival rates are 0
    if (totalRate > 0.0001f) {
        coefficient1 = arrivalRate1 / totalRate;
        coefficient2 = arrivalRate2 / totalRate;
    }
    
    for (const auto &[deviceName, profile] : toBeMergedProfile) {
        if (deviceName != device) {
            continue;
        }
        auto mergedProfileDevice = &mergedProfile[deviceName];
        auto toBeMergedProfileDevice = &toBeMergedProfile.at(deviceName);

        mergedProfileDevice->p95InputSize = std::max(mergedProfileDevice->p95InputSize, toBeMergedProfileDevice->p95InputSize);
        mergedProfileDevice->p95OutputSize = std::max(mergedProfileDevice->p95OutputSize, toBeMergedProfileDevice->p95OutputSize);

        auto mergedBatchInfer = &mergedProfileDevice->batchInfer;

        for (const auto &[batchSize, p] : toBeMergedProfileDevice->batchInfer) {
            if (mergedBatchInfer->find(batchSize) == mergedBatchInfer->end()) {
                (*mergedBatchInfer)[batchSize] = p;
                continue;
            }
            
            // Use reference to avoid repetitive map lookups and safely merge data
            auto& m_p = (*mergedBatchInfer)[batchSize];
            m_p.p95inferLat = m_p.p95inferLat * coefficient1 + p.p95inferLat * coefficient2;
            m_p.p95prepLat  = m_p.p95prepLat * coefficient1 + p.p95prepLat * coefficient2;
            m_p.p95postLat  = m_p.p95postLat * coefficient1 + p.p95postLat * coefficient2;
            m_p.cpuUtil     = std::max(m_p.cpuUtil, p.cpuUtil);
            m_p.gpuUtil     = std::max(m_p.gpuUtil, p.gpuUtil);
            m_p.memUsage    = std::max(m_p.memUsage, p.memUsage);
            m_p.rssMemUsage = std::max(m_p.rssMemUsage, p.rssMemUsage);
            m_p.gpuMemUsage = std::max(m_p.gpuMemUsage, p.gpuMemUsage);
        }

    }
    return true;
}

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel* toBeMergedModel, const std::string& device) {
    if (!mergedModel || !toBeMergedModel) return false;

    // If the merged model is empty, we should just copy the model to be merged
    if (mergedModel->numReplicas == 255) {
        *mergedModel = *toBeMergedModel; // Safe via deep copy operator
        
        // FIX: Safely lock the weak_ptr and updated to use .targetNode for PipelineEdge
        std::string upStreamDevice = "unknown";
        if (!toBeMergedModel->upstreams.empty()) {
            if (auto up = toBeMergedModel->upstreams.front().targetNode.lock()) {
                upStreamDevice = up->device;
            }
        }

        // Collect additions and erasures in temporary containers instead of modifying the map while traversing it.
        std::map<std::pair<std::string, std::string>, NetworkProfile> additions;
        std::vector<std::pair<std::string, std::string>> keysToErase;
        
        for (auto &[pair, profile] : mergedModel->arrivalProfiles.d2dNetworkProfile) {
            if (pair.second == device && pair.first == upStreamDevice) {
                additions[{"merged-" + pair.first, device}] = profile;
            }
            if (pair.first.find("merged") == std::string::npos) {
                keysToErase.push_back(pair);
            }
        }
        
        // Apply erasures
        for (auto &key : keysToErase) {
            mergedModel->arrivalProfiles.d2dNetworkProfile.erase(key);
        }
        // Apply additions
        for (auto &[key, profile] : additions) {
            mergedModel->arrivalProfiles.d2dNetworkProfile[key] = profile;
        }
        
        return true;
    }
    // If the devices are different, we should not merge the models
    if (mergedModel->device != toBeMergedModel->device ||
        toBeMergedModel->merged || mergedModel->device != device || toBeMergedModel->device != device) {
        return false;
    }

    // Add the data source of the model to be merged into the merged model's data source list
    mergedModel->datasourceName.push_back(toBeMergedModel->datasourceName[0]);
    if (mergedModel->name.find("datasource") != std::string::npos ||
        mergedModel->name.find("dsrc") != std::string::npos) {
        return false;
    }

    float rate1 = mergedModel->arrivalProfiles.arrivalRates;
    float rate2 = toBeMergedModel->arrivalProfiles.arrivalRates;
    
    // FIX: Safely lock the upstream weak_ptr and updated to use .targetNode
    std::string upStreamDevice = "unknown";
    if (!toBeMergedModel->upstreams.empty()) {
        if (auto up = toBeMergedModel->upstreams.front().targetNode.lock()) {
            upStreamDevice = up->device;
        }
    }

    mergeArrivalProfiles(mergedModel->arrivalProfiles, toBeMergedModel->arrivalProfiles, device, upStreamDevice);
    mergeProcessProfiles(mergedModel->processProfiles, rate1, toBeMergedModel->processProfiles, rate2, device);

    return true;
}

std::shared_ptr<TaskHandle> Controller::mergePipelines(const std::string& taskName) {
    auto unscheduledTasks = ctrl_savedUnscheduledPipelines.getMap();

    auto mergedPipeline = std::make_shared<TaskHandle>();
    bool found = false;

    /**
     * @brief Block 1: We first initialize the merged pipeline with the identity of one of the server-based pipelines (if any), 
     * while also tracking the strictest SLO and directly adding non-server edge pipelines to the merged pipeline without modification.
     * 
     * @param unscheduledTasks 
     */
    for (const auto& task : unscheduledTasks) {
        if (task.first.find(taskName) == std::string::npos) {
            continue;
        }
        if (task.second->tk_edge_node != "server") {
            ctrl_mergedPipelines.addTask(task.first, task.second);
            for (auto &model : task.second->tk_pipelineModels) {
                model->toBeRun = true;
            }
            continue;
        }
        if (!found) {
            found = true;
            *mergedPipeline = *task.second; 
            // Clear the old index-based graph to prepare for the Graph Union assembly
            mergedPipeline->tk_pipelineModels.clear(); 
        }
    }

    // Registry to hold unique Merged Model mapped by Signature (ModelType + DeviceName)
    std::unordered_map<std::string, std::shared_ptr<PipelineModel>> registry;
    
    if (!found) {
        spdlog::info("No task with type {0:s} has been added", taskName);
        return nullptr;
    }

    /****************************************************************************************************************************************/

    /* @brief Block 2: Merging models to create the merged models and populate the registry.
     * During this pass, we also mark original models as merged and not to be run,
     * effectively retiring them in favor of the new merged models that will be scheduled in the next block.
     * 
     * @param unscheduledTasks 
     */

    for (const auto& task : unscheduledTasks) {
        if (task.first.find(taskName) == std::string::npos || task.second->tk_edge_node != "server") {
            continue;
        }

        // For each model in the pipeline, we determine its signature and either create a new merged model or merge it into an existing one
        // in the registry.
        for (const auto& model : task.second->tk_pipelineModels) {
            std::string modelNameWOPrefix = splitString(model->name, "_").back(); // Extract the base model name without the task prefix
            std::string sig = modelNameWOPrefix + "_" + model->device;
            // If the signature is not in the registry, we create a new merged model for it. Otherwise, we merge the current model into 
            // the existing merged model in the registry.
            if (registry.find(sig) == registry.end()) {
                // First time seeing this model type on this device. Create the isolated Super-Node.
                auto mergedModel = std::make_shared<PipelineModel>(*model);
                mergedModel->upstreams.clear();   // Sever original edges
                mergedModel->downstreams.clear(); // Sever original edges
                mergedModel->merged = true;
                registry[sig] = mergedModel;
            } else {
                // Super-Node exists. Accumulate arrival rates and process profiles.
                mergeModels(registry[sig].get(), model.get(), model->device);
            }
            
            // Mark the original model as absorbed
            model->merged = true;
            model->toBeRun = false;
        }
    }
    /****************************************************************************************************************************************/

    /**
     * @brief Block 3: Edge Reconstruction & Stream Multiplexing
     * After we have established the Merged Models and merged their profiles, we need to reconstruct the DAG by reconnecting the Merged-Models
     * according to the original graph structure.
     * During this process, we also need to multiplex the data streams on the edges to ensure that the merged models receive all necessary data.
     * 
     */

    for (const auto& task : unscheduledTasks) {
        if (task.first.find(taskName) == std::string::npos || task.second->tk_edge_node != "server") {
            continue;
        }

        std::string streamId = task.second->tk_source; // The unique stream identifier

        for (const auto& targetU : task.second->tk_pipelineModels) {
            // The signature of the merged model that this original model maps to
            std::string targetUWithOPrefix = splitString(targetU->name, "_").back(); // Extract the base model name without the task prefix
            std::string sigU = targetUWithOPrefix + "_" + targetU->device;
            // The merged model that this original model maps to
            auto mergedU = registry[sigU];

            for (const auto& downEdge : targetU->downstreams) {
                if (auto targetV = downEdge.targetNode.lock()) {
                    if (targetV->name == targetU->name) {
                        continue; // Skip self-loop edges
                    }
                    std::string targetVWithOPrefix = splitString(targetV->name, "_").back(); // Extract the base model name without the task prefix
                    std::string sigV = targetVWithOPrefix + "_" + targetV->device;
                    auto mergedV = registry[sigV];

                    // Route Downstreams (U -> V)
                    bool edgeExists = false;
                    // Find the edge from mergedU to mergedV with the same classOfInterest and add the streamId to it.
                    for (auto& mergedDownEdge : mergedU->downstreams) {
                        if (mergedDownEdge.targetNode.lock() == mergedV && mergedDownEdge.classOfInterest == downEdge.classOfInterest) {
                            mergedDownEdge.streamNames.insert(streamId);
                            edgeExists = true;
                            break;
                        }
                    }
                    // If such an edge does not exist, we create a new edge from mergedU to mergedV with the classOfInterest and streamId.
                    if (!edgeExists) {
                        mergedU->downstreams.push_back(PipelineEdge{mergedV, downEdge.classOfInterest, {streamId}});
                    }

                    // Route Upstreams (V <- U)
                    bool upEdgeExists = false;
                    for (auto& mergedUpEdge : mergedV->upstreams) {
                        if (mergedUpEdge.targetNode.lock() == mergedU && mergedUpEdge.classOfInterest == downEdge.classOfInterest) {
                            mergedUpEdge.streamNames.insert(streamId);
                            upEdgeExists = true;
                            break;
                        }
                    }
                    if (!upEdgeExists) {
                        mergedV->upstreams.push_back(PipelineEdge{mergedU, downEdge.classOfInterest, {streamId}});
                    }
                }
            }
        }
    }

    /****************************************************************************************************************************************/

    /**
     * @brief Block 4: After the merged pipeline is constructed, we can add it to the merged pipelines registry and return it for scheduling.
     * 
     */
    
    for (auto& [sig, mergedModel] : registry) {
        mergedModel->toBeRun = true;
        
        // Normalize names for server nodes
        if (mergedModel->device == "server" || mergedModel->device == "sink") {
            auto names = splitString(mergedModel->name, "_");
            mergedModel->name = taskName + "_" + names.back();
        }
        mergedPipeline->tk_pipelineModels.push_back(mergedModel);
    }

    mergedPipeline->tk_name = taskName.substr(0, taskName.length());
    mergedPipeline->tk_src_device = mergedPipeline->tk_name;
    mergedPipeline->tk_source  = mergedPipeline->tk_name;

    return mergedPipeline;
}

void Controller::mergePipelines() {
    std::vector<std::string> toMerge = getPipelineNames();

    for (const auto &taskName : toMerge) {
        // FIX: Directly catch the shared_ptr, safely retaining the memory
        std::shared_ptr<TaskHandle> mergedPipeline = mergePipelines(taskName);
        
        if (!mergedPipeline) {
            continue;
        }
        
        // Increase the number of replicas to avoid bottlenecks
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            // only models scheduled to run on the server are merged and to be considered
            if (mergedModel->device != "server") {
                continue;
            }
            auto numIncReps = incNumReplicas(mergedModel.get());
            mergedModel->lastScaleTime = std::chrono::system_clock::now();
            mergedModel->numReplicas += numIncReps;
            estimateModelLatency(mergedModel.get());
        }
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            if (mergedModel->name.find("datasource") == std::string::npos) {
                continue;
            }
            estimatePipelineLatency(mergedModel.get(), mergedModel->expectedStart2HereLatency);
        }
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            // mergedPipeline is a shared_ptr, safely compatible with the weak_ptr 'task' field
            mergedModel->task = mergedPipeline;
        }
        ctrl_mergedPipelines.addTask(mergedPipeline->tk_name, mergedPipeline);
    }
}

/**
 * @brief Recursively traverse the model tree and try shifting models to edge devices
 *
 * @param models
 * @param slo
 */
void Controller::shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string& edgeDevice) {
    if (!currModel) return;

    /**
     * @brief Base cases
     * 
     */
    // The sink model should not be shifted to the edge device because it needs to send the data back to the server
    if (currModel->name.find("sink") != std::string::npos) {
        return;
    }
    // If the model is a datasource and its device is not the edge device, we should not shift it to the edge device 
    // because the datasource container needs to be on the same device as the data source
    if (currModel->name.find("datasource") != std::string::npos) {
        if (currModel->device != edgeDevice) {
            spdlog::get("container_agent")->warn("Edge device {0:s} is not identical to the datasource device {1:s}", edgeDevice, currModel->device);
            return;
        }
    }
    // If the model is already on the edge device, we should attempt to shift downstream models to the edge device as well
    if (currModel->device == edgeDevice) {
        for (auto &dWk: currModel->downstreams) {
            if (auto d = dWk.targetNode.lock()) {
                shiftModelToEdge(pipeline, d.get(), slo, edgeDevice);
            }
        }
        return;
    }

    // If the edge device is not in the list of possible devices, we should not consider it
    if (std::find(currModel->possibleDevices.begin(), currModel->possibleDevices.end(), edgeDevice) == currModel->possibleDevices.end()) {
        return;
    }
    /****************************************************************************************************************************************/

    auto targetDeviceSp = devices.getDevice(edgeDevice);
    if (!targetDeviceSp) return;
    std::string deviceTypeName = getDeviceTypeName(targetDeviceSp->type);
    
    // Guard against missing profiles to prevent out-of-range exceptions
    if (currModel->processProfiles.count(deviceTypeName) == 0) {
        spdlog::get("container_agent")->error("No process profile found for model {0:s} on device type {1:s}. Cannot consider shifting to this edge device.", currModel->name, deviceTypeName);
        return; 
    }

    // Get profiled input and output sizes for the current model on the edge device to help with the decision of whether to shift 
    // the model to the edge device
    uint32_t inputSize = currModel->processProfiles.at(deviceTypeName).p95InputSize;
    uint32_t outputSize = currModel->processProfiles.at(deviceTypeName).p95OutputSize;

    bool shifted = false;

    // We use a simple heuristic here: if the input size is much larger than the output size, we should try shifting the model
    // to use the saving from transmitting large input data over the network to compensate for the potential increase in latency
    // from using a less powerful edge device.
    if (inputSize * 0.6 > outputSize) {
        // This deep copy works perfectly because of our custom operator= in the PipelineModel header!
        PipelineModel oldModel = *currModel;
        
        currModel->device = edgeDevice;
        currModel->deviceTypeName = deviceTypeName;
        
        for (auto &downstreamWk : currModel->downstreams) {
            if (auto downstream = downstreamWk.targetNode.lock()) {
                estimateModelLatency(downstream.get());
            }
        }
        currModel->batchSize = 1;

        if (currModel->batchSize <= currModel->processProfiles.at(deviceTypeName).maxBatchSize) {
            estimateModelLatency(currModel);
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
            
            if (pipeline.empty()) return; // Safety guard
            uint64_t expectedE2ELatency = pipeline.back()->expectedStart2HereLatency;
            
            if (expectedE2ELatency > slo) {
                *currModel = oldModel;
                // break;
            } else {
                oldModel = *currModel;
                shifted = true;
            }
            // currModel->batchSize *= 2;
        }
        // if after shifting the model to the edge device, the pipeline still meets the SLO, we should keep it
        // However, if the pipeline does not meet the SLO, we should shift reverse the model back to the server
        if (!shifted) {
            *currModel = oldModel; // Rollback!
            estimateModelLatency(currModel);
            
            for (auto &downstreamWk : currModel->downstreams) {
                if (auto downstream = downstreamWk.targetNode.lock()) {
                    estimateModelLatency(downstream.get());
                }
            }
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
            // And if the model cannot be shifted to the edge device, its downstreams should not be shifted to the edge device as well
            // The DFS traversal will just return here
            return;
        }
    }
    if (!shifted) {
        return;
    }
    // Shift downstream models to the edge device
    for (auto &dWk: currModel->downstreams) {
        if (auto d = dWk.targetNode.lock()) {
            shiftModelToEdge(pipeline, d.get(), slo, edgeDevice);
        }
    }
}

/**
 * @brief cross-device workload distributor (CWD - seaweed)
 *
 * @param models
 * @param slo
 * @param nObjects
 * @return std::map<ModelType, int>
 */
void Controller::crossDeviceWorkloadDistributor(TaskHandle *task, uint64_t slo) {
    if (!task || task->tk_pipelineModels.empty()) return;

    PipelineModelListType *models = &(task->tk_pipelineModels);

    for (auto &m: *models) {
        m->batchSize = 1;
        m->numReplicas = 1;

        estimateModelLatency(m.get());
        if (m->name.find("datasource") == std::string::npos) {
            for (auto &d: m->possibleDevices) {
                if (d == "server") {
                    m->device = "server";
                    m->deviceTypeName = "server";
                    break;
                }
            }
        }
    }

    // Find datasource model as the root
    std::shared_ptr<PipelineModel> datasource = nullptr;
    for (auto &m: *models) {
        if (m->name.find("datasource") != std::string::npos) {
            datasource = m;
            break;
        }
    }
    if (!datasource) {
        spdlog::get("container_agent")->error("No datasource model found in the pipeline. Cannot perform cross-device workload distribution.");
        return;
    }


    // DFS-style recursively estimate the latency of a pipeline from source to sink
    // The first model should be the datasource
    estimatePipelineLatency(datasource.get(), 0);

    uint64_t expectedE2ELatency = models->back()->expectedStart2HereLatency;

    if (slo < expectedE2ELatency) {
        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
    }

    // Increase number of replicas to avoid bottlenecks
    for (auto &m: *models) {
        auto numIncReplicas = incNumReplicas(m.get());
        m->lastScaleTime = std::chrono::system_clock::now();
        m->numReplicas += numIncReplicas;
    }
    estimatePipelineLatency(datasource.get(), 0);

    // Find near-optimal batch sizes
    auto foundBest = true;
    while (foundBest) {
        foundBest = false;
        uint64_t bestCost = models->back()->estimatedStart2HereCost;
        for (auto &m: *models) {
            if (m->name.find("datasource") != std::string::npos || m->name.find("sink") != std::string::npos) {
                continue;
            }
            
            // Ensure the device type profile exists before querying to avoid map garbage data
            if (m->processProfiles.count(m->deviceTypeName) == 0) {
                spdlog::get("container_agent")->error("No process profile found for model {0:s} on device type {1:s}. Cannot consider increasing batch size.", m->name, m->deviceTypeName);
                continue; 
            }
            
            BatchSizeType oldBatchsize = m->batchSize;
            m->batchSize *= 2;
            
            if (m->batchSize > m->processProfiles.at(m->deviceTypeName).maxBatchSize) {
                m->batchSize = oldBatchsize;
                continue;
            }
            // FIX: Pass raw pointers
            estimateModelLatency(m.get());
            estimatePipelineLatency(m.get(), m->expectedStart2HereLatency);
            
            expectedE2ELatency = models->back()->expectedStart2HereLatency;
            if (expectedE2ELatency < slo) {
                // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
                uint64_t estimatedE2Ecost = models->back()->estimatedStart2HereCost;
                // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
                // 0.9 to avoid small numerical errors during profiling
                if (estimatedE2Ecost * 0.98 < bestCost) {
                    bestCost = estimatedE2Ecost;
                    foundBest = true;
                    spdlog::get("container_agent")->trace("Increasing the batch size of model {0:s} to {1:d}", m->name, m->batchSize);
                }
                if (!foundBest) {
                    m->batchSize = oldBatchsize;
                    estimateModelLatency(m.get());
                    estimatePipelineLatency(m.get(), m->expectedStart2HereLatency);
                    continue;
                }
                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
                auto numDecReplicas = decNumReplicas(m.get());
                m->lastScaleTime = std::chrono::system_clock::now();
                m->numReplicas -= numDecReplicas;
            } else {
                m->batchSize = oldBatchsize;
                estimateModelLatency(m.get());
                estimatePipelineLatency(m.get(), m->expectedStart2HereLatency);
            }
        }
    }
    return;
}

/**
 * @brief estimate the different types of latency, in microseconds
 * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
 *
 * @param model infomation about the model
 * @param modelType
 */
void Controller::estimateModelLatency(PipelineModel *currModel) {
    if (!currModel) return;
    std::string deviceTypeName = currModel->deviceTypeName;
    // We assume datasource and sink models have no latency
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos) {
        currModel->expectedQueueingLatency = 0;
        currModel->expectedAvgPerQueryLatency = 0;
        currModel->expectedMaxProcessLatency = 0;
        currModel->estimatedPerQueryCost = 0;
        currModel->expectedStart2HereLatency = 0;
        currModel->estimatedStart2HereCost = 0;
        return;
    }
    
    // Check if the profile exists to prevent silent insertion of zero-values
    if (currModel->processProfiles.count(deviceTypeName) == 0) {
        spdlog::get("container_agent")->error("Profile for device type {} not found for model {}", deviceTypeName, currModel->name);
        return;
    }
    
    ModelProfile profile = currModel->processProfiles.at(deviceTypeName);
    BatchSizeType batchSize = currModel->batchSize;
    
    if (profile.batchInfer.count(batchSize) == 0) {
        spdlog::get("container_agent")->error("Batch size {} not found in profile for model {}", batchSize, currModel->name);
        return;
    }
    
    uint64_t preprocessLatency = profile.batchInfer.at(batchSize).p95prepLat;
    uint64_t inferLatency = profile.batchInfer.at(batchSize).p95inferLat;
    uint64_t postprocessLatency = profile.batchInfer.at(batchSize).p95postLat;

    if (preprocessLatency == 0) preprocessLatency = 1;
    
    float preprocessRate = 1000000.f / preprocessLatency * currModel->numReplicas;
    while (preprocessRate * 0.8 < currModel->arrivalProfiles.arrivalRates && currModel->numReplicas < 4) {
        currModel->numReplicas++;
        spdlog::get("container_agent")->info("Increasing the number of replicas of model {0:s} to {1:d}", currModel->name, currModel->numReplicas);
        preprocessRate = 1000000.f / preprocessLatency * currModel->numReplicas;
    }

    currModel->expectedQueueingLatency = calculateQueuingLatency(currModel->arrivalProfiles.arrivalRates,
                                                                 preprocessRate);
    currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
    currModel->expectedMaxProcessLatency =
            preprocessLatency * batchSize + inferLatency * batchSize + postprocessLatency * batchSize;
    currModel->estimatedPerQueryCost = preprocessLatency + inferLatency + postprocessLatency + currModel->expectedTransferLatency;
    currModel->expectedStart2HereLatency = 0;
    currModel->estimatedStart2HereCost = 0;
}

void Controller::estimateModelNetworkLatency(PipelineModel *currModel) {
    if (!currModel) return;
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos) {
        currModel->expectedTransferLatency = 0;
        return;
    }

    // Lock the weak_ptr safely before accessing the upstream device property
    if (!currModel->upstreams.empty()) {
        if (auto upModel = currModel->upstreams[0].targetNode.lock()) {
            currModel->expectedTransferLatency = currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(currModel->device, upModel->device)].p95TransferDuration;
        }
    }
}

/**
 * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
 *
 * @param pipeline provides all information about the pipeline needed for scheduling
 * @param currModel
 */
void Controller::estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency) {
    if (!currModel) return;
    // estimateModelLatency(currModel, currModel->device);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency
    // to reach from each upstream.
    if (currModel->name.find("datasource") != std::string::npos) {
        currModel->expectedStart2HereLatency = start2HereLatency;
        currModel->estimatedStart2HereCost = 0;
    } else {
        currModel->estimatedStart2HereCost = currModel->estimatedPerQueryCost;
        currModel->expectedStart2HereLatency = 0;
        
        // Lock weak_ptrs when traversing the upstream DAG
        for (auto &upstreamWk : currModel->upstreams) {
            if (auto upstream = upstreamWk.targetNode.lock()) {
                currModel->estimatedStart2HereCost += upstream->estimatedStart2HereCost;
                currModel->expectedStart2HereLatency = std::max(
                        currModel->expectedStart2HereLatency,
                        upstream->expectedStart2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency +
                        currModel->expectedQueueingLatency
                );
            }
        }
    }

    // Safely lock weak_ptrs for downstream recursion
    for (const auto &dWk: currModel->downstreams) {
        if (auto d = dWk.targetNode.lock()) {
            estimatePipelineLatency(d.get(), currModel->expectedStart2HereLatency);
        }
    }

    if (currModel->downstreams.empty()) {
        return;
    }
}

/**
 * @brief This function traverses the DAG backward (from Sink to Datasource) using Depth First Search (DFS). 
 * It assigns strict deadlines ("time budgets") to every microservice to ensure the pipeline meets its Service Level Objective (SLO).
 * 
 * @param currModel 
 */
void Controller::estimateTimeBudgetLeft(PipelineModel *currModel) {
    if (!currModel) return;
    if (currModel->name.find("sink") != std::string::npos) {
        currModel->timeBudgetLeft = 0;
        return;
    } else if (currModel->name.find("datasource") != std::string::npos) {
        if (auto t = currModel->task.lock()) {
            currModel->timeBudgetLeft = t->tk_slo;
        } else {
            currModel->timeBudgetLeft = 0;
        }
    }

    uint64_t dnstreamBudget = 0;
    for (const auto &dWk : currModel->downstreams) {
        if (auto d = dWk.targetNode.lock()) {
            estimateTimeBudgetLeft(d.get());
            dnstreamBudget = std::max(dnstreamBudget, d->timeBudgetLeft);
        }
    }
    currModel->timeBudgetLeft = dnstreamBudget * 1.2 + (currModel->expectedQueueingLatency + currModel->expectedMaxProcessLatency) * 1.2;
}

/**
 * @brief Estimate the start time and end time of each model in the pipeline based on the expected latency of each model 
 * and the data transfer latency between models.
 * * @param currModel 
 * @param start2HereLatency 
 */
void Controller::estimateModelTiming(PipelineModel *currModel, const uint64_t start2HereLatency) {
    if (!currModel) return;

    if (currModel->name.find("datasource") != std::string::npos) {
        currModel->startTime = 0;
        currModel->endTime = 0;
        // if (currModel->name.find("sink") != std::string::npos) {

    }
    else if (currModel->name.find("sink") != std::string::npos) {
        currModel->startTime = 0;
        currModel->endTime = 0;

        for (auto &upstreamWk : currModel->upstreams) {
            if (auto upstream = upstreamWk.targetNode.lock()) {
                currModel->localDutyCycle = std::max(currModel->localDutyCycle, upstream->endTime);
            }
        }
        return;
    } else {
        // auto batchSize = currModel->batchSize;
        auto profile = currModel->processProfiles.at(currModel->deviceTypeName);

        uint64_t maxStartTime = std::max(currModel->startTime, start2HereLatency);
        
        for (auto &upstreamWk : currModel->upstreams) {
            if (auto upstream = upstreamWk.targetNode.lock()) {
                if (upstream->device != currModel->device) {
                    continue;
                }
                // TODO: Add in-device transfer latency
                maxStartTime = std::max(maxStartTime, upstream->endTime);
            }
        }
        currModel->startTime = maxStartTime;
        currModel->endTime = currModel->startTime + currModel->expectedMaxProcessLatency;
    }

    uint64_t maxDnstreamDutyCycle = currModel->localDutyCycle;
    
    for (auto &downstreamWk : currModel->downstreams) {
        if (auto downstream = downstreamWk.targetNode.lock()) {
            if (downstream->device != currModel->device &&
                downstream->name.find("sink") == std::string::npos) {
                estimateModelTiming(downstream.get(), 0);
                maxDnstreamDutyCycle = std::max(maxDnstreamDutyCycle, start2HereLatency + currModel->expectedMaxProcessLatency);
                continue;
            }
            estimateModelTiming(downstream.get(), start2HereLatency + currModel->expectedMaxProcessLatency);
            maxDnstreamDutyCycle = std::max(maxDnstreamDutyCycle, downstream->localDutyCycle);
        }
    }
    currModel->localDutyCycle = maxDnstreamDutyCycle;
}

/**
 * @brief This function traverses the DAG forward (from Datasource to Sink) to estimate the start time and end time of each model in all pipelines.
 * 
 */
void Controller::estimatePipelineTiming() {
    auto tasks = ctrl_mergedPipelines.getMap();
    for (auto &[taskName, task]: tasks) {
        for (auto &model: task->tk_pipelineModels) {
            // If the model has already been estimated, we should not estimate it again
            if (model->endTime != 0 && model->startTime != 0) {
                continue;
            }
            estimateModelTiming(model.get(), 0);
        }
        // uint64_t localDutyCycle;
        // for (auto &model: task->tk_pipelineModels) {
        //     if (model->name.find("sink") != std::string::npos) {
        //         localDutyCycle = model->localDutyCycle;
        //     }
        // }
        // for (auto &model: task->tk_pipelineModels) {
        //     model->localDutyCycle = localDutyCycle;
        // }
        for (auto &model: task->tk_pipelineModels) {
            if (model->name.find("datasource") == std::string::npos &&
                model->name.find("dsrc") == std::string::npos) {
                continue;
            }
            estimateTimeBudgetLeft(model.get());
        }
    }
}

/**
 * @brief This function traverses the DAG forward (from Datasource to Sink) to estimate the start time and end time of each model in a pipeline.
 * 
 * @param task 
 */
void Controller::estimatePipelineTiming(TaskHandle *task) {
    if (!task) return;
    for (auto &model: task->tk_pipelineModels) {
        // If the model has already been estimated, we should not estimate it again
        if (model->endTime != 0 && model->startTime != 0) {
            continue;
        }
        estimateModelTiming(model.get(), 0);
    }
    for (auto &model: task->tk_pipelineModels) {
        if (model->name.find("datasource") == std::string::npos &&
            model->name.find("dsrc") == std::string::npos) {
            continue;
        }
        estimateTimeBudgetLeft(model.get());
    }
}

/**
 * @brief Attempts to increase the number of replicas to meet the arrival rate
 *
 * @param model the model to be scaled
 * @param deviceName
 * @return uint8_t The number of replicas to be added
 */
uint8_t Controller::incNumReplicas(const PipelineModel *model) {
    if (!model) return 0; // Safety guard
    
    if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos) {
        return 0;
    }
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    
    // Guard against missing profiles throwing exceptions
    if (model->processProfiles.count(deviceTypeName) == 0 || 
        model->processProfiles.at(deviceTypeName).batchInfer.count(model->batchSize) == 0) {
        return 0;
    }
    
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    uint64_t prepLat = profile.batchInfer.at(model->batchSize).p95prepLat;
    uint64_t postLat = profile.batchInfer.at(model->batchSize).p95postLat;

    uint64_t totalLatency = inferenceLatency + prepLat + postLat;
    if (totalLatency == 0) totalLatency = 1;
    if (prepLat == 0) prepLat = 1;
    
    float indiProcessRate = 1000000.f / totalLatency;
    float indiPreprocessRate = 1000000.f / prepLat;
    
    float processRate = indiProcessRate * numReplicas;
    float preprocessRate = indiPreprocessRate * numReplicas;
    while ((processRate * 0.85 < model->arrivalProfiles.arrivalRates ||
            preprocessRate * 0.95 < model->arrivalProfiles.arrivalRates) && numReplicas < 4) {
        numReplicas++;
        spdlog::get("container_agent")->info("Increasing the number of replicas of model {0:s} to {1:d}", model->name, numReplicas);
        processRate = indiProcessRate * numReplicas;
        preprocessRate = indiPreprocessRate * numReplicas;
    }
    return numReplicas - model->numReplicas;
}

/**
 * @brief Decrease the number of replicas as long as it is possible to meet the arrival rate
 *
 * @param model
 * @return uint8_t The number of replicas to be removed
 */
uint8_t Controller::decNumReplicas(const PipelineModel *model) {
    if (!model) return 0; // Safety guard
    
    if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos) {
        return 0;
    }

    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    
    // Guard against missing profiles throwing exceptions
    if (model->processProfiles.count(deviceTypeName) == 0 || 
        model->processProfiles.at(deviceTypeName).batchInfer.count(model->batchSize) == 0) {
        return 0;
    }
    
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    uint64_t prepLat = profile.batchInfer.at(model->batchSize).p95prepLat;
    uint64_t postLat = profile.batchInfer.at(model->batchSize).p95postLat;
    
    // Guard against division by zero
    uint64_t totalLatency = inferenceLatency + prepLat + postLat;
    if (totalLatency == 0) totalLatency = 1;
    if (prepLat == 0) prepLat = 1;
    
    float indiProcessRate = 1000000.f / totalLatency;
    float indiPreprocessRate = 1000000.f / prepLat;
    float processRate, preprocessRate;
    
    while (numReplicas > 1) {
        numReplicas--;
        processRate = indiProcessRate * numReplicas;
        preprocessRate = indiPreprocessRate * numReplicas;
        // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
        if ((processRate * 0.75 < model->arrivalProfiles.arrivalRates ||
             preprocessRate * 0.9 < model->arrivalProfiles.arrivalRates)) {
            numReplicas++;
            break;
        }
        spdlog::get("container_agent")->info("Decreasing the number of replicas of model {0:s} to {1:d}", model->name, numReplicas);
    }
    return model->numReplicas - numReplicas;
}

/**
 * @brief Calculate queueing latency for each query coming to the preprocessor's queue, in microseconds
 * Queue type is expected to be M/D/1
 *
 * @param arrival_rate
 * @param preprocess_rate
 * @return uint64_t
 */
uint64_t Controller::calculateQueuingLatency(float &arrival_rate, const float &preprocess_rate) {
    if (arrival_rate == 0) arrival_rate = 1;
    
    // Protect against division by zero if preprocess_rate is somehow exactly 0.0f
    if (preprocess_rate <= 0.0001f) {
        return 999999999;
    }
    
    float rho = arrival_rate / preprocess_rate;
    
    // Protect against floating point limits and division by zero when rho == 1
    // M/D/1 queue length hits infinity as rho approaches 1.
    if (rho >= 0.99f) {
        return 999999999;
    }
    // float numQueriesInSystem = rho / (1 - rho);
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t) (averageQueueLength / arrival_rate * 1000000);
}

// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================