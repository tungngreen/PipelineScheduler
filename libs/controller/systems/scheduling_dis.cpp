#include "scheduling_dis.h"

void Controller::queryingProfiles(TaskHandle *task)
{
    if (!task) return;

    std::map<std::string, std::shared_ptr<NodeHandle>> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;

    //if task->tk_name ends with number, remove it
    std::string sanitizedTaskName = task->tk_name;
    if (!task->tk_name.empty() && task->tk_name.back() >= '0' && task->tk_name.back() <= '9') {
        sanitizedTaskName = task->tk_name.substr(0, task->tk_name.size() - 1);
    }
    std::string source = task->tk_source.substr(task->tk_source.find_last_of('/') + 1);

    for (auto model : *pipelineModels)
    {
        if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos)
        {
            continue;
        }

        // Ensure the device exists before accessing to prevent out-of-bounds crash
        if (deviceList.find(model->device) == deviceList.end()) {
            spdlog::get("container_agent")->error("Device {} not found in deviceList", model->device);
            continue;
        }
        model->deviceTypeName = getDeviceTypeName(deviceList.at(model->device)->type);
        
        std::vector<std::string> upstreamPossibleDeviceList;
        if (!model->upstreams.empty()) {
            if (auto upNode = model->upstreams.front().targetNode.lock()) {
                upstreamPossibleDeviceList = upNode->possibleDevices;
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

        if (!task->tk_newlyAdded)
        {
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

void Controller::estimateTimeBudgetLeft(PipelineModel *currModel)
{
    if (!currModel) return;

    if (currModel->name.find("sink") != std::string::npos)
    {
        currModel->timeBudgetLeft = 0;
        return;
    } else if (currModel->name.find("datasource") != std::string::npos) {
        if (auto task = currModel->task.lock()) {
            currModel->timeBudgetLeft = task->tk_slo;
        }
    }
    
    uint64_t dnstreamBudget = 0;
    for (const auto &d : currModel->downstreams)
    {
        if (auto dnNode = d.targetNode.lock()) {
            estimateTimeBudgetLeft(dnNode.get());
            dnstreamBudget = std::max(dnstreamBudget, dnNode->timeBudgetLeft);
        }
    }
    currModel->timeBudgetLeft = dnstreamBudget * 1.2 +
                                (currModel->expectedQueueingLatency + currModel->expectedMaxProcessLatency) * 1.2;
}

void Controller::Scheduling()
{
    // Map network messages to handler functions
    api_handlers = {
        {MSG_TYPE[START_TASK], std::bind(&Controller::HandleStartTask, this, std::placeholders::_1)}
    };

    // Initialize the global periodic timer
    ctrl_nextSchedulingTime = std::chrono::system_clock::now();

    while (running)
    {
        Stopwatch schedulingSW;
        schedulingSW.start();

        auto now = std::chrono::system_clock::now();
        
        /**
         * @brief Block 1: Calculate the global timeout for the ZeroMQ socket
         * We wait until the next periodic tick, or process immediately if a message arrives.
         */
        int timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ctrl_nextSchedulingTime - now).count();
        if (timeout_ms < 0) timeout_ms = 0; // Prevent negative timeouts if lagging

        // Set the socket timeout. It blocks until a message arrives OR the timer hits
        api_socket.set(zmq::sockopt::rcvtimeo, timeout_ms);

        /**
         * @brief Block 2: Event-Driven Execution
         */
        bool triggered_by_event = false;
        zmq::message_t message;

        try {
            if (api_socket.recv(message, zmq::recv_flags::none)) {
                std::string raw = message.to_string();
                std::istringstream iss(raw);
                std::string topic;
                iss >> topic;
                iss.get(); // skip the space after the topic
                std::string payload((std::istreambuf_iterator<char>(iss)),
                                    std::istreambuf_iterator<char>());
                if (api_handlers.count(topic)) {
                    // This updates ctrl_savedUnscheduledPipelines
                    api_handlers[topic](payload);
                    triggered_by_event = true;
                } else {
                    spdlog::get("container_agent")->error("Received unknown topic: {}", topic);
                }
            }
        } catch (const zmq::error_t& e) {
            // Timeout reached, proceed to periodic checks.
        }

        /**
         * @brief Block 3: Periodic Execution Check
         */
        now = std::chrono::system_clock::now();
        bool periodic_update_needed = (now >= ctrl_nextSchedulingTime);

        /**
         * @brief Block 4: Distream Execution
         * If triggered by an event OR the periodic timer, we run the algorithm.
         */
        if ((triggered_by_event || periodic_update_needed) && isPipelineInitialised) 
        {
            ctrl_unscheduledPipelines = ctrl_savedUnscheduledPipelines;
            auto taskList = ctrl_unscheduledPipelines.getMap();

            for (auto &taskPair : taskList)
            {
                auto task = taskPair.second;
                queryingProfiles(task.get());
                
                // Adding taskname to model name for clarity
                for (auto &model : task->tk_pipelineModels)
                {
                    if (model->name.find(task->tk_name + "_") != 0) {
                        model->name = task->tk_name + "_" + model->name;
                    }
                }
            }

            auto partitioner = std::make_shared<Partitioner>();
            float ratio = 0.3;

            partitioner->BaseParPoint = ratio;

            Dis::scheduleBaseParPointLoop(partitioner, devices, ctrl_unscheduledPipelines);
            Dis::scheduleFineGrainedParPointLoop(partitioner, devices, ctrl_unscheduledPipelines);
            Dis::DecideAndMoveContainer(devices, ctrl_unscheduledPipelines, partitioner, 2);

            for (auto &taskPair : taskList)
            {
                for (auto &model : taskPair.second->tk_pipelineModels)
                {
                    if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos)
                    {
                        continue;
                    }
                    if (model->name.find("yolov5") != std::string::npos)
                    {
                        model->batchSize = ctrl_initialBatchSizes["yolov5"];
                    }
                    else if (model->device != "server")
                    {
                        model->batchSize = ctrl_initialBatchSizes["edge"];
                    }
                    else
                    {
                        model->batchSize = ctrl_initialBatchSizes["server"];
                    }
                    
                    estimateModelNetworkLatency(model.get());
                    estimateModelLatency(model.get());
                }
                for (auto &model : taskPair.second->tk_pipelineModels)
                {
                    if (model->name.find("datasource") == std::string::npos)
                    {
                        continue;
                    }
                    estimateTimeBudgetLeft(model.get());
                }
            }

            // IMPORTANT DO-NOT-DELETE: This backup keeps the current scheduled pipelines objects as well as their pipelineModel objects alive until ApplyScheduling finishes,
            auto backupScheduledPipelines = ctrl_scheduledPipelines.getMap();
            ctrl_scheduledPipelines = ctrl_unscheduledPipelines;

            ApplyScheduling();
            
            std::cout << "end_scheduleBaseParPoint " << partitioner->BaseParPoint << std::endl;
            std::cout << "end_FineGrainedParPoint " << partitioner->FineGrainedOffset << std::endl;

            // Reset the global periodic timer after a successful schedule run
            ctrl_nextSchedulingTime = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_schedulingIntervalSec);
        }

        schedulingSW.stop();
        if (startTime == std::chrono::system_clock::time_point()) startTime = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(std::chrono::system_clock::now() - startTime).count() > ctrl_runtime) {
            running = false;
            break;
        }
    }
}

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
    if (task == nullptr) {
        spdlog::get("container_agent")->error("Failed to create task from request: {}", msg);
        return;
    }

    api_socket.send(zmq::message_t("success"), zmq::send_flags::dontwait);
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
    ModelProfile profile = currModel->processProfiles[deviceTypeName];
    BatchSizeType batchSize = currModel->batchSize;
    uint64_t preprocessLatency = profile.batchInfer[batchSize].p95prepLat;
    uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
    uint64_t postprocessLatency = profile.batchInfer[batchSize].p95postLat;
    
    float preprocessRate = (preprocessLatency > 0) ? (1000000.f / preprocessLatency) : 0.0f;

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

    currModel->expectedTransferLatency = 0;

    if (!currModel->upstreams.empty()) {
        if (auto upNode = currModel->upstreams.front().targetNode.lock()) {
            currModel->expectedTransferLatency = currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(currModel->device, upNode->device)].p95TransferDuration;
        }
    }
}

/**
 * @brief Calculate queueing latency for each query coming to the preprocessor's queue, in microseconds
 * Queue type is expected to be M/D/1
 *
 * @param arrival_rate
 * @param preprocess_rate
 * @return uint64_t
 */
uint64_t Controller::calculateQueuingLatency(float &arrival_rate, const float &preprocess_rate)
{
    // Prevent divide-by-zero and negative queue lengths
    if (arrival_rate == 0 || preprocess_rate == 0 || arrival_rate >= preprocess_rate)
    {
        return 0;
    }
    
    float rho = arrival_rate / preprocess_rate;
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t)(averageQueueLength / arrival_rate * 1000000);
}

///////////////////////////////////////////////////////////////////////distream add//////////////////////////////////////////////////////////////////////////////////////

double Dis::calculateTotalprocessedRate(Devices &nodes, Tasks &pipelines, bool is_edge)
{
    double totalRequestRate = 0.0;
    std::map<std::string, std::shared_ptr<NodeHandle>> deviceList = nodes.getMap();

    // Iterate over all unscheduled pipeline tasks
    for (const auto &taskPair : pipelines.getMap())
    {
        const auto &task = taskPair.second;
        // Iterate over all models in the task's pipeline
        for (auto &model : task->tk_pipelineModels)
        {
            if (deviceList.find(model->device) == deviceList.end())
            {
                continue;
            }

            // get devicename for the information for get the batchinfer for next step
            std::string deviceType = getDeviceTypeName(deviceList.at(model->device)->type);
            
            // make sure the calculation is only for edge / server, because we need to is_edge to make sure which side information we need.
            if ((is_edge && deviceType != "server") || (!is_edge && deviceType == "server" && model->name.find("sink") == std::string::npos))
            {
                int batchInfer;
                if (is_edge)
                {
                    // calculate the info only on edge side
                    batchInfer = model->processProfiles[deviceType].batchInfer[8].p95inferLat;
                }
                else
                {
                    // calculate info only the server side
                    batchInfer = model->processProfiles[deviceType].batchInfer[16].p95inferLat;
                }

                // calculate the tp because is ms so we need devided by 1000000
                double requestRate = (batchInfer == 0) ? 0.0 : 1000000.0 / batchInfer;
                totalRequestRate += requestRate;
            }
        }
    }

    return totalRequestRate;
}

// calculate the queue based on arrival rate
int Dis::calculateTotalQueue(Devices &nodes, Tasks &pipelines, bool is_edge)
{
    // init the info
    double totalQueue = 0.0;
    std::map<std::string, std::shared_ptr<NodeHandle>> deviceList = nodes.getMap();

    // for loop every model in the system
    for (const auto &taskPair : pipelines.getMap())
    {
        const auto &task = taskPair.second;
        for (auto &model : task->tk_pipelineModels)
        {
            if (deviceList.find(model->device) == deviceList.end())
            {
                continue;
            }

            std::string deviceType = getDeviceTypeName(deviceList.at(model->device)->type);
            
            // make sure the calculation is only for edge / server, because we need to is_edge to make sure which side information we need.
            if ((is_edge && deviceType != "server" && model->name.find("datasource") == std::string::npos) || 
                (!is_edge && deviceType == "server" && model->name.find("sink") == std::string::npos))
            {
                int queue;
                if (is_edge)
                {
                    // calculate the queue only on edge
                    queue = model->arrivalProfiles.arrivalRates;
                }
                else
                {
                    // calculate the queue only on server
                    queue = model->arrivalProfiles.arrivalRates;
                }

                // add all the nodes queue
                double totalqueue = (queue == 0) ? 0.0 : queue;
                totalQueue += totalqueue;
            }
        }
    }

    return totalQueue;
}

// calculate the BaseParPoint based on the TP
void Dis::scheduleBaseParPointLoop(std::shared_ptr<Partitioner> partitioner, Devices &nodes, Tasks &pipelines)
{
    if (!partitioner) return;
    // init the data
    float TPedgesAvg = 0.0f;
    float TPserverAvg = 0.0f;
    const float smooth = 0.4f;

    while (true)
    {
        // get the TP on edge and server sides.
        float TPEdges = calculateTotalprocessedRate(nodes, pipelines, true);
        std::cout << "TPEdges: " << TPEdges << std::endl;
        float TPServer = calculateTotalprocessedRate(nodes, pipelines, false);
        std::cout << "TPServer: " << TPServer << std::endl;

        // init the TPedgesAvg and TPserverAvg based on the current runtime
        TPedgesAvg = smooth * TPedgesAvg + (1 - smooth) * TPEdges;
        TPserverAvg = smooth * TPserverAvg + (1 - smooth) * TPServer; // this is server throughput
        std::cout << " TPserverAvg:" << TPserverAvg << std::endl;

        // partition the parpoint, calculate based on the TP
        if (TPedgesAvg > TPserverAvg + 10 * 4)
        {
            if (TPedgesAvg > 1.5 * TPserverAvg)
            {
                partitioner->BaseParPoint += 0.006f;
            }
            else if (TPedgesAvg > 1.3 * TPserverAvg)
            {
                partitioner->BaseParPoint += 0.003f;
            }
            else
            {
                partitioner->BaseParPoint += 0.001f;
            }
        }
        else if (TPedgesAvg < TPserverAvg - 10 * 4)
        {
            if (1.5 * TPedgesAvg < TPserverAvg)
            {
                partitioner->BaseParPoint -= 0.006f;
            }
            else if (1.3 * TPedgesAvg < TPserverAvg)
            {
                partitioner->BaseParPoint -= 0.003f;
            }
            else
            {
                partitioner->BaseParPoint -= 0.001f;
            }
        }

        if (partitioner->BaseParPoint > 1)
        {
            partitioner->BaseParPoint = 1;
        }
        else if (partitioner->BaseParPoint < 0)
        {
            partitioner->BaseParPoint = 0;
        }
        break;
    }
}

// fine grained the parpoint based on the queue
void Dis::scheduleFineGrainedParPointLoop(std::shared_ptr<Partitioner> partitioner, Devices &nodes, Tasks &pipelines)
{
    if (!partitioner) return;

    float w;
    float tmp;
    while (true)
    {
        // get edge and server sides queue data
        float wbar = calculateTotalQueue(nodes, pipelines, true);
        std::cout << "wbar " << wbar << std::endl;
        w = calculateTotalQueue(nodes, pipelines, false);
        std::cout << "w " << w << std::endl;
        
        // based on the queue sides to claculate the fine grained point
        // If there's no queue on the edge, set a default adjustment factor
        if (wbar == 0)
        {
            tmp = 1.0f;
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        // Otherwise, calculate the fine grained offset based on the relative queue sizes
        else
        {
            tmp = (wbar - w) / std::max(wbar, w);
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        break;
    }
}

void Dis::DecideAndMoveContainer(Devices &nodes, Tasks &pipelines, std::shared_ptr<Partitioner> partitioner,
                                 int cuda_device)
{
    if (!partitioner) return;

    // Calculate the decision point by adding the base and fine grained partition
    float decisionPoint = partitioner->BaseParPoint + partitioner->FineGrainedOffset*0.2;
    // tolerance threshold for decision making
    float tolerance = 0.1;
    // ratio for current worload 
    float ratio = 0.0f;

    if (calculateTotalQueue(nodes, pipelines, false) != 0)
    {                                                  
        ratio = calculateTotalQueue(nodes, pipelines, true) / calculateTotalQueue(nodes, pipelines, false);
        ratio = std::max(0.0f, std::min(1.0f, ratio)); 
    }

    // the decisionpoint is much larger than the current workload that means we need give the edge more work
    if (decisionPoint > ratio + tolerance)
    {
        std::cout << "Move Container from server to edge based on model priority: " << std::endl;
        // for loop every model to find out the current splitpoint.
        for (const auto &taskPair : pipelines.getMap())
        {
            const auto &task = taskPair.second;

            for (auto &model : task->tk_pipelineModels)
            {
                // we don't move the datasource and sink because it has to be on edge or server
                if (model->isSplitPoint && model->name.find("datasource") == std::string::npos && model->name.find("sink") == std::string::npos)
                {
                    std::lock_guard<std::mutex> lock(model->pipelineModelMutex);

                    // change the device from server to the source of edge device
                    if (model->device == "server")
                    {
                        model->device = task->tk_src_device;
                    }
                }
            }
        }
    }
    // Similar logic for the server side
    if (decisionPoint < ratio - tolerance)
    {
        std::cout << "Move Container from edge to server based on model priority: " << std::endl;
        for (const auto &taskPair : pipelines.getMap())
        {
            const auto &task = taskPair.second;

            for (auto &model : task->tk_pipelineModels)
            {
                if (model->isSplitPoint && 
                    model->name.find("datasource") == std::string::npos && 
                    model->name.find("sink") == std::string::npos)
                {
                    std::lock_guard<std::mutex> lock(model->pipelineModelMutex);

                    if (model->device != "server")
                    {
                        model->device = "server";
                    }
                    
                }
                {
                    // because we need tp move container from edge to server so we have to move the upstream.
                    for (auto &upstreamEdge : model->upstreams)
                    {
                        if (auto upstreamModel = upstreamEdge.targetNode.lock()) {
                            // lock for change information
                            std::lock_guard<std::mutex> upLock(upstreamModel->pipelineModelMutex);

                            // move the container from edge to server
                            if (upstreamModel->device != "server")
                            {
                                upstreamModel->device = "server";
                            }
                        }
                    }
                }
            }
        }
    }
}

void Controller::colocationTemporalScheduling() {} // Dummy Method for Compiler
