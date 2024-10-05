#include "scheduling-dis.h"

void Controller::queryingProfiles(TaskHandle *task)
{

    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;

    for (auto model : *pipelineModels)
    {
        if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos)
        {
            continue;
        }
        model->deviceTypeName = getDeviceTypeName(deviceList.at(model->device)->type);
        std::vector<std::string> upstreamPossibleDeviceList = model->upstreams.front().first->possibleDevices;
        std::vector<std::string> thisPossibleDeviceList = model->possibleDevices;
        std::vector<std::pair<std::string, std::string>> possibleDevicePairList;
        for (const auto &deviceName : upstreamPossibleDeviceList)
        {
            for (const auto &deviceName2 : thisPossibleDeviceList)
            {
                if (deviceName == "server" && deviceName2 != deviceName)
                {
                    continue;
                }
                possibleDevicePairList.push_back({deviceName, deviceName2});
            }
        }
        std::string containerName = model->name + "_" + model->deviceTypeName;
        if (!task->tk_newlyAdded)
        {
            model->arrivalProfiles.arrivalRates = queryArrivalRate(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                ctrl_containerLib[containerName].taskName,
                ctrl_containerLib[containerName].modelName,
                // TODO: Change back once we have profilings in every fps
                //ctrl_systemFPS
                15);
        }

        for (const auto &pair : possibleDevicePairList)
        {
            std::string senderDeviceType = getDeviceTypeName(deviceList.at(pair.first)->type);
            std::string receiverDeviceType = getDeviceTypeName(deviceList.at(pair.second)->type);
            containerName = model->name + "_" + receiverDeviceType;
            std::unique_lock lock(devices.getDevice(pair.first)->nodeHandleMutex);
            NetworkEntryType entry = devices.getDevice(pair.first)->latestNetworkEntries[receiverDeviceType];
            lock.unlock();
            NetworkProfile test = queryNetworkProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                ctrl_containerLib[containerName].taskName,
                ctrl_containerLib[containerName].modelName,
                pair.first,
                senderDeviceType,
                pair.second,
                receiverDeviceType,
                entry,
                // TODO: Change back once we have profilings in every fps
                //ctrl_systemFPS
                15);
            model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
        }

        for (const auto deviceName : model->possibleDevices)
        {
            std::string deviceTypeName = getDeviceTypeName(deviceList.at(deviceName)->type);
            containerName = model->name + "_" + deviceTypeName;
            ModelProfile profile = queryModelProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                deviceName,
                deviceTypeName,
                ctrl_containerLib[containerName].modelName,
                // TODO: Change back once we have profilings in every fps
                //ctrl_systemFPS
                15);
            model->processProfiles[deviceTypeName] = profile;
        }
    }
}

void Controller::estimateTimeBudgetLeft(PipelineModel *currModel)
{
    if (currModel->name.find("sink") != std::string::npos)
    {
        currModel->timeBudgetLeft = 0;
        return;
    } else if (currModel->name.find("datasource") != std::string::npos) {
        currModel->timeBudgetLeft = currModel->task->tk_slo;
    }
    
    uint64_t dnstreamBudget = 0;
    for (const auto &d : currModel->downstreams)
    {
        estimateTimeBudgetLeft(d.first);
        dnstreamBudget = std::max(dnstreamBudget, d.first->timeBudgetLeft);
    }
    currModel->timeBudgetLeft = dnstreamBudget * 1.2 +
                                (currModel->expectedQueueingLatency + currModel->expectedMaxProcessLatency) * 1.2;
}

void Controller::Scheduling()
{
    while (running)
    {
        Stopwatch schedulingSW;
        schedulingSW.start();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - ctrl_nextSchedulingTime).count() < 10) {
            continue;
        }

        ctrl_unscheduledPipelines = ctrl_savedUnscheduledPipelines;
        auto taskList = ctrl_unscheduledPipelines.getMap();

        if (!isPipelineInitialised) {
            continue;
        }

        for (auto &taskPair : taskList)
        {
            auto task = taskPair.second;
            queryingProfiles(task);
            // Adding taskname to model name for clarity
            for (auto &model : task->tk_pipelineModels)
            {
                model->name = task->tk_name + "_" + model->name;
            }
        }



        // init Partitioner
        Partitioner partitioner;
        PipelineModel model;
        float ratio = 0.3;

        partitioner.BaseParPoint = ratio;

        Dis::scheduleBaseParPointLoop(&partitioner, devices, ctrl_unscheduledPipelines);
        Dis::scheduleFineGrainedParPointLoop(&partitioner, devices, ctrl_unscheduledPipelines);
        Dis::DecideAndMoveContainer(devices, ctrl_unscheduledPipelines, &partitioner, 2);

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
                
                estimateModelNetworkLatency(model);
                estimateModelLatency(model);
            }
            for (auto &model : taskPair.second->tk_pipelineModels)
            {
                if (model->name.find("datasource") == std::string::npos)
                {
                    continue;
                }
                estimateTimeBudgetLeft(model);
            }
        }

        ctrl_scheduledPipelines = ctrl_unscheduledPipelines;

        ApplyScheduling();
        std::cout << "end_scheduleBaseParPoint " << partitioner.BaseParPoint << std::endl;
        std::cout << "end_FineGrainedParPoint " << partitioner.FineGrainedOffset << std::endl;
        schedulingSW.stop();
        ctrl_nextSchedulingTime = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_schedulingIntervalSec);
        std::this_thread::sleep_for(TimePrecisionType((ctrl_schedulingIntervalSec + 1) * 1000000 - schedulingSW.elapsed_microseconds()));
    }
}

/**
 * @brief estimate the different types of latency, in microseconds
 * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
 *
 * @param model infomation about the model
 * @param modelType
 */
void Controller::estimateModelLatency(PipelineModel *currModel) {
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
    float preprocessRate = 1000000.f / preprocessLatency;

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
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos) {
        currModel->expectedTransferLatency = 0;
        return;
    }

    currModel->expectedTransferLatency = currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(currModel->device, currModel->upstreams[0].first->device)].p95TransferDuration;
}

/**
 * @brief Calculate queueing latency for each query coming to the preprocessor's queue, in microseconds
 * Queue type is expected to be M/D/1
 *
 * @param arrival_rate
 * @param preprocess_rate
 * @return uint64_t
 */
uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate)
{
    float rho = arrival_rate / preprocess_rate;
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t)(averageQueueLength / arrival_rate * 1000000);
}

///////////////////////////////////////////////////////////////////////distream add//////////////////////////////////////////////////////////////////////////////////////

double Dis::calculateTotalprocessedRate(Devices &nodes, Tasks &pipelines, bool is_edge)
{
    double totalRequestRate = 0.0;
    std::map<std::string, NodeHandle *> deviceList = nodes.getMap();

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
            // std::cout << "calculateTotalprocessedRate deviceType " << deviceType << std::endl;
            // make sure the calculation is only for edge / server, because we need to is_edge to make sure which side information we need.
            if ((is_edge && deviceType != "server") || (!is_edge && deviceType == "server" && model->name.find("sink") == std::string::npos))
            {
                int batchInfer;
                if (is_edge)
                {
                    // calculate the info only on edge side
                    batchInfer = model->processProfiles[deviceType].batchInfer[8].p95inferLat;
                    // std::cout << "edge_batchInfer" << batchInfer << std::endl;
                }
                else
                {
                    // calculate info only the server side
                    batchInfer = model->processProfiles[deviceType].batchInfer[16].p95inferLat;
                    // std::cout << "server_batchInfer" << batchInfer << std::endl;
                }

                // calculate the tp because is ms so we need devided by 1000000
                double requestRate = (batchInfer == 0) ? 0.0 : 1000000.0 / batchInfer;
                totalRequestRate += requestRate;
                // std::cout << "totalRequestRate " << totalRequestRate << std::endl;
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
    std::map<std::string, NodeHandle *> deviceList = nodes.getMap();

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
            // std::cout << "calculateTotalprocessedRate deviceType " << deviceType << std::endl;
            // make sure the calculation is only for edge / server, because we need to is_edge to make sure which side information we need.
            if ((is_edge && deviceType != "server" && model->name.find("datasource") == std::string::npos) || 
                (!is_edge && deviceType == "server" && model->name.find("sink") == std::string::npos))
            {
                int queue;
                if (is_edge)
                {
                    // calculate the queue only on edge
                    queue = model->arrivalProfiles.arrivalRates;
                    // std::cout << "edge_queue" << queue << std::endl;
                }
                else
                {
                    // calculate the queue only on server
                    queue = model->arrivalProfiles.arrivalRates;
                    // std::cout << "server_queue" << queue << std::endl;
                }

                // add all the nodes queue
                double totalqueue = (queue == 0) ? 0.0 : queue;
                totalQueue += totalqueue;
                // std::cout << "totalRequestRate " << totalQueue << std::endl;
            }
        }
    }

    return totalQueue;
}

// calculate the BaseParPoint based on the TP
void Dis::scheduleBaseParPointLoop(Partitioner *partitioner, Devices &nodes, Tasks &pipelines)
{
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
void Dis::scheduleFineGrainedParPointLoop(Partitioner *partitioner, Devices &nodes, Tasks &pipelines)
{
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
        //  If there's no queue on the edge, set a default adjustment factor
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

void Dis::DecideAndMoveContainer(Devices &nodes, Tasks &pipelines, Partitioner *partitioner,
                                 int cuda_device)
{
    // Calculate the decision point by adding the base and fine grained partition
    float decisionPoint = partitioner->BaseParPoint + partitioner->FineGrainedOffset*0.2;
    // tolerance threshold for decision making
    float tolerance = 0.1;
    // ratio for current worload 
    float ratio = 0.0f;
    // ContainerHandle *selectedContainer = nullptr;

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
                    for (auto &upstreamPair : model->upstreams)
                    {
                        auto *upstreamModel = upstreamPair.first; // upstream pointer

                        // lock for change information
                        std::lock_guard<std::mutex> lock(upstreamModel->pipelineModelMutex);

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

void Controller::colocationTemporalScheduling() {} // Dummy Method for Compiler
