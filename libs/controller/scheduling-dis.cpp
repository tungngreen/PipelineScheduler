#include "scheduling-dis.h"
// #include "controller.h"

// ==================================================================Scheduling==================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

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
        std::string containerName = model->name + "-" + model->deviceTypeName;
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
                ctrl_systemFPS);
        }

        for (const auto &pair : possibleDevicePairList)
        {
            std::string senderDeviceType = getDeviceTypeName(deviceList.at(pair.first)->type);
            std::string receiverDeviceType = getDeviceTypeName(deviceList.at(pair.second)->type);
            containerName = model->name + "-" + receiverDeviceType;
            std::unique_lock lock(devices.list[pair.first]->nodeHandleMutex);
            NetworkEntryType entry = devices.list[pair.first]->latestNetworkEntries[receiverDeviceType];
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
                ctrl_systemFPS);
            model->arrivalProfiles.d2dNetworkProfile[std::make_pair(pair.first, pair.second)] = test;
        }

        for (const auto deviceName : model->possibleDevices)
        {
            std::string deviceTypeName = getDeviceTypeName(deviceList.at(deviceName)->type);
            containerName = model->name + "-" + deviceTypeName;
            ModelProfile profile = queryModelProfile(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                task->tk_name,
                task->tk_source,
                deviceName,
                deviceTypeName,
                ctrl_containerLib[containerName].modelName,
                ctrl_systemFPS);
            model->processProfiles[deviceTypeName] = profile;
        }

        // ModelArrivalProfile profile = queryModelArrivalProfile(
        //     *ctrl_metricsServerConn,
        //     ctrl_experimentName,
        //     ctrl_systemName,
        //     t.name,
        //     t.source,
        //     ctrl_containerLib[containerName].taskName,
        //     ctrl_containerLib[containerName].modelName,
        //     possibleDeviceList,
        //     possibleNetworkEntryPairs
        // );
        // std::cout << "sdfsdfasdf" << std::endl;
    }
}

void Controller::Scheduling()
{
    while (running)
    {
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        // std::vector<NodeHandle*> nodes;
        NodeHandle *edgePointer = nullptr;
        NodeHandle *serverPointer = nullptr;
        unsigned long totalEdgeMemory = 0, totalServerMemory = 0;
        // std::vector<std::unique_ptr<NodeHandle>> nodes;
        // int cuda_device = 2; // need to be add
        // std::unique_lock<std::mutex> lock(nodeHandleMutex);
        // std::unique_lock<std::mutex> lock(devices.devicesMutex);
        // for (const auto &devicePair : devices.list)
        // {
        //     nodes.push_back(devicePair.second);
        // }
        nodes.clear();

        auto pointers = devices.getList();

        {
            std::vector<NodeHandle> localNodes;
            for (const auto &ptr : pointers)
            {
                if (ptr != nullptr)
                {
                    localNodes.push_back(*ptr);
                }
            }

            nodes.swap(localNodes);
        }

        // init Partitioner
        Partitioner partitioner;
        PipelineModel model;
        float ratio = 0.3;
        std::cout << "ratio" << std::endl;

        partitioner.BaseParPoint = ratio;

        scheduleBaseParPointLoop(&model, &partitioner, nodes);
        scheduleFineGrainedParPointLoop(&partitioner, nodes);
        DecideAndMoveContainer(&model, nodes, &partitioner, 2);
        // ctrl_scheduledPipelines = ctrl_unscheduledPipelines;
        ApplyScheduling();
        std::cout << "end_scheduleBaseParPoint " << partitioner.BaseParPoint << std::endl;
        std::cout << "end_FineGrainedParPoint " << partitioner.FineGrainedOffset << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(
            5000)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now
    }
}

bool Controller::containerTemporalScheduling(ContainerHandle *container)
{
}

bool Controller::modelTemporalScheduling(PipelineModel *pipelineModel)
{
    if (pipelineModel->name == "datasource" || pipelineModel->name == "sink")
    {
        return true;
    }
    for (auto container : pipelineModel->manifestations)
    {
        containerTemporalScheduling(container);
    }
    for (auto downstream : pipelineModel->downstreams)
    {
        modelTemporalScheduling(downstream.first);
    }
    return true;
}

void Controller::temporalScheduling()
{
    for (auto &[taskName, taskHandle] : ctrl_scheduledPipelines.list)
    {
    }
}

bool Controller::mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile)
{
    mergedProfile.arrivalRates += toBeMergedProfile.arrivalRates;
    auto mergedD2DProfile = &mergedProfile.d2dNetworkProfile;
    auto toBeMergedD2DProfile = &toBeMergedProfile.d2dNetworkProfile;
    for (const auto &[pair, profile] : toBeMergedProfile.d2dNetworkProfile)
    {
        mergedD2DProfile->at(pair).p95TransferDuration = std::max(mergedD2DProfile->at(pair).p95TransferDuration,
                                                                  toBeMergedD2DProfile->at(pair).p95TransferDuration);
        mergedD2DProfile->at(pair).p95PackageSize = std::max(mergedD2DProfile->at(pair).p95PackageSize,
                                                             toBeMergedD2DProfile->at(pair).p95PackageSize);
    }
    return true;
}

bool Controller::mergeProcessProfiles(PerDeviceModelProfileType &mergedProfile, const PerDeviceModelProfileType &toBeMergedProfile)
{
    for (const auto &[deviceName, profile] : toBeMergedProfile)
    {
        auto mergedProfileDevice = &mergedProfile[deviceName];
        auto toBeMergedProfileDevice = &toBeMergedProfile.at(deviceName);

        BatchSizeType batchSize =

            mergedProfileDevice->p95InputSize = std::max(mergedProfileDevice->p95InputSize, toBeMergedProfileDevice->p95InputSize);
        mergedProfileDevice->p95OutputSize = std::max(mergedProfileDevice->p95OutputSize, toBeMergedProfileDevice->p95OutputSize);
        // mergedProfileDevice->p95prepLat = std::max(mergedProfileDevice->p95prepLat, toBeMergedProfileDevice->p95prepLat);
        // mergedProfileDevice->p95postLat = std::max(mergedProfileDevice->p95postLat, toBeMergedProfileDevice->p95postLat);

        auto mergedBatchInfer = &mergedProfileDevice->batchInfer;
        // auto toBeMergedBatchInfer = &toBeMergedProfileDevice->batchInfer;

        for (const auto &[batchSize, p] : toBeMergedProfileDevice->batchInfer)
        {
            mergedBatchInfer->at(batchSize).p95inferLat = std::max(mergedBatchInfer->at(batchSize).p95inferLat, p.p95inferLat);
            mergedBatchInfer->at(batchSize).p95prepLat = std::max(mergedBatchInfer->at(batchSize).p95prepLat, p.p95prepLat);
            mergedBatchInfer->at(batchSize).p95postLat = std::max(mergedBatchInfer->at(batchSize).p95postLat, p.p95postLat);
            mergedBatchInfer->at(batchSize).cpuUtil = std::max(mergedBatchInfer->at(batchSize).cpuUtil, p.cpuUtil);
            mergedBatchInfer->at(batchSize).gpuUtil = std::max(mergedBatchInfer->at(batchSize).gpuUtil, p.gpuUtil);
            mergedBatchInfer->at(batchSize).memUsage = std::max(mergedBatchInfer->at(batchSize).memUsage, p.memUsage);
            mergedBatchInfer->at(batchSize).rssMemUsage = std::max(mergedBatchInfer->at(batchSize).rssMemUsage, p.rssMemUsage);
            mergedBatchInfer->at(batchSize).gpuMemUsage = std::max(mergedBatchInfer->at(batchSize).gpuMemUsage, p.gpuMemUsage);
        }
    }
    return true;
}

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel *toBeMergedModel)
{
    // If the merged model is empty, we should just copy the model to be merged
    if (mergedModel->numReplicas == 0)
    {
        *mergedModel = *toBeMergedModel;
        return true;
    }
    // If the devices are different, we should not merge the models
    if (mergedModel->device != toBeMergedModel->device || toBeMergedModel->merged)
    {
        return false;
    }

    mergeArrivalProfiles(mergedModel->arrivalProfiles, toBeMergedModel->arrivalProfiles);
    mergeProcessProfiles(mergedModel->processProfiles, toBeMergedModel->processProfiles);

    bool merged = false;
    toBeMergedModel->merged = true;
}

TaskHandle Controller::mergePipelines(const std::string &taskName)
{
    TaskHandle mergedPipeline;
    auto mergedPipelineModels = &(mergedPipeline.tk_pipelineModels);

    auto unscheduledTasks = ctrl_unscheduledPipelines.getMap();

    *mergedPipelineModels = getModelsByPipelineType(unscheduledTasks.at(taskName)->tk_type, "server");
    uint16_t numModels = mergedPipeline.tk_pipelineModels.size();

    for (uint16_t i = 0; i < numModels; i++)
    {
        if (mergedPipelineModels->at(i)->name == "datasource")
        {
            continue;
        }
        for (const auto &task : unscheduledTasks)
        {
            if (task.first == taskName)
            {
                continue;
            }
            mergeModels(mergedPipelineModels->at(i), task.second->tk_pipelineModels.at(i));
        }
        auto numIncReps = incNumReplicas(mergedPipelineModels->at(i));
        mergedPipelineModels->at(i)->numReplicas += numIncReps;
        auto deviceList = devices.getMap();
        for (auto j = 0; j < mergedPipelineModels->at(i)->numReplicas; j++)
        {
            mergedPipelineModels->at(i)->manifestations.emplace_back(new ContainerHandle{});
            mergedPipelineModels->at(i)->manifestations.back()->task = &mergedPipeline;
            mergedPipelineModels->at(i)->manifestations.back()->device_agent = deviceList.at(mergedPipelineModels->at(i)->device);
        }
    }
}

void Controller::mergePipelines()
{
    std::vector<std::string> toMerge = {"traffic", "people"};
    TaskHandle mergedPipeline;

    for (const auto &taskName : toMerge)
    {
        mergedPipeline = mergePipelines(taskName);
        std::lock_guard lock(ctrl_scheduledPipelines.tasksMutex);
        ctrl_scheduledPipelines.list.insert({mergedPipeline.tk_name, &mergedPipeline});
    }
}

/**
 * @brief Recursively traverse the model tree and try shifting models to edge devices
 *
 * @param models
 * @param slo
 */
void Controller::shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string &edgeDevice)
{
    if (currModel->name == "sink")
    {
        return;
    }
    if (currModel->name == "datasource")
    {
        if (currModel->device != edgeDevice)
        {
            spdlog::get("container_agent")->warn("Edge device {0:s} is not identical to the datasource device {1:s}", edgeDevice, currModel->device);
        }
        return;
    }

    if (currModel->device == edgeDevice)
    {
        for (auto &d : currModel->downstreams)
        {
            shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
        }
    }

    // If the edge device is not in the list of possible devices, we should not consider it
    if (std::find(currModel->possibleDevices.begin(), currModel->possibleDevices.end(), edgeDevice) == currModel->possibleDevices.end())
    {
        return;
    }

    std::string deviceTypeName = getDeviceTypeName(devices.list[edgeDevice]->type);

    uint32_t inputSize = currModel->processProfiles.at(deviceTypeName).p95InputSize;
    uint32_t outputSize = currModel->processProfiles.at(deviceTypeName).p95OutputSize;

    if (inputSize * 0.3 < outputSize)
    {
        currModel->device = edgeDevice;
        estimateModelLatency(currModel);
        for (auto &downstream : currModel->downstreams)
        {
            estimateModelLatency(downstream.first);
        }
        estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
        uint64_t expectedE2ELatency = pipeline.back()->expectedStart2HereLatency;
        // if after shifting the model to the edge device, the pipeline still meets the SLO, we should keep it

        // However, if the pipeline does not meet the SLO, we should shift reverse the model back to the server
        if (expectedE2ELatency > slo)
        {
            currModel->device = "server";
            estimateModelLatency(currModel);
            for (auto &downstream : currModel->downstreams)
            {
                estimateModelLatency(downstream.first);
            }
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
        }
    }
    // Shift downstream models to the edge device
    for (auto &d : currModel->downstreams)
    {
        shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
    }
}

/**
 * @brief
 *
 * @param models
 * @param slo
 * @param nObjects
 * @return std::map<ModelType, int>
 */
void Controller::getInitialBatchSizes(TaskHandle *task, uint64_t slo)
{

    PipelineModelListType *models = &(task->tk_pipelineModels);

    for (auto m : *models)
    {
        m->batchSize = 1;
        m->numReplicas = 1;

        estimateModelLatency(m);
    }

    // DFS-style recursively estimate the latency of a pipeline from source to sink
    // The first model should be the datasource
    estimatePipelineLatency(models->front(), 0);

    uint64_t expectedE2ELatency = models->back()->expectedStart2HereLatency;

    if (slo < expectedE2ELatency)
    {
        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
    }

    // Increase number of replicas to avoid bottlenecks
    for (auto m : *models)
    {
        auto numIncReplicas = incNumReplicas(m);
        m->numReplicas += numIncReplicas;
    }

    // Find near-optimal batch sizes
    auto foundBest = true;
    while (foundBest)
    {
        foundBest = false;
        uint64_t bestCost = models->back()->estimatedStart2HereCost;
        for (auto m : *models)
        {
            BatchSizeType oldBatchsize = m->batchSize;
            m->batchSize *= 2;
            estimateModelLatency(m);
            estimatePipelineLatency(models->front(), 0);
            expectedE2ELatency = models->back()->expectedStart2HereLatency;
            if (expectedE2ELatency < slo)
            {
                // If increasing the batch size of model `m` creates a pipeline that meets the SLO, we should keep it
                uint64_t estimatedE2Ecost = models->back()->estimatedStart2HereCost;
                // Unless the estimated E2E cost is better than the best cost, we should not consider it as a candidate
                if (estimatedE2Ecost < bestCost)
                {
                    bestCost = estimatedE2Ecost;
                    foundBest = true;
                }
                if (!foundBest)
                {
                    m->batchSize = oldBatchsize;
                    estimateModelLatency(m);
                    continue;
                }
                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
                auto numDecReplicas = decNumReplicas(m);
                m->numReplicas -= numDecReplicas;
            }
            else
            {
                m->batchSize = oldBatchsize;
                estimateModelLatency(m);
            }
        }
    }
}

/**
 * @brief estimate the different types of latency, in microseconds
 * Due to batch inference's nature, the queries that come later has to wait for more time both in preprocessor and postprocessor.
 *
 * @param model infomation about the model
 * @param modelType
 */
void Controller::estimateModelLatency(PipelineModel *currModel)
{
    std::string deviceName = currModel->device;
    // We assume datasource and sink models have no latency
    if (currModel->name == "datasource" || currModel->name == "sink")
    {
        currModel->expectedQueueingLatency = 0;
        currModel->expectedAvgPerQueryLatency = 0;
        currModel->expectedMaxProcessLatency = 0;
        currModel->estimatedPerQueryCost = 0;
        currModel->expectedStart2HereLatency = 0;
        currModel->estimatedStart2HereCost = 0;
        return;
    }
    ModelProfile profile = currModel->processProfiles[deviceName];
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
    currModel->estimatedPerQueryCost = currModel->expectedAvgPerQueryLatency + currModel->expectedQueueingLatency +
                                       currModel->expectedTransferLatency;
    currModel->expectedStart2HereLatency = 0;
    currModel->estimatedStart2HereCost = 0;
}

void Controller::estimateModelNetworkLatency(PipelineModel *currModel)
{
    if (currModel->name == "datasource" || currModel->name == "sink")
    {
        currModel->expectedTransferLatency = 0;
        return;
    }

    currModel->expectedTransferLatency = currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(currModel->device, currModel->upstreams[0].first->device)].p95TransferDuration;
}

/**
 * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
 *
 * @param pipeline provides all information about the pipeline needed for scheduling
 * @param currModel
 */
void Controller::estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency)
{
    // estimateModelLatency(currModel, currModel->device);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency
    // to reach from each upstream.
    if (currModel->name == "datasource")
    {
        currModel->expectedStart2HereLatency = start2HereLatency;
    }
    else
    {
        currModel->expectedStart2HereLatency = std::max(
            currModel->expectedStart2HereLatency,
            start2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency +
                currModel->expectedQueueingLatency);
    }

    // Cost of the pipeline until the current model
    currModel->estimatedStart2HereCost += currModel->estimatedPerQueryCost;

    std::vector<std::pair<PipelineModel *, int>> downstreams = currModel->downstreams;
    for (const auto &d : downstreams)
    {
        estimatePipelineLatency(d.first, currModel->expectedStart2HereLatency);
    }

    if (currModel->downstreams.size() == 0)
    {
        return;
    }
}

/**
 * @brief Attempts to increase the number of replicas to meet the arrival rate
 *
 * @param model the model to be scaled
 * @param deviceName
 * @return uint8_t The number of replicas to be added
 */
uint8_t Controller::incNumReplicas(const PipelineModel *model)
{
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat + profile.batchInfer.at(model->batchSize).p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (processRate < model->arrivalProfiles.arrivalRates)
    {
        numReplicas++;
        processRate = indiProcessRate * numReplicas;
    }
    return numReplicas - model->numReplicas;
}

/**
 * @brief Decrease the number of replicas as long as it is possible to meet the arrival rate
 *
 * @param model
 * @return uint8_t The number of replicas to be removed
 */
uint8_t Controller::decNumReplicas(const PipelineModel *model)
{
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1 / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat + profile.batchInfer.at(model->batchSize).p95postLat);
    float processRate = indiProcessRate * numReplicas;
    while (numReplicas > 1)
    {
        numReplicas--;
        processRate = indiProcessRate * numReplicas;
        // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
        if (processRate < model->arrivalProfiles.arrivalRates)
        {
            numReplicas++;
            break;
        }
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
uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate)
{
    float rho = arrival_rate / preprocess_rate;
    float numQueriesInSystem = rho / (1 - rho);
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t)(averageQueueLength / arrival_rate * 1000000);
}

///////////////////////////////////////////////////////////////////////distream add//////////////////////////////////////////////////////////////////////////////////////

std::pair<std::vector<NodeHandle>, std::vector<NodeHandle>> Controller::categorizeNodes(const std::vector<NodeHandle> &nodes)
{
    std::vector<NodeHandle> edges;
    std::vector<NodeHandle> servers;

    for (const auto &node : nodes)
    {
        if (node.type == NXXavier || node.type == AGXXavier || node.type == OrinNano)
        {
            edges.push_back(node);
            //  std::cout << "edge_push " << node.ip << std::endl;
        }
        else if (node.type == Server)
        {
            servers.push_back(node);
            // std::cout << "server_push " << node.ip << std::endl;
        }
    }

    return {edges, servers};
}

double Controller::calculateTotalprocessedRate(const PipelineModel *model, const std::vector<NodeHandle> &nodes, bool is_edge)
{
    auto [edges, servers] = categorizeNodes(nodes);
    double totalRequestRate = 0.0;
    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

    for (const auto &taskPair : ctrl_unscheduledPipelines.list)
    {
        const auto &task = taskPair.second;
        // queryingProfiles(task); //wrong format

        for (auto &model : task->tk_pipelineModels)
        {
            std::string deviceType = getDeviceTypeName(deviceList.at(model->device)->type);
            std::cout << "calculateTotalprocessedRate deviceType " << deviceType << std::endl;
            int batchInfer;
            if (is_edge && deviceType != "server")
            {
                batchInfer = 3000; // model->processProfiles[deviceType].batchInfer[8].p95inferLat;
                std::cout << "edge_batchInfer" << batchInfer << std::endl;
            }
            else
            {
                batchInfer = 6000; // model->processProfiles.at(deviceType).batchInfer.at(32).p95inferLat;
                // std::cout << "server_batchInfer" << batchInfer<< std::endl;
            }
            // std::cout << "batchInfer" << model->processProfiles.at(nodeType).batchInfer<< std::endl;
            // int timePerFrame = batchInfer.at(batchSize).p95inferLat;
            // std::cout << "timePerFrame" << timePerFrame << std::endl;
            // processProfiles["server"].batchInfer[8].p95inferLat

            double requestRate = (batchInfer == 0) ? 0.0 : 1000000000.0 / batchInfer;
            // std::cout << "requestRate " << requestRate << std::endl;
            totalRequestRate += requestRate;
            std::cout << "totalRequestRate " << totalRequestRate << std::endl;
        }
    }

    return totalRequestRate;
}

int Controller::calculateTotalQueue(const std::vector<NodeHandle> &nodes, bool is_edge)
{
    auto [edges, servers] = categorizeNodes(nodes);
    double totalQueue = 0.0;
    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

    // const auto &relevantNodes = is_edge ? edges : servers;
    // std::vector<std::pair<std::string, std::string>> possibleDevicePairList = {{"edge", "server"}};
    // std::map<std::pair<std::string, std::string>, NetworkEntryType> possibleNetworkEntryPairs;

    for (const auto &taskPair : ctrl_unscheduledPipelines.list)
    {
        const auto &task = taskPair.second; // everyTaskHandle

        for (const auto &model : task->tk_pipelineModels)
        { // every model
            std::string deviceType = getDeviceTypeName(deviceList.at(model->device)->type);
            std::cout << "deviceType " << deviceType << std::endl;

            std::string containerName = model->name + "-" + deviceType;
            std::cout << "containerName " << containerName << std::endl;
            if (containerName.find("datasource") != std::string::npos || containerName.find("sink") != std::string::npos)
            {
                continue;
            }

            double queueLength;
            if (is_edge == true && deviceType != "server")
            {
                double arrivalRate = 5000; // model->arrivalProfiles.arrivalRates;  is zero
                std::cout << "edge arrivalRate " << arrivalRate << std::endl;
                queueLength = 1.0 / arrivalRate; // profile.batchInfer.at(1).p95inferLat;
            }
            else
            {
                double arrivalRate = 500000; // model->arrivalProfiles.arrivalRates;  is zero
                std::cout << "server arrivalRate " << arrivalRate << std::endl;
                queueLength = 1.0 / arrivalRate; // profile.batchInfer.at(1).p95inferLat;
            }
            totalQueue += queueLength;
        }
    }

    return totalQueue;
}

double Controller::getMaxTP(const PipelineModel *model, std::vector<NodeHandle> nodes, bool is_edge)
{
    int processedRate = calculateTotalprocessedRate(model, nodes, is_edge);
    if (calculateTotalQueue(nodes, is_edge) == 0.0)
    {
        return 0;
    }
    else
    {
        return processedRate;
    }
}

void Controller::scheduleBaseParPointLoop(const PipelineModel *model, Partitioner *partitioner, std::vector<NodeHandle> nodes)
{
    float TPedgesAvg = 0.0f;
    float TPserverAvg = 0.0f;
    const float smooth = 0.4f;

    while (true)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(250));
        // float TPEdges = 0.0f;

        // auto [edges, servers] = categorizeNodes(nodes);
        float TPEdges = getMaxTP(model, nodes, true);
        std::cout << "TPEdges: " << TPEdges << std::endl;
        float TPServer = getMaxTP(model, nodes, false);
        std::cout << "TPServer: " << TPServer << std::endl;

        // init the TPedgesAvg and TPserverAvg based on the current runtime
        TPedgesAvg = smooth * TPedgesAvg + (1 - smooth) * TPEdges;
        TPserverAvg = smooth * TPserverAvg + (1 - smooth) * TPServer; // this is server throughput
        std::cout << " TPserverAvg:" << TPserverAvg << std::endl;

        // partition the parpoint
        if (TPedgesAvg > TPserverAvg + 10) //* 4)
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
        else if (TPedgesAvg < TPserverAvg - 10) //* 4)
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

// float Controller::ComputeAveragedNormalizedWorkload(const std::vector<NodeHandle> &nodes, bool is_edge)
// {
//     float sum = 0.0;
//     int N = nodes.size();
//     // float edgeQueueCapacity = 10000000000.0; // need to know the  real Capacity

//     if (N == 0)
//         return 0; // incase N=0

//     float tmp = calculateTotalQueue(nodes, is_edge);
//     sum += tmp;
//     float norm = sum / static_cast<float>(N);
//     return norm;
// }

void Controller::scheduleFineGrainedParPointLoop(Partitioner *partitioner, const std::vector<NodeHandle> &nodes)
{
    float w;
    int totalServerQueue;
    float ServerCapacity = 1000000000000.0;
    float edgeQueueCapacity = 100000000000000.0;
    float tmp;
    while (true)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(250));  // every 250 weakup
        auto [edges, servers] = categorizeNodes(nodes);

        float wbar = calculateTotalQueue(nodes, true) / edgeQueueCapacity;
        std::cout << "wbar " << wbar << std::endl;
        float totalServerQueue = calculateTotalQueue(nodes, false);
        std::cout << "totalServerQueue " << totalServerQueue << std::endl;
        float w = totalServerQueue / ServerCapacity;
        std::cout << "w " << w << std::endl;
        if (w == 0)
        {
            float tmp = 1.0f;
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        else
        {
            float tmp = (wbar - w) / std::max(wbar, w);
            // std::cout << "tmp " << tmp << std::endl;
            // std::cout << "(wbar - w) " << (wbar - w) << std::endl;
            // std::cout << "std::max(wbar, w) " << std::max(wbar, w) << std::endl;
            partitioner->FineGrainedOffset = tmp * partitioner->BaseParPoint;
        }
        // std::cout << "tmp " << tmp << std::endl;
        break;
    }
}

// float Controller::calculateRatio(const std::vector<NodeHandle> &nodes)
// {
//     auto [edges, servers] = categorizeNodes(nodes);
//     float edgeMem = 0.0f;
//     float serverMem = 0.0f;
//     float ratio = 0.0f;
//     NodeHandle *edgePointer = nullptr;
//     NodeHandle *serverPointer = nullptr;

//     for (const NodeHandle &node : nodes)
//     {
//         if (!node.type == SystemDeviceType::Server)
//         {
//             edgePointer = const_cast<NodeHandle *>(&node);
//             edgeMem += std::accumulate(node.mem_size.begin(), node.mem_size.end(), 0UL);
//         }
//         else
//         {
//             serverPointer = const_cast<NodeHandle *>(&node);
//             serverMem += std::accumulate(node.mem_size.begin(), node.mem_size.end(), 0UL);
//         }
//     }

//     if (edgePointer == nullptr)
//     {
//         std::cout << "No edge device found.\n";
//     }

//     std::cout << "Total serverMem: " << serverMem << std::endl;
//     std::cout << "Total edgeMem: " << edgeMem << std::endl;

//     if (serverMem != 0)
//     {
//         ratio = edgeMem / serverMem;
//     }
//     else
//     {
//         ratio = 0.0f;
//     }

//     std::cout << "Calculated Ratio: " << ratio << std::endl;
//     return ratio;
// }

void Controller::DecideAndMoveContainer(const PipelineModel *model, std::vector<NodeHandle> &nodes, Partitioner *partitioner, int cuda_device)
{
    float decisionPoint = partitioner->BaseParPoint + partitioner->FineGrainedOffset;
    // float ratio = 0.7;
    float tolerance = 0.1;
    auto [edges, servers] = categorizeNodes(nodes);
    float currEdgeWorkload = calculateTotalQueue(nodes, true);
    float currServerWorkload = calculateTotalQueue(nodes, false);
    float ratio = currEdgeWorkload / currServerWorkload;
    // ContainerHandle *selectedContainer = nullptr;

    // while (decisionPoint < ratio - tolerance || decisionPoint > ratio + tolerance)
    // {
    if (decisionPoint > ratio + tolerance)
    {
        std::cout << "Move Container from server to edge based on model priority: " << std::endl;
        // extern Tasks ctrl_unscheduledPipelines;
        for (const auto &taskPair : ctrl_unscheduledPipelines.list)
        {
            const auto &task = taskPair.second;

            for (auto &model : task->tk_pipelineModels)
            {
                if (model->isSplitPoint)
                {

                    std::lock_guard<std::mutex> lock(model->pipelineModelMutex);

                    if (model->device == "server")
                    {
                        model->device = model->task->tk_src_device;
                    }
                }
            }
        }
    }
    // Similar logic for the server side
    if (decisionPoint < ratio - tolerance)
    {
        std::cout << "Move Container from edge to server based on model priority: " << std::endl;
        for (const auto &taskPair : ctrl_unscheduledPipelines.list)
        {
            const auto &task = taskPair.second;

            for (auto &model : task->tk_pipelineModels)
            {
                if (model->isSplitPoint)
                {
                    // handle the upstream
                    for (auto &upstreamPair : model->upstreams)
                    {
                        auto *upstreamModel = upstreamPair.first; // upstream pointer

                        // lock for change information
                        std::lock_guard<std::mutex> lock(upstreamModel->pipelineModelMutex);

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

// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================