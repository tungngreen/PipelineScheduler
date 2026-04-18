#include "scheduling_jlf.h"
#include "controller.h"
#include <memory>
#include <spdlog/spdlog.h>
#include <unordered_set>

void Controller::queryingProfiles(TaskHandle *task)
{
    // FIX: Update to shared_ptr to match the Devices struct definition
    std::map<std::string, std::shared_ptr<NodeHandle>> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;

    std::vector<std::string> dsrcDeviceList;
    for (auto &model : *pipelineModels) {
        if (model->name.find("datasource") == std::string::npos) {
            continue;
        }
        dsrcDeviceList.push_back(model->device);
    }

    for (auto &model : *pipelineModels)
    {
        if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos)
        {
            continue;
        }
        model->deviceTypeName = getDeviceTypeName(deviceList.at(model->device)->type);
        std::vector<std::string> upstreamPossibleDeviceList;
        if (model->name.find("yolo") != std::string::npos)
        {
            upstreamPossibleDeviceList = dsrcDeviceList;
        }
        else
        {
            if (auto upNode = model->upstreams.front().targetNode.lock()) {
                upstreamPossibleDeviceList = upNode->possibleDevices;
            }
        }
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
            
            auto dev = devices.getDevice(pair.first);
            if (dev) {
                std::unique_lock<std::mutex> lock(dev->nodeHandleMutex);
                NetworkEntryType entry = dev->latestNetworkEntries[receiverDeviceType];
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
        }

        for (const auto &deviceName : model->possibleDevices)
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

void Controller::estimateModelLatency(PipelineModel *currModel)
{
    if (!currModel) return;
    
    std::string deviceName = currModel->device;
    // We assume datasource and sink models have no latency
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos)
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
    if (currModel->name.find("yolo") == std::string::npos)
    {
        currModel->batchSize = 12;
    }
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
    if (!currModel) return;
    
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos)
    {
        currModel->expectedTransferLatency = 0;
        return;
    }
    currModel->expectedTransferLatency = 0;
    
    if (currModel->name.find("yolo") != std::string::npos)
    {
        uint8_t numUpstreams = 0;
        if (auto task = currModel->task.lock()) {
            for (auto &datasource : task->tk_pipelineModels) {
                if (datasource->name.find("datasource") == std::string::npos) {
                    continue;
                }
                currModel->expectedTransferLatency += currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(datasource->device, currModel->device)].p95TransferDuration;
                numUpstreams++;
            }
        }
        if (numUpstreams > 0) {
            currModel->expectedTransferLatency /= numUpstreams;
        }
        return;
    }

    if (!currModel->upstreams.empty()) {
        if (auto upNode = currModel->upstreams[0].targetNode.lock()) {
            currModel->expectedTransferLatency = currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(upNode->device, currModel->device)].p95TransferDuration;
        }
    }
}

/**
 * @brief DFS-style recursively estimate the latency of a pipeline from source to sink
 *
 * @param pipeline provides all information about the pipeline needed for scheduling
 * @param currModel
 */
void Controller::estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency)
{
    if (!currModel) return;
    
    // estimateModelLatency(currModel, currModel->device);

    // Update the expected latency to reach the current model
    // In case a model has multiple upstreams, the expected latency to reach the model is the maximum of the expected latency
    // to reach from each upstream.
    if (currModel->name.find("datasource") != std::string::npos)
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

    std::vector<PipelineEdge> downstreams = currModel->downstreams;
    for (const auto &d : downstreams)
    {
        if (auto dnNode = d.targetNode.lock()) {
            // Pass the raw pointer to the recursive function
            estimatePipelineLatency(dnNode.get(), currModel->expectedStart2HereLatency);
        }
    }

    if (currModel->downstreams.size() == 0)
    {
        return;
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

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel* toBeMergedModel, const std::string& device) {
    if (!mergedModel || !toBeMergedModel) return false;

    // If the devices are different, or they don't match the target device, we do not merge
    if (mergedModel->device != toBeMergedModel->device ||
        toBeMergedModel->merged || mergedModel->device != device || toBeMergedModel->device != device) {
        return false;
    }

    // Datasources themselves shouldn't be merged into single nodes
    if (mergedModel->name.find("datasource") != std::string::npos ||
        mergedModel->name.find("dsrc") != std::string::npos) {
        return false;
    }

    // Accumulate data sources so the merged model knows all the streams it's handling
    if (!toBeMergedModel->datasourceName.empty()) {
        // Prevent duplicates
        if (std::find(mergedModel->datasourceName.begin(), mergedModel->datasourceName.end(), toBeMergedModel->datasourceName[0]) == mergedModel->datasourceName.end()) {
            mergedModel->datasourceName.push_back(toBeMergedModel->datasourceName[0]);
        }
    }

    return true;
}

std::shared_ptr<TaskHandle> Controller::mergePipelines(const std::string& taskName) {
    auto unscheduledTasks = ctrl_savedUnscheduledPipelines.getMap();
    auto mergedPipeline = std::make_shared<TaskHandle>();
    bool found = false;

    // Registry to hold unique Merged Models mapped by Signature (ModelType + DeviceName)
    std::unordered_map<std::string, std::shared_ptr<PipelineModel>> registry;

    /**
     * @brief Block 1: Merging models to create the merged models and populate the registry.
     * During this pass, we also mark original models as merged and not to be run,
     * effectively retiring them in favor of the new merged models that will be scheduled in the next block.
     * 
     */
    for (const auto& [originalTaskName, task] : unscheduledTasks) {
        if (originalTaskName.find(taskName) == std::string::npos) {
            continue;
        }

        if (!found) {
            found = true;
            *mergedPipeline = *task; 
            mergedPipeline->tk_pipelineModels.clear(); 
            mergedPipeline->tk_name = taskName;
            mergedPipeline->tk_src_device = taskName;
            mergedPipeline->tk_source = taskName;
        }

        for (const auto& model : task->tk_pipelineModels) {
            std::string sig;
            
            // Datasources remain isolated per camera
            if (model->name.find("datasource") != std::string::npos) {
                sig = originalTaskName + "_datasource_" + model->device;
            } else {
                // Server models group by exact variant (e.g., yolov5s640_server)
                std::string baseName = splitString(model->name, "_").back();
                sig = baseName + "_" + model->device;
            }

            // If the signature is not in the registry, we create a new merged model for it.
            // We wipe the upstream/downstream edges because we will reconstruct them in the next pass.
            // We also mark the merged model as "merged" for clarity,
            if (registry.find(sig) == registry.end()) {
                auto newMergedModel = std::make_shared<PipelineModel>(*model);
                newMergedModel->upstreams.clear();   // Wipe original edges
                newMergedModel->downstreams.clear(); // Wipe original edges
                newMergedModel->merged = true;
                
                // Normalize names for clarity
                if (model->name.find("datasource") != std::string::npos) {
                    newMergedModel->name = originalTaskName + "_datasource";
                } else {
                    newMergedModel->name = splitString(model->name, "_").back();
                }
                
                registry[sig] = newMergedModel;
            } else {
                // Node exists. Just accumulate data source trackers.
                mergeModels(registry[sig].get(), model.get(), model->device);
            }
            
            // Mark the old model as retired
            model->merged = true;
            model->toBeRun = false;
        }
    }

    if (!found) return nullptr;

    /****************************************************************************************************************************************/

    /**
     * @brief Block 2: Edge Reconstruction & Stream Multiplexing
     * After we have established the Merged Models and merged their profiles, we need to reconstruct the DAG by reconnecting the Merged-Models
     * according to the original graph structure.
     * During this process, we also need to multiplex the data streams on the edges to ensure that the merged models receive all necessary data.
     * 
     * @param unscheduledTasks 
     */
    for (const auto& [originalTaskName, task] : unscheduledTasks) {
        if (originalTaskName.find(taskName) == std::string::npos) continue;

        std::string streamId = task->tk_source; // The unique stream driving this path

        for (const auto& targetU : task->tk_pipelineModels) {
            std::string sigU;
            if (targetU->name.find("datasource") != std::string::npos) {
                sigU = originalTaskName + "_datasource_" + targetU->device;
            } else {
                sigU = splitString(targetU->name, "_").back() + "_" + targetU->device;
            }
            auto mergedU = registry[sigU];

            for (const auto& downEdge : targetU->downstreams) {
                if (auto targetV = downEdge.targetNode.lock()) {
                    if (targetV->name == targetU->name) continue; 

                    std::string sigV;
                    if (targetV->name.find("datasource") != std::string::npos) {
                        sigV = originalTaskName + "_datasource_" + targetV->device;
                    } else {
                        sigV = splitString(targetV->name, "_").back() + "_" + targetV->device;
                    }
                    auto mergedV = registry[sigV];

                    // --- Route Downstreams (U -> V) ---
                    bool edgeExists = false;
                    for (auto& mergedDownEdge : mergedU->downstreams) {
                        if (mergedDownEdge.targetNode.lock() == mergedV && mergedDownEdge.classOfInterest == downEdge.classOfInterest) {
                            mergedDownEdge.streamNames.insert(streamId); // Edge exists, append stream!
                            edgeExists = true;
                            break;
                        }
                    }
                    if (!edgeExists) {
                        // Create brand new edge using new PipelineEdge struct
                        mergedU->downstreams.push_back(PipelineEdge{mergedV, downEdge.classOfInterest, {streamId}});
                    }

                    // --- Route Upstreams (V <- U) ---
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
     * @brief Block 3: After the merged pipeline is constructed, we can add it to the merged pipelines registry and return it for scheduling.
     * 
     */

    for (auto& [sig, mergedModel] : registry) {
        mergedModel->toBeRun = true;
        mergedModel->task = mergedPipeline; // Link back to the parent TaskHandle
        mergedPipeline->tk_pipelineModels.push_back(mergedModel);
    }

    mergedPipeline->tk_name = taskName.substr(0, taskName.length());
    mergedPipeline->tk_src_device = mergedPipeline->tk_name;
    mergedPipeline->tk_source  = mergedPipeline->tk_name;

    return mergedPipeline;
}

void Controller::mergePipelines() {
    std::vector<std::string> toMerge = {"traffic", "people"}; // or getPipelineNames()

    for (const auto &taskName : toMerge) {
        std::shared_ptr<TaskHandle> mergedPipeline = mergePipelines(taskName);
        if (!mergedPipeline) continue;

        // The fully multiplexed DAG is now ready for scheduling logic
        ctrl_mergedPipelines.addTask(mergedPipeline->tk_name, mergedPipeline);
    }
}

void Controller::Scheduling()
{
    // Map network messages to handler functions
    api_handlers = {
        {MSG_TYPE[START_TASK], std::bind(&Controller::HandleStartTask, this, std::placeholders::_1)}
    };

    std::map<std::string, std::shared_ptr<ClientProfilesJF>> clientProfilesCSJF = {
        {"people", std::make_shared<ClientProfilesJF>()},
        {"traffic", std::make_shared<ClientProfilesJF>()}
    };
    std::map<std::string, std::shared_ptr<ModelProfilesJF>> modelProfilesCSJF = {
        {"people", std::make_shared<ModelProfilesJF>()},
        {"traffic", std::make_shared<ModelProfilesJF>()}
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
         * @brief Block 4: Global Merge & Apply (Jellyfish DP)
         * If triggered by an event OR the periodic timer, we run the global algorithm.
         */
        if ((triggered_by_event || periodic_update_needed) && isPipelineInitialised) 
        {
            ctrl_unscheduledPipelines = {};
            auto untrimmedTaskList = ctrl_savedUnscheduledPipelines.getMap();
            auto deviceList = devices.getMap();

            std::unordered_set<std::string> taskTypes;

            std::cout << "===================== before ==========================" << std::endl;
            for (auto &[task_name, task] : untrimmedTaskList)
            {
                // Task names format tasktype{number} e.g. people1, people2, traffic1, traffic2, etc. We extract the task type by removing the trailing number.
                std::string type = task_name.substr(0, task_name.find_last_of("0123456789"));   
                taskTypes.insert(type);
                auto pipes = task->tk_pipelineModels;
                for (auto &pipe : pipes)
                {
                    std::unique_lock<std::mutex> pipe_lock(pipe->pipelineModelMutex);
                    std::cout << pipe->name << ", ";
                    pipe_lock.unlock();
                }
                std::cout << "end" << std::endl;
            }
            std::cout << "======================================================" << std::endl;

            mergePipelines();
            ctrl_unscheduledPipelines = ctrl_mergedPipelines;
            ctrl_mergedPipelines = {};

            
            std::map<std::string, uint64_t> pipelineSLOs;

            for (auto &taskType : taskTypes)
            {
                auto task = ctrl_unscheduledPipelines.getTask(taskType).get();
                if (!task) {
                    spdlog::get("container_agent")->error("Task {} not found in unscheduled pipelines during scheduling.", taskType);
                    continue;
                }
                queryingProfiles(task);
                std::cout << "debugging query profile" << std::endl;
                for (auto &model : task->tk_pipelineModels)
                {
                    std::unique_lock<std::mutex> lock(model->pipelineModelMutex);
                    std::cout << "model name: " << model->name << ", " << model->device << std::endl;
                    for (auto &downstream : model->downstreams)
                    {
                        // FIX: Lock downstream targetNode before accessing name
                        if (auto dnNode = downstream.targetNode.lock()) {
                            std::cout << "downstream: " << dnNode->name << ", " << downstream.classOfInterest << std::endl;
                        }
                    }
                    lock.unlock();
                }
                std::cout << "debugging query profile end" << std::endl;
                for (auto &model : task->tk_pipelineModels)
                {
                    if (model->name.find("datasource") != std::string::npos)
                    {
                        continue;
                    }
                    model->name = taskType + "_" + model->name;
                    if (model->name.find("yolo") != std::string::npos)
                    {
                        continue;
                    }
                }
                // Assigned dummy value for yolo batch size
                task->tk_pipelineModels.at(1)->batchSize = 1;
                for (auto &model : task->tk_pipelineModels) {
                    // FIX: Extract raw pointer via .get()
                    estimateModelNetworkLatency(model.get());
                    estimateModelLatency(model.get());
                }
                for (auto &model : task->tk_pipelineModels)
                {
                    if (model->name.find("datasource") == std::string::npos)
                    {
                        continue;
                    }
                    // FIX: Extract raw pointer via .get()
                    estimatePipelineLatency(model.get(), 0);
                }
                pipelineSLOs[taskType] = task->tk_slo;
                task->tk_slo -= task->tk_pipelineModels.back()->expectedStart2HereLatency;

                for (auto &model : task->tk_pipelineModels)
                {
                    if (model->name.find("datasource") == std::string::npos)
                    {
                        continue;
                    }
                    // FIX: Extract raw pointer via .get()
                    estimateTimeBudgetLeft(model.get());
                }
            }

            std::cout << "===================== after ==========================" << std::endl;
            for (auto task : ctrl_unscheduledPipelines.getList())
            {
                auto pipes = task->tk_pipelineModels;
                for (auto &pipe : pipes)
                {
                    std::unique_lock<std::mutex> pipe_lock(pipe->pipelineModelMutex);
                    std::cout << pipe->name << ", ";
                    pipe_lock.unlock();
                }
                std::cout << "end" << std::endl;
            }
            std::cout << "======================================================" << std::endl;

            // FIX: Securely re-instantiate fresh profiles instead of clearing private internal vectors
            for (auto &task_name : taskTypes)
            {
                clientProfilesCSJF[task_name] = std::make_shared<ClientProfilesJF>();
                modelProfilesCSJF[task_name] = std::make_shared<ModelProfilesJF>();
            }

            // collect all information
            for (auto &[task_name, task] : ctrl_unscheduledPipelines.getMap())
            {
                std::cout << "task name: " << task_name << std::endl;
                for (auto model : task->tk_pipelineModels)
                {
                    std::unique_lock<std::mutex> lock_pipeline_model(model->pipelineModelMutex);
                    std::cout << "model name: " << model->name << std::endl;
                    if (model->name.find("datasource") == std::string::npos)
                    {
                        model->device = model->possibleDevices[0];
                        model->deviceTypeName = "server";
                        model->deviceAgent = deviceList[model->possibleDevices[0]];
                    }
                    else
                    {
                        // FIX: Safe type extraction from shared_ptr
                        model->deviceTypeName = getDeviceTypeName(deviceList[model->device]->type);
                        model->deviceAgent = deviceList[model->device];
                    }
                    lock_pipeline_model.unlock();
                }
            }

            int count = 0;
            for (auto task : ctrl_unscheduledPipelines.getList())
            {
                if (count == 2)
                {
                    break;
                }
                for (auto model : task->tk_pipelineModels)
                {

                    std::unique_lock<std::mutex> lock_pipeline_model(model->pipelineModelMutex);
                    if (model->name.find("yolo") != std::string::npos)
                    {
                        // parse name
                        std::size_t pos1 = model->name.find("_");
                        std::string model_name = model->name.substr(pos1 + 1);

                        std::string containerName = model_name + "_" + model->deviceTypeName;
                        std::cout << "model name in finding: " << model_name << std::endl;

                        BatchInferProfileListType batch_proilfes = queryBatchInferLatency(
                            *ctrl_metricsServerConn.get(),
                            ctrl_experimentName,
                            ctrl_systemName,
                            task->tk_name,
                            task->tk_source,
                            model->device,
                            model->deviceTypeName,
                            ctrl_containerLib[containerName].modelName,
                            // TODO: Change back once we have profilings in every fps
                            //ctrl_systemFPS
                            15);


                        // parse the resolution of the model
                        std::size_t pos = model_name.find("_");
                        std::string yolo = model_name.substr(0, pos);
                        int rs;
                        try
                        {
                            size_t pos_idx;
                            rs = std::stoi(yolo.substr(model_name.length() - 3, 3), &pos_idx);
                            if (pos_idx != 3)
                            {
                                throw std::invalid_argument("yolov5n, set the default resolution 640");
                            }
                            yolo = yolo.substr(0, yolo.length() - 3);
                        }
                        catch (const std::invalid_argument &e)
                        {
                            rs = 640;
                        }
                        int width = rs;
                        int height = rs;
                        for (auto &[batch_size, profile] : batch_proilfes)
                        {
                            modelProfilesCSJF[task->tk_name]->add(model_name, ACC_LEVEL_MAP.at(yolo + std::to_string(rs)), batch_size, profile.p95inferLat, width, height, model);
                        }
                    }
                    else if (model->name.find("datasource") != std::string::npos)
                    {
                        // collect information of data source
                        auto downstream = model->downstreams.front();
                        // FIX: Safely lock downstream target node and agent weak_ptr
                        if (auto dnNode = downstream.targetNode.lock()) {
                            auto downstream_device = dnNode->deviceTypeName;
                            if (auto agent = model->deviceAgent.lock()) {
                                auto entry = agent->latestNetworkEntries.at(downstream_device);
                                clientProfilesCSJF[task->tk_name]->add(model->name, task->tk_slo, ctrl_systemFPS, model, task->tk_name, task->tk_source, entry);
                            }
                        }
                    }

                    lock_pipeline_model.unlock();
                }
                count++;
            }

            // debugging
            std::cout << "========================= Task Info =========================" << std::endl;
            for (auto &task_name : taskTypes)
            {
                auto client_profiles = clientProfilesCSJF[task_name];
                auto model_profiles = modelProfilesCSJF[task_name];
                
                // FIX: Safely fetch the copied infos via thread-safe getters
                auto c_infos = client_profiles->getInfos();
                auto m_infos = model_profiles->getInfos();
                
                std::cout << task_name << ", n client: " << c_infos.size() << std::endl;
                std::cout << task_name << ", n model: " << m_infos.size() << std::endl;
                for (auto &client_info : c_infos)
                {
                    if (auto mod = client_info.model.lock()) {
                        std::cout << "client name: " << client_info.name << ", " << client_info.task_name << ", client device: " << mod->device << std::endl;
                    }
                }
                for (auto &model_info : m_infos)
                {
                    std::cout << model_info.second.front().name << std::endl;
                }
            }
            std::cout << "=============================================================" << std::endl;

            for (auto &task_name : taskTypes)
            {
                auto client_profiles_jf = clientProfilesCSJF[task_name];
                auto model_profiles_jf = modelProfilesCSJF[task_name];
                
                // FIX: Extract vector directly so we can mutate the latency internally
                std::vector<ClientInfoJF> clients_to_schedule = client_profiles_jf->getInfos();

                for (auto &client_info : clients_to_schedule)
                {
                    auto client_model = client_info.model.lock();
                    if (!client_model) continue;
                    
                    std::unique_lock<std::mutex> client_lock(client_model->pipelineModelMutex);
                    auto client_device = client_model->device;
                    auto client_device_type = client_model->deviceTypeName;
                    
                    // downstream yolo information
                    auto downstream = client_model->downstreams.front().targetNode.lock();
                    if (!downstream) continue;
                    
                    std::unique_lock<std::mutex> model_lock(downstream->pipelineModelMutex);
                    std::string model_name = downstream->name;
                    size_t pos = model_name.find("_");
                    model_name = model_name.substr(pos + 1);
                    std::string model_device = downstream->device;
                    std::string model_device_typename = downstream->deviceTypeName;
                    std::string containerName = model_name + "_" + model_device_typename;
                    model_lock.unlock();
                    client_lock.unlock();

                    std::cout << "before query Network" << std::endl;
                    std::cout << "name of the client model: " << containerName << std::endl;
                    std::cout << "downstream name: " << downstream->name << std::endl;

                    NetworkProfile network_proflie = queryNetworkProfile(
                        *ctrl_metricsServerConn,
                        ctrl_experimentName,
                        ctrl_systemName,
                        client_info.task_name,
                        client_info.task_source,
                        ctrl_containerLib[containerName].taskName,
                        ctrl_containerLib[containerName].modelName,
                        client_device,
                        client_device_type,
                        model_device,
                        model_device_typename,
                        client_info.network_entry);
                    auto lat = network_proflie.p95TransferDuration;
                    client_info.set_transmission_latency(lat);
                }

                // start scheduling
                // FIX: Using mutated vector and dereferenced model_profiles pointer
                auto mappings = Jlf::mapClient(clients_to_schedule, *model_profiles_jf);

                // clean the upstream of not selected yolo
                auto model_infos_map = model_profiles_jf->getInfos();
                std::vector<std::shared_ptr<PipelineModel>> not_selected_yolos;
                std::vector<std::shared_ptr<PipelineModel>> selected_yolos;
                for (auto &mapping : mappings)
                {
                    auto model_info = std::get<0>(mapping);
                    auto yolo_pipeliemodel = model_infos_map[model_info][0].model.lock();
                    if (yolo_pipeliemodel) selected_yolos.push_back(yolo_pipeliemodel);
                }
                for (auto &yolo : model_infos_map)
                {
                    auto yolo_pipeliemodel = yolo.second.front().model.lock();
                    if (yolo_pipeliemodel && std::find(selected_yolos.begin(), selected_yolos.end(), yolo_pipeliemodel) == selected_yolos.end())
                    {
                        not_selected_yolos.push_back(yolo_pipeliemodel);
                    }
                }

                for (auto &not_select_yolo : not_selected_yolos)
                {
                    not_select_yolo->upstreams.clear();
                    not_select_yolo->batchSize = 1;
                }

                for (auto &mapping : mappings)
                {
                    // retrieve the mapping for one model and its paired clients
                    auto model_info = std::get<0>(mapping);
                    auto selected_clients = std::get<1>(mapping);
                    int batch_size = std::get<2>(mapping);

                    // find the PipelineModel* of that model
                    ModelInfoJF m = model_infos_map[model_info][0];
                    for (auto &model : model_infos_map[model_info])
                    {
                        if (model.batch_size == batch_size)
                        {
                            m = model;
                            break;
                        }
                    }
                    
                    // clear the upstream of that model safely
                    if (auto mod = m.model.lock()) {
                        std::unique_lock<std::mutex> model_lock(mod->pipelineModelMutex);
                        mod->upstreams.clear();
                        mod->batchSize = m.batch_size;

                        // adjust downstream, upstream and resolution
                        for (auto &client : selected_clients)
                        {
                            if (auto cmod = client.model.lock()) {
                                // FIX: Insert properly structured PipelineEdges
                                mod->upstreams.push_back({cmod, 1, {}});
                                std::unique_lock<std::mutex> client_lock(cmod->pipelineModelMutex);
                                cmod->downstreams.clear();
                                cmod->downstreams.push_back({mod, 1, {}});

                                // retrieve new resolution
                                int width = m.width;
                                int height = m.height;
                                mod->batchSize = batch_size;

                                std::vector<int> rs = {width, height};
                                cmod->dimensions = rs;
                                client_lock.unlock();
                            }
                        }
                        model_lock.unlock();
                    }
                }

                std::cout << "SCHEDULING END" << std::endl;

                // for debugging mappings
                std::cout << "================================ Mapping ===================================" << std::endl;
                for (auto &mapping : mappings)
                {
                    auto model_info = std::get<0>(mapping);
                    std::cout << "Model name: " << std::get<0>(model_info) << ", acc: " << std::get<1>(model_info) << ", batch_size: " << std::endl;
                    auto clients_info = std::get<1>(mapping);
                    for (auto &client : clients_info)
                    {
                        if (auto mod = client.model.lock()) {
                            std::cout << "Client name: " << client.name << ", budget: " << client.budget << ", lat: " << client.transmission_latency << ", client device: " << mod->device << std::endl;
                        }
                    }
                    std::cout << "Batch size: " << std::get<2>(mapping) << std::endl;
                    std::cout <<"-----------------------------------" << std::endl;
                }
                std::cout << "============================= End Mapping =================================" << std::endl;

                // for debugging
                std::cout << "============================== check all clients downstream ==================================" << task_name << std::endl;
                for (auto &client : clients_to_schedule)
                {
                    auto p = client.model.lock();
                    if (!p) continue;
                    std::unique_lock<std::mutex> lock(p->pipelineModelMutex);
                    std::cout << "datasource name: " << p->datasourceName.size();
                    for (auto &ds : p->downstreams)
                    {
                        if (auto dsNode = ds.targetNode.lock()) {
                            std::cout << ", ds name: " << dsNode->name << std::endl;
                        }
                    }
                    lock.unlock();
                }
                std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;

                std::cout << "================================= check all models upstream ==================================" << task_name << std::endl;
                for (auto &model : model_infos_map)
                {
                    auto p = model.second.front().model.lock();
                    if (!p) continue;
                    std::unique_lock<std::mutex> lock(p->pipelineModelMutex);
                    std::cout << "model name: " << p->name;
                    for (auto us : p->upstreams)
                    {
                        if (auto usNode = us.targetNode.lock()) {
                            std::cout << ", us name: " << usNode->name << ", address of client: " << usNode.get() << "; ";
                        }
                    }
                    std::cout << std::endl;
                    lock.unlock();
                }
                std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
                std::cout << "================================= check all models downstream ==================================" << task_name << std::endl;
                for (auto &model : model_infos_map)
                {
                    auto p = model.second.front().model.lock();
                    if (!p) continue;
                    std::unique_lock<std::mutex> lock(p->pipelineModelMutex);
                    std::cout << "model name: " << p->name;
                    for (auto ds : p->downstreams)
                    {
                        if (auto dsNode = ds.targetNode.lock()) {
                            std::cout << ", ds name: " << dsNode->name << ", address of client: " << dsNode.get() << "; ";
                        }
                    }
                    std::cout << std::endl;
                    lock.unlock();
                }
                std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << 
                std::endl;
            }

            for (auto &task : ctrl_unscheduledPipelines.getMap())
            {
                for (auto &model : task.second->tk_pipelineModels)
                {
                    if (model->name.find("datasource") != std::string::npos || model->upstreams.size() == 0)
                    {
                        continue;
                    }
                    if (model->name.find("datasource") != std::string::npos ||
                        model->name.find("sink") != std::string::npos ||
                        model->name.find("yolo") != std::string::npos)
                    {
                        continue;
                    }
                    model->batchSize = 12;
                }

                for (auto &model : task.second->tk_pipelineModels)
                {
                    estimateModelLatency(model.get());
                }
                for (auto &model : task.second->tk_pipelineModels)
                {
                    if (model->name.find("datasource") == std::string::npos)
                    {
                        continue;
                    }
                    estimateTimeBudgetLeft(model.get());
                }
                for (auto &model : task.second->tk_pipelineModels)
                {
                    if (model->name.find("datasource") != std::string::npos ||
                            model->name.find("sink") != std::string::npos ||
                            model->name.find("yolo") != std::string::npos)
                    {
                        model->numReplicas = 1;
                        continue;
                    }
                    // set specific number of replicas for each downstream
                    model->numReplicas = 4;
                }
                task.second->tk_slo = pipelineSLOs[task.second->tk_name];
                
            }

            // IMPORTANT DO-NOT-DELETE: This backup keeps the current scheduled pipelines objects as well as their pipelineModel objects alive until ApplyScheduling finishes,
            auto backupScheduledPipelines = ctrl_scheduledPipelines.getMap();
            ctrl_scheduledPipelines = ctrl_unscheduledPipelines;
            ApplyScheduling();

            // Reset the global periodic timer
            ctrl_nextSchedulingTime = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_controlTimings.schedulingIntervalSec);
        }

        schedulingSW.stop();
        if (startTime == std::chrono::system_clock::time_point()) startTime = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(std::chrono::system_clock::now() - startTime).count() > ctrl_runtime) {
            running = false;
            break;
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
    if (arrival_rate == 0)
    {
        return 0;
    }
    float rho = arrival_rate / preprocess_rate;
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t)(averageQueueLength / arrival_rate * 1000000);
}

// ----------------------------------------------------------------------------------------------------------------
//                                             implementations
// ----------------------------------------------------------------------------------------------------------------
ModelInfoJF::ModelInfoJF(int bs, float il, int w, int h, const std::string& n, float acc, std::weak_ptr<PipelineModel> m)
{
    batch_size = bs;

    // the inference_latency is us
    inference_latency = il;

    // throughput is req/s
    // now the time stamp is us, and the gcd of all throughputs is 10, maybe need change to ease the dp table
    throughput = (int(bs / (il * 1e-6)) / 10) * 10; // round it to be devidisble by 10 for better dp computing
    width = w;
    height = h;
    name = n;
    accuracy = acc;
    model = m;
}

ClientInfoJF::ClientInfoJF(const std::string& _name, float _budget, int _req_rate,
                           std::weak_ptr<PipelineModel> _model, const std::string& _task_name, const std::string& _task_source,
                           NetworkEntryType _network_entry)
{
    name = _name;
    budget = _budget;
    req_rate = _req_rate;
    model = _model;
    task_name = _task_name;
    task_source = _task_source;
    transmission_latency = -1;
    network_entry = _network_entry;
}

// -------------------------------------------------------------------------------------------
//                               implementation of ModelProfilesJF
// -------------------------------------------------------------------------------------------

/**
 * @brief add profiled information of model
 *
 * @param model_type
 * @param accuracy
 * @param batch_size
 * @param inference_latency
 * @param throughput
 */
void ModelProfilesJF::add(const std::string& name, float accuracy, int batch_size, float inference_latency, int width, int height, std::weak_ptr<PipelineModel> m)
{
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::tuple<std::string, float>{name, accuracy};
    ModelInfoJF value(batch_size, inference_latency, width, height, name, accuracy, m);
    auto it = std::find(infos[key].begin(), infos[key].end(), value);
    // record the model which is a new model
    if (it == infos[key].end())
    {
        infos[key].push_back(value);
    }
}

void ModelProfilesJF::add(const ModelInfoJF &model_info)
{
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::tuple<std::string, float>{model_info.name, model_info.accuracy};
    infos[key].push_back(model_info);
}

void ModelProfilesJF::debugging()
{
    auto infos_copy = getInfos();
    std::cout << "======================ModelProfiles Debugging=======================" << std::endl;
    for (auto it = infos_copy.begin(); it != infos_copy.end(); ++it)
    {
        auto key = it->first;
        auto profilings = it->second;
        std::cout << "Model: " << std::get<0>(key) << ", Accuracy: " << std::get<1>(key) << std::endl;
        for (const auto &model_info : profilings)
        {
            std::cout << "batch size: " << model_info.batch_size << ", latency: " << model_info.inference_latency
                      << ", width: " << model_info.width << ", height: " << model_info.height << ", throughput: " << model_info.throughput << std::endl;
        }
    }
    std::cout << "======================ModelProfiles Debugging End=======================" << std::endl;
}

// -------------------------------------------------------------------------------------------
//                               implementation of ClientProfilesJF
// -------------------------------------------------------------------------------------------

/**
 * @brief sort the budget which equals (SLO - networking time)
 *
 * @param clients
 */
void ClientProfilesJF::sortBudgetDescending(std::vector<ClientInfoJF> &clients)
{
    std::sort(clients.begin(), clients.end(),
              [](const ClientInfoJF &a, const ClientInfoJF &b)
              {
                  return a.budget - a.transmission_latency > b.budget - b.transmission_latency;
              });
}

void ClientProfilesJF::add(const std::string &name, float budget, int req_rate,
                           std::weak_ptr<PipelineModel> model, const std::string& task_name, const std::string& task_source,
                           NetworkEntryType network_entry)
{
    std::lock_guard<std::mutex> lock(mtx);
    infos.push_back(ClientInfoJF(name, budget, req_rate, model, task_name, task_source, network_entry));
}

void ClientProfilesJF::debugging()
{
    auto infos_copy = getInfos();
    std::cout << "===================================ClientProfiles Debugging==========================" << std::endl;
    for (const auto &client_info : infos_copy)
    {
        std::cout << "Unique id: " << client_info.name << ", buget: " << client_info.budget << ", req_rate: " << client_info.req_rate << std::endl;
    }
    std::cout << "===================================ClientProfiles Debugging End==========================" << std::endl;
}

// -------------------------------------------------------------------------------------------
//                               implementation of scheduling algorithms
// -------------------------------------------------------------------------------------------

std::vector<ClientInfoJF> Jlf::findOptimalClients(const std::vector<ModelInfoJF> &models,
                                             std::vector<ClientInfoJF> &clients)
{
    // sort clients
    ClientProfilesJF::sortBudgetDescending(clients);
    std::tuple<int, int> best_cell;
    int best_value = 0;

    // dp
    auto [max_batch_size, max_index] = Jlf::findMaxBatchSize(models, clients[0], 16);
    std::cout << "max batch size: " << max_batch_size << " and index: " << max_index << std::endl;
    assert(max_batch_size > 0);

    // construct the dp matrix
    int rows = clients.size() + 1;
    int h = 10; // assume gcd of all clients' req rate
    // find max throughput
    int max_throughput = 0;
    for (auto &model : models)
    {
        if (model.throughput > max_throughput)
        {
            max_throughput = model.throughput;
        }
    }
    // init matrix
    int cols = max_throughput / h + 1;
    std::vector<std::vector<int>> dp_mat(rows, std::vector<int>(cols, 0));
    // iterating
    for (unsigned int client_index = 1; client_index <= clients.size(); client_index++)
    {
        auto &client = clients[client_index - 1];
        auto result = Jlf::findMaxBatchSize(models, client, max_batch_size);
        max_batch_size = std::get<0>(result);
        max_index = std::get<1>(result);
        if (max_batch_size <= 0)
        {
            break;
        }
        int cols_upperbound = int(models[max_index].throughput / h);
        int lambda_i = client.req_rate;
        int v_i = client.req_rate;
        for (int k = 1; k <= cols_upperbound; k++)
        {

            int w_k = k * h;
            if (lambda_i <= w_k)
            {
                int k_prime = (w_k - lambda_i) / h;
                int v = v_i + dp_mat[client_index - 1][k_prime];
                assert(v >= 0 && k_prime >= 0);
                if (v > dp_mat[client_index - 1][k])
                {
                    dp_mat[client_index][k] = v;
                }
                else
                {
                    dp_mat[client_index][k] = dp_mat[client_index - 1][k];
                }
                if (v > best_value)
                {
                    best_cell = std::make_tuple(client_index, k);
                    best_value = v;
                }
            }
            else
            {
                dp_mat[client_index][k] = dp_mat[client_index - 1][k];
            }
        }
    }

    // perform backtracing from (row, col)
    // using dp_mat, best_cell, best_value

    std::vector<ClientInfoJF> selected_clients;
    auto [row, col] = best_cell;
    int w = dp_mat[row][col];
    while (row > 0 && col > 0)
    {
        if (dp_mat[row][col] == dp_mat[row - 1][col])
        {
            row = row - 1;
        }
        else
        {
            auto c = clients[row - 1];
            int w_i = c.req_rate;
            row = row - 1;
            col = int((w - w_i) / h);
            w = col * h;
            selected_clients.push_back(c);
        }
    }

    return selected_clients;
}

/**
 * @brief client dnn mapping algorithm strictly following the paper jellyfish's Algo1
 *
 * @param clients
 * @param model_profiles
 * @return a vector of [ (model_name, accuracy), vec[clients], batch_size ]
 */
std::vector<
    std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
// FIX: Matched signature to accept the clients vector directly, just like the header
Jlf::mapClient(std::vector<ClientInfoJF> &clients, ModelProfilesJF &model_profiles)
{

    std::vector<
        std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
        mappings;
        
    // FIX: Retrieve the thread-safe copy of the map instead of public access
    auto model_infos_map = model_profiles.getInfos();

    int map_size = model_infos_map.size();
    int key_index = 0;
    
    // FIX: Iterate over the retrieved map copy
    for (auto it = model_infos_map.begin(); it != model_infos_map.end(); ++it)
    {
        key_index++;
        auto selected_clients = Jlf::findOptimalClients(it->second, clients);

        // tradeoff:
        // assign all left clients to the last available model
        if (key_index == map_size)
        {

            if (clients.size() == 0)
            {
                break;
            }

            selected_clients = clients;
            clients.clear();
            assert(clients.size() == 0);
        }

        int batch_size = Jlf::check_and_assign(it->second, selected_clients);
        mappings.push_back(
            std::make_tuple(it->first, selected_clients, batch_size));
        Jlf::differenceClients(clients, selected_clients);
        if (clients.size() == 0)
        {
            break;
        }
    }
    return mappings;
}

/**
 * @brief find the max available batch size for the associated clients of
 * corresponding model
 *
 * @param model
 * @param selected_clients
 * @return int
 */
int Jlf::check_and_assign(std::vector<ModelInfoJF> &model,
                     std::vector<ClientInfoJF> &selected_clients)
{
    int total_req_rate = 0;
    // sum all selected req rate
    for (auto &client : selected_clients)
    {
        total_req_rate += client.req_rate;
    }
    int max_batch_size = 1;

    for (auto &model_info : model)
    {
        if (model_info.throughput > total_req_rate &&
            max_batch_size < model_info.batch_size)
        {
            // NOTE: in our case, our model's throughput is too high, so
            // in the experiment, it seems to always assign the small batch size.
            // In Jellyfish, their model throughput is at most 80, and they just choose the batch size
            // which could simply match the total request. The code here follows that.
            max_batch_size = model_info.batch_size;
            break;
        }
    }
    return max_batch_size;
}

// ====================== helper functions implementation ============================

/**
 * @brief find the maximum batch size for the client, the model vector is the set of model only different in batch size
 *
 * @param models
 * @param budget
 * @return max_batch_size, index
 */
std::tuple<int, int> Jlf::findMaxBatchSize(const std::vector<ModelInfoJF> &models,
                                      const ClientInfoJF &client, int max_available_batch_size)
{
    int max_batch_size = 2;
    int index = 0;
    int max_index = 1;
    for (const auto &model : models)
    {
        if (model.inference_latency * 2.0 < client.budget - client.transmission_latency &&
            model.batch_size > max_batch_size && model.batch_size <= max_available_batch_size)
        {
            max_batch_size = model.batch_size;
            max_index = index;
        }
        index++;
    }
    return std::make_tuple(max_batch_size, max_index);
}

/**
 * @brief remove the selected clients
 *
 * @param src
 * @param diff
 */
void Jlf::differenceClients(std::vector<ClientInfoJF> &src,
                       const std::vector<ClientInfoJF> &diff)
{
    auto is_in_diff = [&diff](const ClientInfoJF &client)
    {
        return std::find(diff.begin(), diff.end(), client) != diff.end();
    };
    src.erase(std::remove_if(src.begin(), src.end(), is_in_diff), src.end());
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

// -------------------------------------------------------------------------------------------
//                                  end of implementations
// -------------------------------------------------------------------------------------------

void Controller::colocationTemporalScheduling() {} // Dummy Method for Compiler