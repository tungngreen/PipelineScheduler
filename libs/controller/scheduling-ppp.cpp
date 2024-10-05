#include "scheduling-ppp.h"

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

    if (node.type == SystemDeviceType::Server) {
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


// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

// ==================================================================Scheduling==================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

void Controller::queryingProfiles(TaskHandle *task) {

    std::map<std::string, NodeHandle*> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;

    //if task->tk_name ends with number, remove it
    std::string sanitizedTaskName = task->tk_name;
    if (task->tk_name.back() >= '0' && task->tk_name.back() <= '9') {
        sanitizedTaskName = task->tk_name.substr(0, task->tk_name.size() - 1);
    }
    std::string source = task->tk_source.substr(task->tk_source.find_last_of('/') + 1);

    for (auto model: *pipelineModels) {
        if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos) {
            continue;
        }
        model->deviceTypeName = getDeviceTypeName(deviceList.at(model->device)->type);
        std::vector<std::string> upstreamPossibleDeviceList = model->upstreams.front().first->possibleDevices;
        std::vector<std::string> thisPossibleDeviceList = model->possibleDevices;
        std::vector<std::pair<std::string, std::string>> possibleDevicePairList;
        for (const auto &deviceName : upstreamPossibleDeviceList) {
            for (const auto &deviceName2 : thisPossibleDeviceList) {
                if (deviceName == "server" && deviceName2 != deviceName) {
                    continue;
                }
                possibleDevicePairList.push_back({deviceName, deviceName2});
            }
        }
        std::string containerName = model->name + "_" + model->deviceTypeName;
        if (!task->tk_newlyAdded) {
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

        for (const auto &pair : possibleDevicePairList) {
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

        for (const auto &deviceName : model->possibleDevices) {
            std::string deviceTypeName = getDeviceTypeName(deviceList.at(deviceName)->type);
            containerName = model->name + "_" + deviceTypeName;
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
            model->processProfiles[deviceTypeName].maxBatchSize = std::max_element(
                profile.batchInfer.begin(),
                profile.batchInfer.end(),
                [](const auto &p1, const auto &p2) {
                    return p1.first < p2.first;
                }
            )->first;
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

void Controller::Scheduling() {
    while (running) {
        Stopwatch schedulingSW;
        schedulingSW.start();
        // Check if it is the next scheduling period
        ctrl_controlTimings.currSchedulingTime = std::chrono::system_clock::now();
        // Only rescale periodically and if its not right before the next scheduling period, because scheduling() will take care of rescaling as well.
        if (ctrl_controlTimings.currSchedulingTime >= ctrl_controlTimings.nextRescalingTime &&
            (ctrl_controlTimings.currSchedulingTime < ctrl_controlTimings.nextSchedulingTime &&
             std::chrono::duration_cast<std::chrono::seconds>(ctrl_controlTimings.nextSchedulingTime - ctrl_controlTimings.currSchedulingTime).count() >= 30)) {
            Rescaling();
            schedulingSW.stop();
            ctrl_controlTimings.nextRescalingTime = ctrl_controlTimings.currSchedulingTime + std::chrono::seconds(ctrl_controlTimings.rescalingIntervalSec -
                                                                                                                  schedulingSW.elapsed_microseconds() / 1000000);
            uint64_t sleepTime = std::chrono::duration_cast<TimePrecisionType>(ctrl_controlTimings.nextRescalingTime - std::chrono::system_clock::now()).count();
            std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
            continue;
        }

        if (ctrl_controlTimings.currSchedulingTime < ctrl_nextSchedulingTime) {
            std::this_thread::sleep_for(
                std::chrono::seconds(
                    std::chrono::duration_cast<std::chrono::seconds>(ctrl_nextSchedulingTime - ctrl_controlTimings.currSchedulingTime).count()
                )
            );
            continue;
        }

        ctrl_unscheduledPipelines = ctrl_savedUnscheduledPipelines;
        auto taskList = ctrl_unscheduledPipelines.getMap();
        if (!isPipelineInitialised) {
            continue;
        }
        ctrl_controlTimings.nextSchedulingTime = ctrl_controlTimings.currSchedulingTime + std::chrono::seconds(ctrl_controlTimings.schedulingIntervalSec);


        for (auto &[taskName, taskHandle]: taskList) {
            queryingProfiles(taskHandle);
            crossDeviceWorkloadDistributor(taskHandle, taskHandle->tk_slo / 2);
            shiftModelToEdge(taskHandle->tk_pipelineModels, taskHandle->tk_pipelineModels.front(), taskHandle->tk_slo, taskHandle->tk_pipelineModels.front()->device);
            for (auto &model: taskHandle->tk_pipelineModels) {
                model->name = taskName + "_" + model->name;
            }
            estimateModelTiming(taskHandle->tk_pipelineModels.front(), 0);
            taskHandle->tk_newlyAdded = false;
        }

        mergePipelines();

        auto mergedTasks = ctrl_mergedPipelines.getMap();
        for (auto &[taskName, taskHandle]: mergedTasks) {
            for (auto &model: taskHandle->tk_pipelineModels) {
                for (auto i = 0; i < model->numReplicas; i++) {
                    model->cudaDevices.push_back(0); // Add dummy cuda device value to create a container manifestation
                    // model->manifestations.push_back(TranslateToContainer(model, devices.list[model->device], 0));
                    // model->manifestations.back()->task = taskHandle;
                }
                if (model->name.find("sink") != std::string::npos) {
                    model->device = "sink";
                }
            }
        }

        estimatePipelineTiming();
        ctrl_scheduledPipelines = ctrl_mergedPipelines;
        ApplyScheduling();
        ctrl_controlTimings.nextRescalingTime = ctrl_controlTimings.currSchedulingTime + std::chrono::seconds(ctrl_controlTimings.rescalingIntervalSec);
        schedulingSW.stop();

        ClockType nextTime = std::min(ctrl_controlTimings.nextSchedulingTime, ctrl_controlTimings.nextRescalingTime);
        uint64_t sleepTime = std::chrono::duration_cast<TimePrecisionType>(nextTime - std::chrono::system_clock::now()).count();
        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
    }
}

void Controller::ScaleUp(PipelineModel *model, uint8_t numIncReps) {
    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - 
                                   model->lastScaleTime).count() < ctrl_controlTimings.scaleUpIntervalThresholdSec) {
        spdlog::get("container_agent")->info("The model {0:s} has been scaled up recently."
                                             "Skipping the scaling up process to avoid uncessary waste.", model->name);
        return;
    }
    model->numReplicas += numIncReps;
    std::vector<ContainerHandle*> currContainers = model->task->tk_subTasks[model->name];
    uint16_t numCurrContainers = currContainers.size();
    model->lastScaleTime = std::chrono::system_clock::now();
    for (uint16_t i = numCurrContainers; i < model->numReplicas; i++) {
        ContainerHandle *newContainer = TranslateToContainer(model, devices.getDevice(model->device), i);
        if (newContainer == nullptr) {
            spdlog::get("container_agent")->error("Failed to create container for model {0:s} of pipeline {1:s}", model->name, model->task->tk_name);
            continue;
        }
        newContainer->pipelineModel = model;
        for (auto &downstream : model->downstreams) {
            for (auto &downstreamContainer : downstream.first->task->tk_subTasks[downstream.first->name]) {
                downstreamContainer->upstreams.push_back(newContainer);
                newContainer->downstreams.push_back(downstreamContainer);
            }
        }
        for (auto &upstream : model->upstreams) {
            for (auto &upstreamContainer : upstream.first->task->tk_subTasks[upstream.first->name]) {
                upstreamContainer->downstreams.push_back(newContainer);
                newContainer->upstreams.push_back(upstreamContainer);
            }
        }
        containerColocationTemporalScheduling(newContainer);
        containers.addContainer(newContainer->name, newContainer);
        StartContainer(newContainer);
        for (auto &upstream : model->upstreams) {
            for (auto &upstreamContainer : upstream.first->task->tk_subTasks[upstream.first->name]) {
                AdjustUpstream(newContainer->recv_port, upstreamContainer, newContainer->device_agent,
                               model->name, AdjustUpstreamMode::Add);
            }
        }
    }
    model->lastScaleTime = std::chrono::system_clock::now();
}

void Controller::ScaleDown(PipelineModel *model, uint8_t numDecReps) {

    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - 
                                   model->lastScaleTime).count() < ctrl_controlTimings.scaleDownIntervalThresholdSec) {
        spdlog::get("container_agent")->info("The model {0:s} has been scaled up recently."
                                             "Skipping the scaling down process to avoid THRASHING.", model->name);
        return;
    }
    model->numReplicas -= numDecReps;
    std::vector<ContainerHandle*> currContainers = model->task->tk_subTasks[model->name];
    uint16_t numCurrContainers = currContainers.size();
    for (uint16_t i = model->numReplicas; i < numCurrContainers; i++) {
        for (auto &upstream : model->upstreams) {
            for (auto &upstreamContainer : upstream.first->task->tk_subTasks[upstream.first->name]) {
                AdjustUpstream(currContainers[i]->recv_port, upstreamContainer, currContainers[i]->device_agent,
                               model->name, AdjustUpstreamMode::Remove);
            }
        }
        StopContainer(currContainers[i], currContainers[i]->device_agent);
        auto reclaimed = reclaimGPUPortion(currContainers[i]->executionPortion);
        if (!reclaimed) {
            spdlog::get("container_agent")->error("Failed to reclaim portion for container {0:s}", currContainers[i]->name);
            return;
        }
        containers.removeContainer(currContainers[i]->name);
    }
    model->lastScaleTime = std::chrono::system_clock::now();
}

void Controller::Rescaling() {
    auto taskList = ctrl_scheduledPipelines.getMap();
    // std::mt19937 gen(100);
    // std::uniform_int_distribution<int> dist(0, 2);


    for (auto &[taskName, taskHandle]: taskList) {
        std::string source = taskHandle->tk_source.substr(taskHandle->tk_source.find_last_of('/') + 1);
        for (auto &model: taskHandle->tk_pipelineModels) {
            if (model->name.find("datasource") != std::string::npos || model->name.find("dsrc") != std::string::npos
                || model->name.find("sink") != std::string::npos) {
                continue;
            }
            std::string taskName = splitString(model->name, "_").back();
            auto ratesAndCoeffVars = queryArrivalRateAndCoeffVar(
                *ctrl_metricsServerConn,
                ctrl_experimentName,
                ctrl_systemName,
                taskHandle->tk_name,
                source,
                taskName,
                ctrl_containerLib[taskName + "_" + model->deviceTypeName].modelName,
                // TODO: Change back once we have profilings in every fps
                //ctrl_systemFPS
                15,
                {15, 30, 60}
            );
            model->arrivalProfiles.arrivalRates = ratesAndCoeffVars.first;
            model->arrivalProfiles.coeffVar = ratesAndCoeffVars.second;

            auto candidates = model->task->tk_subTasks[model->name];

            auto numIncReps = incNumReplicas(model);

            // // testing scaling up
            // if (model->device != "server") {
            //     continue;
            // }
            // auto numIncReps = dist(gen);
            // // testing done

            if (numIncReps > 0) {
                ScaleUp(model, numIncReps);
                spdlog::get("container_agent")->info("Rescaling tried increasing number of replicas of model {0:s} of pipeline {1:s} by {2:d}", model->name, taskHandle->tk_name, numIncReps);
                continue;
            }

            // //testing
            // if (numIncReps) {
            //     model->numReplicas -= numIncReps;
            //     ScaleDown(model);
            // }
            // //testing done

            auto numDecReps = decNumReplicas(model);
            if (numDecReps > 0) {
                ScaleDown(model, numDecReps);
                spdlog::get("container_agent")->info("Rescaling tried decreasing number of replicas of model {0:s} of pipeline {1:s} by {2:d}", model->name, taskHandle->tk_name, numDecReps);
            }

        }
    }
    ctrl_pastScheduledPipelines = ctrl_scheduledPipelines;
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
    auto gpu = gpuLane->gpuHandle;

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

    auto gpuHandle = toBeRemovedPortion->lane->gpuHandle;
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

/**
 * @brief colocationTemporalScheduler (CORAL) for container instances
 * 
 * @param container 
 * @return true 
 * @return false 
 */
bool Controller::containerColocationTemporalScheduling(ContainerHandle *container) {
    std::string deviceName = container->device_agent->name;
    auto deviceList = devices.getMap();
    auto portion = findFreePortionForInsertion(deviceList[deviceName]->freeGPUPortions, container);

    if (portion == nullptr) {
        spdlog::get("container_agent")->error("No free portion found for container {0:s}", container->name);
        return false;
    }
    container->executionPortion = portion;
    container->gpuHandle = portion->lane->gpuHandle;
    auto newPortions = insertUsedGPUPortion(deviceList[deviceName]->freeGPUPortions, container, portion);

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
bool Controller::modelColocationTemporalScheduling(PipelineModel *pipelineModel, unsigned int replica_id) {
    if (pipelineModel->gpuScheduled) { return true; }
    if (pipelineModel->name.find("datasource") == std::string::npos &&
        (pipelineModel->name.find("dsrc") == std::string::npos ||
         pipelineModel->name.find("yolov5ndsrc") != std::string::npos) &&
        pipelineModel->name.find("sink") == std::string::npos) {
        for (auto &container : pipelineModel->task->tk_subTasks[pipelineModel->name]) {
            if (container->replica_id == replica_id) {
                container->startTime = pipelineModel->startTime;
                container->endTime = pipelineModel->endTime;
                containerColocationTemporalScheduling(container);
            }
        }
    }
    bool allScheduled = true;
    for (auto downstream : pipelineModel->downstreams) {
        if (!modelColocationTemporalScheduling(downstream.first, replica_id)) allScheduled = false;
    }
    if (!allScheduled) return false;
    if (replica_id >= pipelineModel->numReplicas - 1) {
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
    auto deviceList = devices.getMap();
    for (auto &[deviceName, deviceHandle]: deviceList) {
        initiateGPULanes(*deviceHandle);
    }
    bool process_flag = true;
    unsigned int replica_id = 0;
    while (process_flag) {
        process_flag = false;
        for (auto &[taskName, taskHandle]: ctrl_scheduledPipelines.getMap()) {
            auto front_model = taskHandle->tk_pipelineModels.front();
            if (!front_model->gpuScheduled) {
                process_flag = process_flag || !modelColocationTemporalScheduling(front_model, replica_id);
            }
        }
        replica_id++;
    }
}

bool Controller::mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile, const std::string& device, const std::string& upstreamDevice) {
    mergedProfile.arrivalRates += toBeMergedProfile.arrivalRates;
    auto mergedD2DProfile = &mergedProfile.d2dNetworkProfile;
    auto toBeMergedD2DProfile = &toBeMergedProfile.d2dNetworkProfile;
    float coefficient1 = mergedProfile.arrivalRates / (mergedProfile.arrivalRates + toBeMergedProfile.arrivalRates);
    float coefficient2 = toBeMergedProfile.arrivalRates / (mergedProfile.arrivalRates + toBeMergedProfile.arrivalRates);

    // There should be only 1 pair in the d2dNetworkProfile with key {"merged-...", device}
    D2DNetworkProfile newProfile = {};
    auto mergedPair = mergedProfile.d2dNetworkProfile.key_comp();
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
    float coefficient1 = arrivalRate1 / (arrivalRate1 + arrivalRate2);
    float coefficient2 = arrivalRate2 / (arrivalRate1 + arrivalRate2);
    for (const auto &[deviceName, profile] : toBeMergedProfile) {
        if (deviceName != device) {
            continue;
        }
        auto mergedProfileDevice = &mergedProfile[deviceName];
        auto toBeMergedProfileDevice = &toBeMergedProfile.at(deviceName);

        mergedProfileDevice->p95InputSize = std::max(mergedProfileDevice->p95InputSize, toBeMergedProfileDevice->p95InputSize);
        mergedProfileDevice->p95OutputSize = std::max(mergedProfileDevice->p95OutputSize, toBeMergedProfileDevice->p95OutputSize);

        auto mergedBatchInfer = &mergedProfileDevice->batchInfer;
        // auto toBeMergedBatchInfer = &toBeMergedProfileDevice->batchInfer;

        for (const auto &[batchSize, p] : toBeMergedProfileDevice->batchInfer) {
            mergedBatchInfer->at(batchSize).p95inferLat =
                mergedBatchInfer->at(batchSize).p95inferLat * coefficient1 + p.p95inferLat * coefficient2;
            mergedBatchInfer->at(batchSize).p95prepLat =
                mergedBatchInfer->at(batchSize).p95prepLat * coefficient1 + p.p95prepLat * coefficient2;
            mergedBatchInfer->at(batchSize).p95postLat =
                mergedBatchInfer->at(batchSize).p95postLat * coefficient1 + p.p95postLat * coefficient2;
            mergedBatchInfer->at(batchSize).cpuUtil = std::max(mergedBatchInfer->at(batchSize).cpuUtil, p.cpuUtil);
            mergedBatchInfer->at(batchSize).gpuUtil = std::max(mergedBatchInfer->at(batchSize).gpuUtil, p.gpuUtil);
            mergedBatchInfer->at(batchSize).memUsage = std::max(mergedBatchInfer->at(batchSize).memUsage, p.memUsage);
            mergedBatchInfer->at(batchSize).rssMemUsage = std::max(mergedBatchInfer->at(batchSize).rssMemUsage, p.rssMemUsage);
            mergedBatchInfer->at(batchSize).gpuMemUsage = std::max(mergedBatchInfer->at(batchSize).gpuMemUsage, p.gpuMemUsage);
        }

    }
    return true;
}

bool Controller::mergeModels(PipelineModel *mergedModel, PipelineModel* toBeMergedModel, const std::string& device) {
    // If the merged model is empty, we should just copy the model to be merged
    if (mergedModel->numReplicas == 255) {
        *mergedModel = *toBeMergedModel;
        std::string upStreamDevice = toBeMergedModel->upstreams.front().first->device;
        for (auto &[pair, profile] : mergedModel->arrivalProfiles.d2dNetworkProfile) {
            if (pair.second != device || pair.first != upStreamDevice) {
                continue;
            }
            mergedModel->arrivalProfiles.d2dNetworkProfile[{"merged-" + pair.first, device}] = profile;
        }
        std::vector<decltype(mergedModel->arrivalProfiles.d2dNetworkProfile.begin())> keysToErase;
        for (auto &[pair, profile] : mergedModel->arrivalProfiles.d2dNetworkProfile) {
            if (pair.first.find("merged") != std::string::npos) {
                continue;
            }
            keysToErase.push_back(mergedModel->arrivalProfiles.d2dNetworkProfile.find(pair));
        }
        for (auto key : keysToErase) {
            mergedModel->arrivalProfiles.d2dNetworkProfile.erase(key);
        }
        return true;
    }
    // If the devices are different, we should not merge the models
    if (mergedModel->device != toBeMergedModel->device ||
        toBeMergedModel->merged || mergedModel->device != device || toBeMergedModel->device != device) {
        return false;
    }


    float rate1 = mergedModel->arrivalProfiles.arrivalRates;
    float rate2 = toBeMergedModel->arrivalProfiles.arrivalRates;

    mergeArrivalProfiles(mergedModel->arrivalProfiles, toBeMergedModel->arrivalProfiles, device, toBeMergedModel->upstreams.front().first->device);
    mergeProcessProfiles(mergedModel->processProfiles, rate1, toBeMergedModel->processProfiles, rate2, device);

    return true;
}

TaskHandle* Controller::mergePipelines(const std::string& taskName) {

    auto unscheduledTasks = ctrl_unscheduledPipelines.getMap();

    PipelineType tk_type;
    for (const auto& task : unscheduledTasks) {
        if (task.first.find(taskName) == std::string::npos) {
            continue;
        }
        tk_type = task.second->tk_type;
        break;
    }


    TaskHandle* mergedPipeline = new TaskHandle{};
    bool found;
    for (const auto& task : unscheduledTasks) {
        if (task.first.find(taskName) == std::string::npos) {
            continue;
        }
        found = true;
        // Initialize the merged pipeline with one of the added tasks in the task type
        *mergedPipeline = *task.second;
    }
    if (!found) {
        spdlog::info("No task with type {0:s} has been added", taskName);
        return nullptr;
    }
    uint16_t numModels = mergedPipeline->tk_pipelineModels.size();

    for (auto &model : mergedPipeline->tk_pipelineModels) {
        model->toBeRun = true;
    }

    // Loop through all the models in the pipeline and merge models of the same type
    for (uint16_t i = 0; i < numModels; i++) {
        // Find this model in all the scheduled tasks
        for (const auto& task : unscheduledTasks) {
            // This task is the one we used to initialize the merged pipeline, we should not merge it
            if (task.second->tk_name == mergedPipeline->tk_name) {
                continue;
            }
            // Cannot merge if model is they do not belong to the same task group
            if (task.first.find(taskName) == std::string::npos) {
                continue;
            }
            // If model is not scheduled to be run on the server, we should not merge it.
            // However, the model is still
            if (task.second->tk_pipelineModels[i]->device != "server") {
                mergedPipeline->tk_pipelineModels.emplace_back(new PipelineModel(*task.second->tk_pipelineModels[i]));
                task.second->tk_pipelineModels[i]->merged = true;
                task.second->tk_pipelineModels[i]->toBeRun = false;
                mergedPipeline->tk_pipelineModels.back()->toBeRun = false;
                continue;
            }
            // We attempt to merge to model i of this unscheduled task into the model i of the merged pipeline
            bool merged = mergeModels(mergedPipeline->tk_pipelineModels[i], task.second->tk_pipelineModels.at(i), "server");
            task.second->tk_pipelineModels.at(i)->merged = true;
            task.second->tk_pipelineModels.at(i)->toBeRun = false;
        }
        // auto numIncReps = incNumReplicas(mergedPipeline.tk_pipelineModels[i]);
        // mergedPipeline.tk_pipelineModels[i]->numReplicas += numIncReps;
        // auto deviceList = devices.getMap();
        // for (auto j = 0; j < mergedPipeline.tk_pipelineModels[i]->numReplicas; j++) {
        //     mergedPipeline.tk_pipelineModels[i]->manifestations.emplace_back(new ContainerHandle{});
        //     mergedPipeline.tk_pipelineModels[i]->manifestations.back()->task = &mergedPipeline;
        //     mergedPipeline.tk_pipelineModels[i]->manifestations.back()->device_agent = deviceList.at(mergedPipeline.tk_pipelineModels[i]->device);
        // }
    }
    for (auto &model : mergedPipeline->tk_pipelineModels) {
        // If toBeRun is true means the model is not a newly added one, there's no need to modify its up and downstreams
        if (model->toBeRun) {
            continue;
        }
        for (auto &oldDownstream : model->downstreams) {
            std::string oldDnstreamModelName = splitString(oldDownstream.first->name, "_").back();
            for (auto &newDownstream : mergedPipeline->tk_pipelineModels) {
                std::string newDownstreamModelName = splitString(newDownstream->name, "_").back();
                if (oldDnstreamModelName == newDownstreamModelName &&
                    oldDownstream.first->device == newDownstream->device &&
                    oldDownstream.first != newDownstream) {
                    model->downstreams.emplace_back(std::make_pair(newDownstream, oldDownstream.second));
                    newDownstream->upstreams.emplace_back(std::make_pair(model, oldDownstream.second));
                    break;
                }
            }
        }
        model->toBeRun = true;
    }
    for (auto &model : mergedPipeline->tk_pipelineModels) {
        for (auto it = model->downstreams.begin(); it != model->downstreams.end();) {
            if (!(it->first->toBeRun)) {
                it = model->downstreams.erase(it);
            } else {
                it++;
            }
        }
        for (auto it = model->upstreams.begin(); it != model->upstreams.end();) {
            if (!(it->first->toBeRun)) {
                it = model->upstreams.erase(it);
            } else {
                it++;
            }
        }
    }
    for (auto &model : mergedPipeline->tk_pipelineModels) {
        if (model->device != "server") {
            continue;
        }
        auto names = splitString(model->name, "_");
        model->name = taskName + "_" + names[1];
    }
    mergedPipeline->tk_name = taskName.substr(0, taskName.length());
    mergedPipeline->tk_src_device = mergedPipeline->tk_name;
    mergedPipeline->tk_source  = mergedPipeline->tk_name;
    return mergedPipeline;
}

void Controller::mergePipelines() {
    std::vector<std::string> toMerge = {"traffic", "people"};
    TaskHandle* mergedPipeline;

    for (const auto &taskName : toMerge) {
        mergedPipeline = mergePipelines(taskName);
        if (mergedPipeline == nullptr) {
            continue;
        }
        // Increase the number of replicas to avoid bottlenecks
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            // only models scheduled to run on the server are merged and to be considered
            if (mergedModel->device != "server") {
                continue;
            }
            auto numIncReps = incNumReplicas(mergedModel);
            mergedModel->lastScaleTime = std::chrono::system_clock::now();
            mergedModel->numReplicas += numIncReps;
            estimateModelLatency(mergedModel);
        }
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
            if (mergedModel->name.find("datasource") == std::string::npos) {
                continue;
            }
            estimatePipelineLatency(mergedModel, mergedModel->expectedStart2HereLatency);
        }
        for (auto &mergedModel : mergedPipeline->tk_pipelineModels) {
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
    if (currModel->name.find("sink") != std::string::npos) {
        return;
    }
    if (currModel->name.find("datasource") != std::string::npos) {
        if (currModel->device != edgeDevice) {
            spdlog::get("container_agent")->warn("Edge device {0:s} is not identical to the datasource device {1:s}", edgeDevice, currModel->device);
            return;
        }
    }

    if (currModel->device == edgeDevice) {
        for (auto d: currModel->downstreams) {
            shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
        }
        return;
    }

    // If the edge device is not in the list of possible devices, we should not consider it
    if (std::find(currModel->possibleDevices.begin(), currModel->possibleDevices.end(), edgeDevice) == currModel->possibleDevices.end()) {
        return;
    }

    std::string deviceTypeName = getDeviceTypeName(devices.getDevice(edgeDevice)->type);

    uint32_t inputSize = currModel->processProfiles.at(deviceTypeName).p95InputSize;
    uint32_t outputSize = currModel->processProfiles.at(deviceTypeName).p95OutputSize;
    bool shifted = false;

    if (inputSize * 0.6 > outputSize) {
        PipelineModel oldModel = *currModel;
        currModel->device = edgeDevice;
        currModel->deviceTypeName = deviceTypeName;
        for (auto downstream : currModel->downstreams) {
            estimateModelLatency(downstream.first);
        }
        currModel->batchSize = 1;
        if (currModel->batchSize <= currModel->processProfiles[deviceTypeName].maxBatchSize) {
            estimateModelLatency(currModel);
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
            uint64_t expectedE2ELatency = pipeline.back()->expectedStart2HereLatency;
            if (expectedE2ELatency > slo) {
                *currModel = oldModel;
                // break;
            }
            oldModel = *currModel;
            shifted = true;
            // currModel->batchSize *= 2;
        }
        // if after shifting the model to the edge device, the pipeline still meets the SLO, we should keep it

        // However, if the pipeline does not meet the SLO, we should shift reverse the model back to the server
        if (!shifted) {
            *currModel = oldModel;
            estimateModelLatency(currModel);
            for (auto &downstream : currModel->downstreams) {
                estimateModelLatency(downstream.first);
            }
            estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
            return;
        }
    }
    estimateModelLatency(currModel);
    estimatePipelineLatency(currModel, currModel->expectedStart2HereLatency);
    // Shift downstream models to the edge device
    for (auto d: currModel->downstreams) {
        shiftModelToEdge(pipeline, d.first, slo, edgeDevice);
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

    PipelineModelListType *models = &(task->tk_pipelineModels);

    for (auto m: *models) {
        m->batchSize = 1;
        m->numReplicas = 1;

        estimateModelLatency(m);
        if (m->name.find("datasource") == std::string::npos) {
            m->device = "server";
            m->deviceTypeName = "server";
        }
    }


    // DFS-style recursively estimate the latency of a pipeline from source to sink
    // The first model should be the datasource
    estimatePipelineLatency(models->front(), 0);

    uint64_t expectedE2ELatency = models->back()->expectedStart2HereLatency;

    if (slo < expectedE2ELatency) {
        spdlog::info("SLO is too low for the pipeline to meet. Expected E2E latency: {0:d}, SLO: {1:d}", expectedE2ELatency, slo);
    }

    // Increase number of replicas to avoid bottlenecks
    for (auto m: *models) {
        auto numIncReplicas = incNumReplicas(m);
        m->lastScaleTime = std::chrono::system_clock::now();
        m->numReplicas += numIncReplicas;
    }
    estimatePipelineLatency(models->front(), 0);

    // Find near-optimal batch sizes
    auto foundBest = true;
    while (foundBest) {
        foundBest = false;
        uint64_t bestCost = models->back()->estimatedStart2HereCost;
        for (auto m: *models) {
            if (m->name.find("datasource") != std::string::npos || m->name.find("sink") != std::string::npos) {
                continue;
            }
            BatchSizeType oldBatchsize = m->batchSize;
            m->batchSize *= 2;
            if (m->batchSize > m->processProfiles[m->deviceTypeName].maxBatchSize) {
                m->batchSize = oldBatchsize;
                continue;
            }
            estimateModelLatency(m);
            estimatePipelineLatency(m, m->expectedStart2HereLatency);
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
                    estimateModelLatency(m);
                    estimatePipelineLatency(m, m->expectedStart2HereLatency);
                    continue;
                }
                // If increasing the batch size meets the SLO, we can try decreasing the number of replicas
                auto numDecReplicas = decNumReplicas(m);
                m->lastScaleTime = std::chrono::system_clock::now();
                m->numReplicas -= numDecReplicas;
            } else {
                m->batchSize = oldBatchsize;
                estimateModelLatency(m);
                estimatePipelineLatency(m, m->expectedStart2HereLatency);
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
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos) {
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
void Controller::estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency) {
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
        for (auto &upstream : currModel->upstreams) {
            currModel->estimatedStart2HereCost += upstream.first->estimatedStart2HereCost;
            currModel->expectedStart2HereLatency = std::max(
                currModel->expectedStart2HereLatency,
                upstream.first->expectedStart2HereLatency + currModel->expectedMaxProcessLatency + currModel->expectedTransferLatency +
                currModel->expectedQueueingLatency
            );
        }

    }

    std::vector<std::pair<PipelineModel *, int>> downstreams = currModel->downstreams;
    for (const auto &d: downstreams) {
        estimatePipelineLatency(d.first, currModel->expectedStart2HereLatency);
    }

    if (currModel->downstreams.empty()) {
        return;
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

void Controller::estimateModelTiming(PipelineModel *currModel, const uint64_t start2HereLatency) {

    if (currModel->name.find("datasource") != std::string::npos) {
        currModel->startTime = 0;
        currModel->endTime = 0;
        // if (currModel->name.find("sink") != std::string::npos) {

    }
    else if (currModel->name.find("sink") != std::string::npos) {
        currModel->startTime = 0;
        currModel->endTime = 0;
        for (auto &upstream : currModel->upstreams) {
            currModel->localDutyCycle = std::max(currModel->localDutyCycle, upstream.first->endTime);
        }
        return;
    } else {
        auto batchSize = currModel->batchSize;
        auto profile = currModel->processProfiles.at(currModel->deviceTypeName);

        uint64_t maxStartTime = std::max(currModel->startTime, start2HereLatency);
        for (auto &upstream : currModel->upstreams) {
            if (upstream.first->device != currModel->device) {
                continue;
            }
            // TODO: Add in-device transfer latency
            maxStartTime = std::max(maxStartTime, upstream.first->endTime);
        }
        currModel->startTime = maxStartTime;
        currModel->endTime = currModel->startTime + currModel->expectedMaxProcessLatency;
    }

    uint64_t maxDnstreamDutyCycle = currModel->localDutyCycle;
    for (auto &downstream : currModel->downstreams) {
        if (downstream.first->device != currModel->device &&
            downstream.first->name.find("sink") == std::string::npos) {
            estimateModelTiming(downstream.first, 0);
            maxDnstreamDutyCycle = std::max(maxDnstreamDutyCycle, start2HereLatency + currModel->expectedMaxProcessLatency);
            continue;
        }
        estimateModelTiming(downstream.first, start2HereLatency + currModel->expectedMaxProcessLatency);
        maxDnstreamDutyCycle = std::max(maxDnstreamDutyCycle, downstream.first->localDutyCycle);
    }
    currModel->localDutyCycle = maxDnstreamDutyCycle;
}

void Controller::estimatePipelineTiming() {
    auto tasks = ctrl_mergedPipelines.getMap();
    for (auto &[taskName, task]: tasks) {
        for (auto &model: task->tk_pipelineModels) {
            // If the model has already been estimated, we should not estimate it again
            if (model->endTime != 0 && model->startTime != 0) {
                continue;
            }
            // TODO
            estimateModelTiming(model, 0);
        }
        uint64_t localDutyCycle;
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
            estimateTimeBudgetLeft(model);
        }
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
    if (model->name.find("datasource") != std::string::npos || model->name.find("sink") != std::string::npos) {
        return 0;
    }
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1000000.f / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat
                                 + profile.batchInfer.at(model->batchSize).p95postLat);
    float indiPreprocessRate = 1000000.f / profile.batchInfer.at(model->batchSize).p95prepLat;
    float processRate = indiProcessRate * numReplicas;
    float preprocessRate = indiPreprocessRate * numReplicas;
    while ((processRate * 0.8 < model->arrivalProfiles.arrivalRates ||
           preprocessRate * 0.95 < model->arrivalProfiles.arrivalRates)) {
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
    uint8_t numReplicas = model->numReplicas;
    std::string deviceTypeName = model->deviceTypeName;
    ModelProfile profile = model->processProfiles.at(deviceTypeName);
    uint64_t inferenceLatency = profile.batchInfer.at(model->batchSize).p95inferLat;
    float indiProcessRate = 1000000.f / (inferenceLatency + profile.batchInfer.at(model->batchSize).p95prepLat
                                         + profile.batchInfer.at(model->batchSize).p95postLat);
    float indiPreprocessRate = 1000000.f / profile.batchInfer.at(model->batchSize).p95prepLat;
    float processRate, preprocessRate;
    while (numReplicas > 1) {
        numReplicas--;
        processRate = indiProcessRate * numReplicas;
        preprocessRate = indiPreprocessRate * numReplicas;
        // If the number of replicas is no longer enough to meet the arrival rate, we should not decrease the number of replicas anymore.
        if ((processRate * 0.7 < model->arrivalProfiles.arrivalRates ||
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
uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate) {
    float rho = arrival_rate / preprocess_rate;
    if (rho > 1) {
        return 999999999;
    }
    float numQueriesInSystem = rho / (1 - rho);
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t) (averageQueueLength / arrival_rate * 1000000);
}

// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================
// ============================================================================================================================================