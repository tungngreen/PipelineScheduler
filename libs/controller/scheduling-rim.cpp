#include "scheduling-rim.h"

void Controller::queryingProfiles(TaskHandle *task) {

    std::map<std::string, NodeHandle*> deviceList = devices.getMap();

    auto pipelineModels = &task->tk_pipelineModels;
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
                15
            );
        }
        for (const auto deviceName : model->possibleDevices) {
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
                15
            );


            //----------------------------------------
            // std::cout << ctrl_experimentName << std::endl;
            // std::cout << ctrl_systemName << std::endl;
            // std::cout << task->tk_name << std::endl;
            // std::cout << task->tk_source << std::endl;
            // std::cout << deviceName << std::endl;
            // std::cout << deviceTypeName << std::endl;
            // std::cout << ctrl_containerLib[containerName].modelName << std::endl;
            // std::cout << ctrl_systemFPS << std::endl;

            //----------------------------------------

            model->processProfiles[deviceTypeName] = profile;
            model->processProfiles[deviceTypeName].maxBatchSize = std::max_element(
                profile.batchInfer.begin(),
                profile.batchInfer.end(),
                [](const auto &p1, const auto &p2) {
                    return p1.first < p2.first;
                }
            )->first;
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

void Controller::Scheduling() {
    while (running) {
        // Check if it is the next scheduling period
        Stopwatch schedulingSW;
        schedulingSW.start();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - ctrl_nextSchedulingTime)
                .count() < 10)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        ctrl_unscheduledPipelines = ctrl_savedUnscheduledPipelines;
        auto localTaskList = ctrl_unscheduledPipelines.getMap();
        if (!isPipelineInitialised) {
            continue;
        }

        std::vector<std::string> processedTasks;

        for (auto &[task_name, task] : localTaskList)
        {
            try {
                std::cout << "Processing task: " << task_name << std::endl;
                
                queryingProfiles(task);

                try {
                    std::cout << "START SCHEDULING" << std::endl;
                    Rim::rim_action(task, devices);
                    std::cout << "SCHEDULING END" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Exception in rim_action: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Unknown exception in rim_action" << std::endl;
                }

                processedTasks.push_back(task_name);
            } catch (const std::exception& e) {
                std::cerr << "Exception caught while processing task " << task_name << ": " << e.what() << std::endl;
            }
        }

        for (auto &[task_name, task] : localTaskList) {
            for (auto &model : task->tk_pipelineModels) {
                if (model->name == "datasource" || model->name == "sink") {
                    continue;
                }
                if (model->name.find("yolov5") != std::string::npos) {
                    model->batchSize = ctrl_initialBatchSizes["yolov5"];
                } else if (model->device == "server") {
                    model->batchSize = ctrl_initialBatchSizes["server"];
                } else {
                    model->batchSize = ctrl_initialBatchSizes["edge"];
                }
                estimateModelNetworkLatency(model);
                estimateModelLatency(model);
            }
            for (auto &model : task->tk_pipelineModels) {
                if (model->name.find("datasource") == std::string::npos)
                {
                    continue;
                }
                estimateTimeBudgetLeft(model);
            }
        }

        ctrl_scheduledPipelines = ctrl_unscheduledPipelines;
        ApplyScheduling();
        schedulingSW.stop();
        ctrl_nextSchedulingTime = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_schedulingIntervalSec);
        std::this_thread::sleep_for(TimePrecisionType((ctrl_schedulingIntervalSec + 1) * 1000000 - schedulingSW.elapsed_microseconds()));
        // break;
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
uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate) {
    float rho = arrival_rate / preprocess_rate;
    float averageQueueLength = rho * rho / (1 - rho);
    return (uint64_t) (averageQueueLength / arrival_rate * 1000000);
}


void Controller::colocationTemporalScheduling() {} // Dummy Method for Compiler