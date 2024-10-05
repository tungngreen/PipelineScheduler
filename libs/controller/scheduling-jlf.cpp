#include "scheduling-jlf.h"

void Controller::queryingProfiles(TaskHandle *task)
{

    std::map<std::string, NodeHandle *> deviceList = devices.getMap();

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
            upstreamPossibleDeviceList = model->upstreams.front().first->possibleDevices;
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

void Controller::estimateModelLatency(PipelineModel *currModel)
{
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
    if (currModel->name.find("datasource") != std::string::npos || currModel->name.find("sink") != std::string::npos)
    {
        currModel->expectedTransferLatency = 0;
        return;
    }
    currModel->expectedTransferLatency = 0;
    if (currModel->name.find("yolo") != std::string::npos)
    {
        uint8_t numUpstreams = 0;
        for (PipelineModel *&datasource : currModel->task->tk_pipelineModels) {
            if (datasource->name.find("datasource") == std::string::npos) {
                continue;
            }
            currModel->expectedTransferLatency += currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(datasource->device, currModel->device)].p95TransferDuration;
            numUpstreams++;
        }
        currModel->expectedTransferLatency /= numUpstreams;
        return;
    }

    currModel->expectedTransferLatency = currModel->arrivalProfiles.d2dNetworkProfile[std::make_pair(currModel->upstreams[0].first->device, currModel->device)].p95TransferDuration;
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
    std::map<std::string, ClientProfilesJF> clientProfilesCSJF = {{"people", ClientProfilesJF()}, {"traffic", ClientProfilesJF()}};
    std::map<std::string, ModelProfilesJF> modelProfilesCSJF = {{"people", ModelProfilesJF()}, {"traffic", ModelProfilesJF()}};

    while (running)
    {
        // Check if it is the next scheduling period
        Stopwatch schedulingSW;
        schedulingSW.start();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - ctrl_nextSchedulingTime).count() < 10) {
            continue;
        }

        ctrl_unscheduledPipelines = {};
        auto untrimmedTaskList = ctrl_savedUnscheduledPipelines.getMap();
        auto deviceList = devices.getMap();
        if (!isPipelineInitialised)
        {
            continue;
        }

        std::cout << "===================== before ==========================" << std::endl;
        for (auto &[task_name, task] : untrimmedTaskList)
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

        std::vector<std::string> taskTypes = {"traffic", "people"};
        for (auto taskType : taskTypes)
        {
            for (auto &[taskName, taskHandle] : untrimmedTaskList)
            {
                if (taskName.find(taskType) == std::string::npos || taskName == taskType)
                {
                    continue;
                }
                std::map<std::string, TaskHandle*> taskList = ctrl_unscheduledPipelines.getMap();
                if (taskList.find(taskType) == taskList.end())
                {
                    ctrl_unscheduledPipelines.addTask(taskType, new TaskHandle(*taskHandle));
                    ctrl_unscheduledPipelines.getTask(taskType)->tk_pipelineModels.front()->name = taskName + "_datasource";
                    ctrl_unscheduledPipelines.getTask(taskType)->tk_name = taskType;
                }
                else
                {
                    ctrl_unscheduledPipelines.getTask(taskType)->tk_pipelineModels.emplace_back(new PipelineModel(*taskHandle->tk_pipelineModels.front()));
                    ctrl_unscheduledPipelines.getTask(taskType)->tk_pipelineModels.back()->downstreams = {};
                    auto yolo = ctrl_unscheduledPipelines.getTask(taskType)->tk_pipelineModels.front()->downstreams.front().first;
                    ctrl_unscheduledPipelines.getTask(taskType)->tk_pipelineModels.back()->downstreams.emplace_back(std::make_pair(yolo, -1));
                    ctrl_unscheduledPipelines.getTask(taskType)->tk_pipelineModels.back()->name = taskName + "_datasource";
                }
            }
        }
        
        std::map<std::string, uint64_t> pipelineSLOs;

        for (auto &taskType : taskTypes)
        {
            auto task = ctrl_unscheduledPipelines.getTask(taskType);
            queryingProfiles(task);
            std::cout << "debugging query profile" << std::endl;
            for (auto &model : task->tk_pipelineModels)
            {
                std::unique_lock<std::mutex> lock(model->pipelineModelMutex);
                std::cout << "model name: " << model->name << ", " << model->device << std::endl;
                for (auto &downstream : model->downstreams)
                {
                    std::cout << "downstream: " << downstream.first->name << ", " << downstream.second << std::endl;
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
                estimateModelNetworkLatency(model);
                estimateModelLatency(model);
            }
            for (auto &model : task->tk_pipelineModels)
            {
                if (model->name.find("datasource") == std::string::npos)
                {
                    continue;
                }
                estimatePipelineLatency(model, 0);
            }
            pipelineSLOs[taskType] = task->tk_slo;
            task->tk_slo -= task->tk_pipelineModels.back()->expectedStart2HereLatency;

            for (auto &model : task->tk_pipelineModels)
            {
                if (model->name.find("datasource") == std::string::npos)
                {
                    continue;
                }
                estimateTimeBudgetLeft(model);
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

        // clear all the information
        for (auto &pair : clientProfilesCSJF)
        {
            pair.second.infos.clear();
        }
        for (auto &pair : modelProfilesCSJF)
        {
            pair.second.infos.clear();
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
                    model->deviceTypeName = deviceList[model->device]->type;
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
            // std::unique_lock<std::mutex> lock_task(task->tk_mutex);
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
                        size_t pos;
                        rs = std::stoi(yolo.substr(model_name.length() - 3, 3), &pos);
                        if (pos != 3)
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
                        modelProfilesCSJF[task->tk_name].add(model_name, ACC_LEVEL_MAP.at(yolo + std::to_string(rs)), batch_size, profile.p95inferLat, width, height, model);
                    }
                }
                else if (model->name.find("datasource") != std::string::npos)
                {
                    // collect information of data source
                    auto downstream = model->downstreams.front();
                    auto downstream_device = downstream.first->deviceTypeName;
                    auto entry = model->deviceAgent->latestNetworkEntries.at(downstream_device);

                    clientProfilesCSJF[task->tk_name].add(model->name, task->tk_slo, ctrl_systemFPS, model, task->tk_name, task->tk_source, entry);
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
            std::cout << task_name << ", n client: " << client_profiles.infos.size() << std::endl;
            std::cout << task_name << ", n model: " << model_profiles.infos.size() << std::endl;
            for (auto &client_info : client_profiles.infos)
            {
                std::cout << "client name: " << client_info.name << ", " << client_info.task_name << ", client address: " << client_info.model << ", client device: " << client_info.model->device << std::endl;
            }
            for (auto &model_info : model_profiles.infos)
            {
                std::cout << model_info.second.front().name << std::endl;
            }
        }
        std::cout << "=============================================================" << std::endl;

        for (auto &task_name : taskTypes)
        {
            auto client_profiles_jf = clientProfilesCSJF[task_name];
            auto model_profiles_jf = modelProfilesCSJF[task_name];

            for (auto &client_info : client_profiles_jf.infos)
            {
                auto client_model = client_info.model;
                std::unique_lock<std::mutex> client_lock(client_model->pipelineModelMutex);
                auto client_device = client_model->device;
                auto client_device_type = client_model->deviceTypeName;
                // downstream yolo information
                auto downstream = client_model->downstreams.front().first;
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
                // std::cout << "queried latency is: " << lat << std::endl;
                client_info.set_transmission_latency(lat);
            }

            // start scheduling

            // std::cout << "START SCHEDULING" << std::endl;

            auto mappings = Jlf::mapClient(client_profiles_jf, model_profiles_jf);

            // std::cout << "FINISH STRATEGY COMPUTING" << std::endl;

            // clean the upstream of not selected yolo
            std::vector<PipelineModel *> not_selected_yolos;
            std::vector<PipelineModel *> selected_yolos;
            for (auto &mapping : mappings)
            {
                auto model_info = std::get<0>(mapping);
                auto yolo_pipeliemodel = model_profiles_jf.infos[model_info][0].model;
                selected_yolos.push_back(yolo_pipeliemodel);
            }
            for (auto &yolo : model_profiles_jf.infos)
            {
                auto yolo_pipeliemodel = yolo.second.front().model;
                if (std::find(selected_yolos.begin(), selected_yolos.end(), yolo_pipeliemodel) == selected_yolos.end())
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
                // std::cout << "Selected Mapping batch size: " << batch_size << std::endl;

                // find the PipelineModel* of that model
                ModelInfoJF m = model_profiles_jf.infos[model_info][0];
                for (auto &model : model_profiles_jf.infos[model_info])
                {
                    if (model.batch_size == batch_size)
                    {
                        // note: if occurs core dump, it's possible that there is no matchable pointer
                        // and the p is null
                        m = model;
                        break;
                    }
                }
                // clear the upstream of that model
                std::unique_lock<std::mutex> model_lock(m.model->pipelineModelMutex);
                m.model->upstreams.clear();
                m.model->batchSize = m.batch_size;

                // adjust downstream, upstream and resolution
                for (auto &client : selected_clients)
                {
                    m.model->upstreams.push_back(std::make_pair(client.model, 1));
                    std::unique_lock<std::mutex> client_lock(client.model->pipelineModelMutex);
                    client.model->downstreams.clear();
                    client.model->downstreams.push_back(std::make_pair(m.model, 1));

                    // retrieve new resolution
                    int width = m.width;
                    int height = m.height;
                    m.model->batchSize = batch_size;

                    std::vector<int> rs = {width, height};
                    client.model->dimensions = rs;
                    client_lock.unlock();

                    // std::unique_lock<std::mutex> container_lock(this->containers.containersMutex);
                    // for (auto it = this->containers.list.begin(); it != this->containers.list.end(); ++it)
                    // {
                    //     if (it->first == client.name)
                    //     {
                    //         std::vector<int> rs = {width, height, 3};
                    //         AdjustResolution(it->second, rs);
                    //     }
                    // }
                    // container_lock.unlock();
                }
                model_lock.unlock();
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
                    std::cout << "Client name: " << client.name << ", budget: " << client.budget << ", lat: " << client.transmission_latency << ", client device: " << client.model->device << std::endl;
                }
                std::cout << "Batch size: " << std::get<2>(mapping) << std::endl;
                std::cout <<"-----------------------------------" << std::endl;
            }
            std::cout << "============================= End Mapping =================================" << std::endl;

            // for debugging
            std::cout << "============================== check all clients downstream ==================================" << task_name << std::endl;
            for (auto &client : client_profiles_jf.infos)
            {
                auto p = client.model;
                std::unique_lock<std::mutex> lock(p->pipelineModelMutex);
                std::cout << "datasource name: " << p->datasourceName;
                for (auto &ds : p->downstreams)
                {
                    std::cout << ", ds name: " << ds.first->name << std::endl;
                }
                lock.unlock();
            }
            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;

            std::cout << "================================= check all models upstream ==================================" << task_name << std::endl;
            for (auto &model : model_profiles_jf.infos)
            {
                auto p = model.second.front().model;
                std::unique_lock<std::mutex> lock(p->pipelineModelMutex);
                std::cout << "model name: " << p->name;
                for (auto us : p->upstreams)
                {
                    std::cout << ", us name: " <<us.first->name << ", address of client: " << us.first << "; ";
                }
                std::cout << std::endl;
                lock.unlock();
            }
            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
            std::cout << "================================= check all models downstream ==================================" << task_name << std::endl;
            for (auto &model : model_profiles_jf.infos)
            {
                auto p = model.second.front().model;
                std::unique_lock<std::mutex> lock(p->pipelineModelMutex);
                std::cout << "model name: " << p->name;
                for (auto us : p->downstreams)
                {
                    std::cout << ", ds name: " <<us.first->name << ", address of client: " << us.first << "; ";
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
                estimateModelLatency(model);
            }
            for (auto &model : task.second->tk_pipelineModels)
            {
                if (model->name.find("datasource") == std::string::npos)
                {
                    continue;
                }
                estimateTimeBudgetLeft(model);
            }
            for (auto &model : task.second->tk_pipelineModels)
            {
                // model->timeBudgetLeft = pipelineSLOs[task.second->tk_name] - model->expectedStart2HereLatency -
                //                             model->expectedMaxProcessLatency + model->expectedQueueingLatency + model->expectedTransferLatency;

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

        ctrl_scheduledPipelines = ctrl_unscheduledPipelines;
        ApplyScheduling();
        schedulingSW.stop();
        ctrl_nextSchedulingTime = std::chrono::system_clock::now() + std::chrono::seconds(ctrl_schedulingIntervalSec);
        std::this_thread::sleep_for(TimePrecisionType((ctrl_schedulingIntervalSec + 1) * 1000000 - schedulingSW.elapsed_microseconds()));
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
uint64_t Controller::calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate)
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
ModelInfoJF::ModelInfoJF(int bs, float il, int w, int h, std::string n, float acc, PipelineModel *m)
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

ClientInfoJF::ClientInfoJF(std::string _name, float _budget, int _req_rate,
                           PipelineModel *_model, std::string _task_name, std::string _task_source,
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
void ModelProfilesJF::add(std::string name, float accuracy, int batch_size, float inference_latency, int width, int height, PipelineModel *m)
{
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
    auto key = std::tuple<std::string, float>{model_info.name, model_info.accuracy};
    infos[key].push_back(model_info);
}

void ModelProfilesJF::debugging()
{
    std::cout << "======================ModelProfiles Debugging=======================" << std::endl;
    for (auto it = infos.begin(); it != infos.end(); ++it)
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
                           PipelineModel *model, std::string task_name, std::string task_source,
                           NetworkEntryType network_entry)
{
    infos.push_back(ClientInfoJF(name, budget, req_rate, model, task_name, task_source, network_entry));
}

void ClientProfilesJF::debugging()
{
    std::cout << "===================================ClientProfiles Debugging==========================" << std::endl;
    for (const auto &client_info : infos)
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
 * @param client_profile
 * @param model_profiles
 * @return a vector of [ (model_name, accuracy), vec[clients], batch_size ]
 */
std::vector<
    std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
Jlf::mapClient(ClientProfilesJF &client_profile, ModelProfilesJF &model_profiles)
{

    std::vector<
        std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
        mappings;
    std::vector<ClientInfoJF> clients = client_profile.infos;

    int map_size = model_profiles.infos.size();
    int key_index = 0;
    for (auto it = model_profiles.infos.begin(); it != model_profiles.infos.end();
         ++it)
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

// -------------------------------------------------------------------------------------------
//                                  end of implementations
// -------------------------------------------------------------------------------------------

void Controller::colocationTemporalScheduling() {} // Dummy Method for Compiler
