#include "controller.h"

std::map<ModelType, std::vector<std::string>> MODEL_INFO = {
        {DataSource,        {":datasource",         "./Container_DataSource"}},
        {Sink,              {":basesink",           "./runSink"}},
        {Yolov5,            {":yolov5",             "./Container_Yolov5"}},
        {Yolov5Datasource,  {":yolov5datasource",   "./Container_Yolov5"}},
        {Retinaface,        {":retinaface",         "./Container_RetinaFace"}},
        {Yolov5_Plate,      {":platedetection",     "./Container_Yolov5-plate"}},
        {Movenet,           {":movenet",            "./Container_MoveNet"}},
        {Emotionnet,        {":emotionnet",         "./Container_EmotionNet"}},
        {Arcface,           {":arcface",            "./Container_ArcFace"}},
        {Age,               {":age",                "./Container_Age"}},
        {Gender,            {":gender",             "./Container_Gender"}},
        {CarBrand,          {":carbrand",           "./Container_CarBrand"}},
};

void TaskDescription::to_json(nlohmann::json &j, const TaskDescription::TaskStruct &val)
{
    j = json{{"name", val.name},
             {"slo", val.slo},
             {"type", val.type},
             {"source", val.source},
             {"device", val.device}};
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val)
{
    j.at("name").get_to(val.name);
    j.at("slo").get_to(val.slo);
    j.at("type").get_to(val.type);
    j.at("source").get_to(val.source);
    j.at("device").get_to(val.device);
}

Controller::Controller()
{
    running = true;
    devices = std::map<std::string, NodeHandle>();
    tasks = std::map<std::string, TaskHandle>();
    containers = std::map<std::string, ContainerHandle>();

    // ========================= added =============================
    auto clients_profiles = ClientProfiles();
    auto models_profiles = ModelProfiles();

    // =============================================================

    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", 60001);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();
}

Controller::~Controller()
{
    for (auto &msvc : containers)
    {
        StopContainer(msvc.first, msvc.second.device_agent, true);
    }
    for (auto &device : devices)
    {
        device.second.cq->Shutdown();
        void *got_tag;
        bool ok = false;
        while (device.second.cq->Next(&got_tag, &ok))
            ;
    }
    server->Shutdown();
    cq->Shutdown();
}

void Controller::HandleRecvRpcs()
{
    new DeviseAdvertisementHandler(&service, cq.get(), this);
    while (running)
    {
        void *tag;
        bool ok;
        if (!cq->Next(&tag, &ok))
        {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::AddTask(const TaskDescription::TaskStruct &t)
{
    std::cout << "Adding task: " << t.name << std::endl;
    tasks.insert({t.name, {t.slo, t.type, {}}});
    TaskHandle *task = &tasks[t.name];
    NodeHandle *device = &devices[t.device];
    auto models = getModelsByPipelineType(t.type);

    std::string tmp = t.name;
    containers.insert({tmp.append(":datasource"), {tmp, DataSource, device, task, 33, 1, {-1}}});

    // ================== added ===============

    std::string client_ip = device->ip;

    // ========================================
    task->subtasks.insert({tmp, &containers[tmp]});
    task->subtasks[tmp]->recv_port = device->next_free_port++;
    device->containers.insert({tmp, task->subtasks[tmp]});
    device = &devices["server"];

    auto batch_sizes = getInitialBatchSizes(models, t.slo, 10);
    for (const auto &m : models)
    {
        tmp = t.name;
        // TODO: get correct initial cuda devices based on TaskDescription and System State
        int cuda_device = 1;
        containers.insert(
                {tmp.append(MODEL_INFO[m.first][0]), {tmp, m.first, device, task, batch_sizes[m.first], 1, {cuda_device}, -1, device->next_free_port++, {}, {}, {}, {}}});
        task->subtasks.insert({tmp, &containers[tmp]});
        device->containers.insert({tmp, task->subtasks[tmp]});
    }

    task->subtasks[t.name + ":datasource"]->downstreams.push_back(task->subtasks[t.name + MODEL_INFO[models[0].first][0]]);
    task->subtasks[t.name + MODEL_INFO[models[0].first][0]]->upstreams.push_back(task->subtasks[t.name + ":datasource"]);

    // ======== added ========

    // get the req of the first model(yolov5) of the pipeline
    // FIXME@yuheng: the budget is the SLO - networking time in the paper
    clients_profiles.add(client_ip, t.slo, task->subtasks[t.name + models[0].first]->metrics.requestRate);
    // TODO@yuheng: wait for the implementation to retrieve the model profiles
    models_profiles.add(models[0].first, 0, 2, 0.1, 320, 20);
    // record the data source and first container of each pipeline for further update the upstream when switching mapping
    data_sources.push_back(task->subtasks[t.name + ":datasource"]);
    first_containers.push_back(task->subtasks[t.name + models[0].first]);

    // =======================
    for (const auto &m : models)
    {
        for (const auto &d : m.second)
        {
            tmp = t.name;
            task->subtasks[tmp.append(MODEL_INFO[d.first][0])]->class_of_interest = d.second;
            task->subtasks[tmp]->upstreams.push_back(task->subtasks[t.name + MODEL_INFO[m.first][0]]);
            task->subtasks[t.name + MODEL_INFO[m.first][0]]->downstreams.push_back(task->subtasks[tmp]);
        }
    }

    for (std::pair<std::string, ContainerHandle *> msvc : task->subtasks)
    {
        StartContainer(msvc, task->slo, t.source);
    }
}

// ========================== added ================================

/**
 * @brief update client-dnn mapping every 0.5s, and adjust the upstream of each model, set the input size of clients
 *
 */
void Controller::update_and_adjust(int mills)
{
    std::chrono::milliseconds interval(mills);
    while (true)
    {
        auto start = std::chrono::system_clock::now();
        auto mappings = mapClient(this->clients_profiles, this->models_profiles);

        // adjust the upstream of first container in the pipeline

        // find the new downstream of data_source
        for (auto &mapping : mappings)
        {

            const auto [model_info, selected_clients, batch_size] = mapping;

            // match model with corresponding ContainerHandle
            Controller::ContainerHandle *p;
            for (auto &first_container : first_containers)
            {
                if (first_container->name == model_info.name)
                {
                    p = first_container;
                    break;
                }
            }
            // clear the upstream of this p, later adding new upstreams
            p->upstreams.clear();

            // adjust downstream
            for (auto &client : selected_clients)
            {
                // match with corresponding ContainerHandle

                Controller::ContainerHandle *ds;
                for (auto &data_source : data_sources)
                {
                    if (data_source->device_agent->ip == client.ip)
                    {
                        ds = data_source;
                        break;
                    }
                }

                // adjust the upstream and downstream
                ds->downstreams.clear();
                ds->downstreams.push_back(p);
                p->upstreams.push_back(ds);

                // TODO: set (1) the batch_size of this container (2) the resolution of client
            }
        }

        std::this_thread::sleep_until(start + interval);
    }
}

// =================================================================

void Controller::UpdateLightMetrics() {
    // TODO: Replace with Database Scraping
//    for (auto metric: metrics) {
//        containers[metric.name()].queue_lengths = metric.queue_size();
//        containers[metric.name()].metrics.requestRate = metric.request_rate();
//    }
}

void Controller::UpdateFullMetrics() {
    //TODO: Replace with Database Scraping
//    for (auto metric: metrics) {
//        containers[metric.name()].queue_lengths = metric.queue_size();
//        Metrics *m = &containers[metric.name()].metrics;
//        m->requestRate = metric.request_rate();
//        m->cpuUsage = metric.cpu_usage();
//        m->memUsage = metric.mem_usage();
//        m->gpuUsage = metric.gpu_usage();
//        m->gpuMemUsage = metric.gpu_mem_usage();
//    }
}

void Controller::DeviseAdvertisementHandler::Proceed()
{
    if (status == CREATE)
    {
        status = PROCESS;
        service->RequestAdvertiseToController(&ctx, &request, &responder, cq, cq, this);
    }
    else if (status == PROCESS)
    {
        new DeviseAdvertisementHandler(service, cq, controller);
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), 60002);
        controller->devices.insert({request.device_name(),
                                    {request.ip_address(),
                                     ControlCommunication::NewStub(
                                         grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<SystemDeviceType>(request.device_type()),
                                     request.processors(),
                                     std::vector<double>(request.processors(), 0.0),
                                     std::vector<unsigned long>(request.memory().begin(), request.memory().end()),
                                     std::vector<double>(request.processors(), 0.0),
                                     55001,
                                     {}}});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    }
    else
    {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::StartContainer(std::pair<std::string, ContainerHandle *> &container, int slo, std::string source,
                                int replica)
{
    std::cout << "Starting container: " << container.first << std::endl;
    ContainerConfig request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(container.first);
    request.set_model(container.second->model);
    request.set_batch_size(container.second->batch_size);
    request.set_recv_port(container.second->recv_port);
    request.set_slo(slo);
    request.set_device(container.second->cuda_device[replica - 1]);
    for (auto dwnstr : container.second->downstreams)
    {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name(dwnstr->name);
        dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port));
        dwn->set_class_of_interest(dwnstr->class_of_interest);
        dwn->set_gpu_connection((container.second->device_agent == dwnstr->device_agent) &&
                                (container.second->cuda_device == dwnstr->cuda_device));
    }
    if (request.downstream_size() == 0)
    {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name("video_sink");
        dwn->set_ip("./out.log"); // output log file
        dwn->set_class_of_interest(-1);
        dwn->set_gpu_connection(false);
    }
    if (container.second->model == DataSource)
    {
        Neighbor *up = request.add_upstream();
        up->set_name("video_source");
        up->set_ip(source);
        up->set_class_of_interest(-1);
        up->set_gpu_connection(false);
    }
    else
    {
        for (auto upstr : container.second->upstreams)
        {
            Neighbor *up = request.add_upstream();
            up->set_name(upstr->name);
            up->set_ip(absl::StrFormat("%s:%d", upstr->device_agent->ip, upstr->recv_port));
            up->set_class_of_interest(-2);
            up->set_gpu_connection((container.second->device_agent == upstr->device_agent) &&
                                   (container.second->cuda_device == upstr->cuda_device));
        }
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        container.second->device_agent->stub->AsyncStartContainer(&context, request,
                                                                      container.second->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(container.second->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    if (!status.ok())
    {
        std::cout << status.error_code() << ": An error occured while sending the request" << std::endl;
    }
}

void Controller::MoveContainer(ContainerHandle *msvc, int cuda_device, bool to_edge, int replica)
{
    NodeHandle *old_device = msvc->device_agent;
    NodeHandle *device;
    if (to_edge)
    {
        device = msvc->upstreams[0]->device_agent;
    }
    else
    {
        device = &devices["server"];
    }
    msvc->device_agent = device;
    msvc->recv_port = device->next_free_port++;
    device->containers.insert({msvc->name, msvc});
    msvc->cuda_device[replica -1] = cuda_device;
    std::pair<std::string, ContainerHandle *> pair = {msvc->name, msvc};
    StartContainer(pair, msvc->task->slo, "");
    for (auto upstr : msvc->upstreams)
    {
        AdjustUpstream(msvc->recv_port, upstr, device, msvc->name);
    }
    StopContainer(msvc->name, old_device);
    old_device->containers.erase(msvc->name);
}

void Controller::AdjustUpstream(int port, Controller::ContainerHandle *upstr, Controller::NodeHandle *new_device,
                                const std::string &dwnstr)
{
    ContainerLink request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(upstr->name);
    request.set_downstream_name(dwnstr);
    request.set_ip(new_device->ip);
    request.set_port(port);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        upstr->device_agent->stub->AsyncUpdateDownstream(&context, request, upstr->device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(upstr->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::AdjustBatchSize(Controller::ContainerHandle *msvc, int new_bs) {
    msvc->batch_size = new_bs;
    ContainerInt request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.set_value(new_bs);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(msvc->device_agent->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::StopContainer(std::string name, NodeHandle *device, bool forced)
{
    ContainerSignal request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(name);
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
        device->stub->AsyncStopContainer(&context, request, containers[name].device_agent->cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(device->cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void Controller::optimizeBatchSizeStep(
        const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models,
        std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects) {
    ModelType candidate;
    int max_saving = 0;
    std::vector<ModelType> blacklist;
    for (const auto &m: models) {
        int saving;
        if (max_saving == 0) {
            saving =
                    estimated_infer_times[m.first] - InferTimeEstimator(m.first, batch_sizes[m.first] * 2);
        } else {
            if (batch_sizes[m.first] == 64 ||
                std::find(blacklist.begin(), blacklist.end(), m.first) != blacklist.end()) {
                continue;
            }
            for (const auto &d: m.second) {
                if (batch_sizes[d.first] > batch_sizes[m.first]) {
                    blacklist.push_back(d.first);
                }
            }
            saving = estimated_infer_times[m.first] -
                     (InferTimeEstimator(m.first, batch_sizes[m.first] * 2) * (nObjects / batch_sizes[m.first] * 2));
        }
        if (saving > max_saving) {
            max_saving = saving;
            candidate = m.first;
        }
    }
    batch_sizes[candidate] *= 2;
    estimated_infer_times[candidate] -= max_saving;
}

std::map<ModelType, int> Controller::getInitialBatchSizes(
        const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models, int slo,
        int nObjects) {
    std::map<ModelType, int> batch_sizes = {};
    std::map<ModelType, int> estimated_infer_times = {};

    for (const auto &m: models) {
        batch_sizes[m.first] = 1;
        if (estimated_infer_times.size() == 0) {
            estimated_infer_times[m.first] = (InferTimeEstimator(m.first, 1));
        } else {
            estimated_infer_times[m.first] = (InferTimeEstimator(m.first, 1) * nObjects);
        }
    }

    int sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<ModelType, int> &p) {
                                  return acc + p.second;
                              });

    while (slo < sum) {
        optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
        sum = std::accumulate(estimated_infer_times.begin(), estimated_infer_times.end(), 0,
                              [](int acc, const std::pair<ModelType, int> &p) {
                                  return acc + p.second;
                              });
    }
    optimizeBatchSizeStep(models, batch_sizes, estimated_infer_times, nObjects);
    return batch_sizes;
}

std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>>
Controller::getModelsByPipelineType(PipelineType type)
{
    switch (type)
    {
    case PipelineType::Traffic:
        return {{ModelType::Yolov5,       {{ModelType::Retinaface, 0}, {ModelType::CarBrand, 2}, {ModelType::Yolov5_Plate, 2}}},
                {ModelType::Retinaface,   {{ModelType::Arcface, -1}}},
                {ModelType::Arcface,      {{ModelType::Sink, -1}}},
                {ModelType::CarBrand,     {{ModelType::Sink, -1}}},
                {ModelType::Yolov5_Plate, {{ModelType::Sink, -1}}},
                {ModelType::Sink, {}}};
    case PipelineType::Video_Call:
        return {{ModelType::Retinaface, {{ModelType::Emotionnet, -1}, {ModelType::Age, -1}, {ModelType::Gender, -1}, {ModelType::Arcface, -1}}},
                {ModelType::Gender,     {{ModelType::Sink, -1}}},
                {ModelType::Age,        {{ModelType::Sink, -1}}},
                {ModelType::Emotionnet, {{ModelType::Sink, -1}}},
                {ModelType::Arcface,    {{ModelType::Sink, -1}}},
                {ModelType::Sink, {}}};
    case PipelineType::Building_Security:
        return {{ModelType::Yolov5,     {{ModelType::Retinaface, 0}}},
                {ModelType::Retinaface, {{ModelType::Gender, -1}, {ModelType::Age, -1}}},
                {ModelType::Movenet,    {{ModelType::Sink,   -1}}},
                {ModelType::Gender,     {{ModelType::Sink, -1}}},
                {ModelType::Age,        {{ModelType::Sink, -1}}},
                {ModelType::Sink, {}}};
    default:
        return {};
    }
}

double Controller::LoadTimeEstimator(const char *model_path, double input_mem_size)
{
    // Load the pre-trained model
    BoosterHandle booster;
    int num_iterations = 1;
    int ret = LGBM_BoosterCreateFromModelfile(model_path, &num_iterations, &booster);

    // Prepare the input data
    std::vector<double> input_data = {input_mem_size};

    // Perform inference
    int64_t out_len;
    std::vector<double> out_result(1);
    ret = LGBM_BoosterPredictForMat(booster,
                                    input_data.data(),
                                    C_API_DTYPE_FLOAT64,
                                    1,                    // Number of rows
                                    1,                    // Number of columns
                                    1,                    // Is row major
                                    C_API_PREDICT_NORMAL, // Predict type
                                    0,                    // Start iteration
                                    -1,                   // Number of iterations, -1 means use all
                                    "",                   // Parameter
                                    &out_len,
                                    out_result.data());
    if (ret != 0)
    {
        std::cout << "Failed to perform inference!" << std::endl;
        exit(ret);
    }

    // Print the predicted value
    std::cout << "Predicted value: " << out_result[0] << std::endl;

    // Free the booster handle
    LGBM_BoosterFree(booster);

    return out_result[0];
}

// returns the profiling results for inference time per frame in a full batch in nanoseconds
int Controller::InferTimeEstimator(ModelType model, int batch_size) {
    std::map<int, int> time_per_frame;
    switch (model) {
        case ModelType::Yolov5:
            time_per_frame = {{1,  3602348},
                              {2,  2726377},
                              {4,  2467065},
                              {8,  2575456},
                              {16, 3220761},
                              {32, 4680154},
                              {64, 7773959}};
            break;
        case ModelType::Yolov5Datasource:
            time_per_frame = {{1,  3602348},
                              {2,  2726377},
                              {4,  2467065},
                              {8,  2575456},
                              {16, 3220761},
                              {32, 4680154},
                              {64, 7773959}};
            break;
        case ModelType::Retinaface:
            time_per_frame = {{1,  1780280},
                              {2,  1527410},
                              {4,  1357906},
                              {8,  1164929},
                              {16, 2177011},
                              {32, 3399701},
                              {64, 8146690}};
            break;
        case ModelType::CarBrand:
            time_per_frame = {{1,  4998407},
                              {2,  3335101},
                              {4,  2344440},
                              {8,  2176385},
                              {16, 2483317},
                              {32, 2357686},
                              {64, 1155050}};
            break;
        case ModelType::Yolov5_Plate:
            time_per_frame = {{1,  7304176},
                              {2,  4909581},
                              {4,  3225549},
                              {8,  2883803},
                              {16, 2871236},
                              {32, 2004165},
                              {64, 3094331}};
            break;
        case ModelType::Movenet:
            time_per_frame = {{1,  1644526},
                              {2,  3459537},
                              {4,  2703916},
                              {8,  2377614},
                              {16, 2647643},
                              {32, 2900894},
                              {64, 2197719}};
            break;
        case ModelType::Arcface:
            time_per_frame = {{1,  18120029},
                              {2,  11226197},
                              {4,  7883673},
                              {8,  6364369},
                              {16, 5620677},
                              {32, 3370018},
                              {64, 3206726}};
            break;
        case ModelType::Emotionnet:
            time_per_frame = {{1,  3394144},
                              {2,  1365037},
                              {4,  1615653},
                              {8,  1967143},
                              {16, 1500867},
                              {32, 1665680},
                              {64, 1957914}};
            break;
        case ModelType::Age:
            time_per_frame = {{1,  14729041},
                              {2,  9050828},
                              {4,  6112501},
                              {8,  5015442},
                              {16, 3927934},
                              {32, 3523500},
                              {64, 2899034}};
            break;
        case ModelType::Gender:
            time_per_frame = {{1,  1357500},
                              {2,  831649},
                              {4,  687484},
                              {8,  749792},
                              {16, 1021500},
                              {32, 1800263},
                              {64, 4002824}};
            break;
        default:
            return 0;
    }
    int i = 1;
    while (i < batch_size) {
        i *= 2;
    }
    return time_per_frame[batch_size] * batch_size;
}

// ============================================== added ===================================================

bool ModelSetCompare::operator()(const std::tuple<std::string, float> &lhs, const std::tuple<std::string, float> &rhs) const
{
    return std::get<1>(lhs) < std::get<1>(rhs);
}

/**
 * @brief add profiled information of model
 *
 * @param model_type
 * @param accuracy
 * @param batch_size
 * @param inference_latency
 * @param throughput
 */
void ModelProfiles::add(std::string name, float accuracy, int batch_size, float inference_latency, int resolution, int throughput)
{
    auto key = std::tuple<std::string, float>{name, accuracy};
    ModelInfo value = {batch_size, inference_latency, throughput, resolution, name, accuracy};
    infos[key].push_back(value);
}

void ClientProfiles::sortBudgetDescending(std::vector<ClientInfo> &clients)
{
    std::sort(clients.begin(), clients.end(), [](const ClientInfo &a, const ClientInfo &b)
    { return a.budget > b.budget; });
}

void ClientProfiles::add(const std::string &ip, float budget, int req_rate)
{
    infos.push_back({ip, budget, req_rate});
}

std::vector<ClientInfo> findOptimalClients(const std::vector<ModelInfo> &models, std::vector<ClientInfo> &clients)
{
    // sort clients
    ClientProfiles::sortBudgetDescending(clients);
    std::tuple<int, int> best_cell;
    int best_value;

    // dp
    auto [max_batch_size, max_index] = findMaxBatchSize(models, clients[0]);
    assert(max_batch_size > 0);

    // construct the dp matrix
    int rows = clients.size() + 1;
    int h = 10; // assume gcd of all clients' req rate
    int max_throughput = std::max_element(
            models.begin(), models.end(),
            [](const ModelInfo &a, ModelInfo &b)
            {
                return a.throughput < b.throughput;
            });

    int cols = max_throughput % h + 1;
    // init matrix
    std::vector<std::vector<int>> dp_mat(rows, std::vector<int>(cols, 0));
    // iterating
    int client_index = 1;
    for (auto &client : clients)
    {
        auto [max_batch_size, max_index] = findMaxBatchSize(models, client);
        if (max_batch_size <= 0)
        {
            break;
        }
        int cols_upperbound = models[max_index].throughput % h;
        int lambda_i = client.req_rate;
        int v_i = client.req_rate;
        for (int k = 1; k <= cols_upperbound; k++)
        {

            int w_k = k * h;
            if (lambda_i <= w_k)
            {
                int k_prime = (w_k - lambda_i) / h;
                int v = v_i + dp_mat[client_index - 1][k_prime];
                if (v > dp_mat[client_index - 1][k])
                {
                    dp_mat[client_index][k] = v;
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
        client_index++;
    }
    // perform backtracing from (row, col)
    // using dp_mat, best_cell, best_value

    std::vector<ClientInfo> selected_clients;

    auto [row, col] = best_cell;
    int w = dp_mat[row][col];
    while (row > 0 && col > 0)
    {
        if (dp_mat[row][col] == dp_mat[row - 1][col])
        {
            row--;
        }
        else
        {
            int w_i = clients[row - 1].req_rate;
            row = row - 1;
            col = (w - w_i) / h;
            w = col * h;
            assert(w == dp_mat[row][col]);
            selected_clients.push_back(clients[row - 1]);
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
std::vector<std::tuple<std::tuple<std::string, float>, std::vector<ClientInfo>, int>>
mapClient(ClientProfiles client_profile, ModelProfiles model_profiles)
{
    std::vector<std::tuple<std::tuple<std::string, float>, std::vector<ClientInfo>, int>> mappings;
    std::vector<ClientInfo> clients = client_profile.infos;
    for (auto it = model_profiles.infos.begin(); it != model_profiles.infos.end(); ++it)
    {
        auto selected_clients = findOptimalClients(it->second, clients);
        int batch_size = check_and_assign(it->second, selected_clients);
        mappings.push_back(std::make_tuple(it->first, selected_clients, batch_size));
        differenceClients(clients, selected_clients);
    }
    return mappings;
}

/**
 * @brief find the max available batch size for the associated clients of corresponding model
 *
 * @param model
 * @param selected_clients
 * @return int
 */
int check_and_assign(std::vector<ModelInfo> &model, std::vector<ClientInfo> &selected_clients)
{
    int total_req_rate = 0;
    // sum all selected req rate
    for (auto &client : selected_clients)
    {
        total_req_rate += client.req_rate;
    }
    int max_batch_size = 0;

    for (auto &model_info : model)
    {
        if (model_info.throughput > total_req_rate && max_batch_size < model_info.batch_size)
        {
            max_batch_size = model_info.batch_size;
        }
    }
    return max_batch_size;
}

// ====================== helper functions implementation ===========================

/**
 * @brief find the maximum batch size for the client
 *
 * @param models
 * @param budget
 * @return int
 */
std::tuple<int, int> findMaxBatchSize(const std::vector<ModelInfo> &models, const ClientInfo &client)
{
    int max_batch_size = 0;
    float budget = client.budget;
    int index = 0;
    int max_index = 0;
    for (const auto &model : models)
    {
        if (model.inferent_latency * 2.0 < client.budget && model.batch_size > max_batch_size)
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
void differenceClients(std::vector<ClientInfo> &src, const std::vector<ClientInfo> &diff)
{
    auto is_in_diff = [&diff](const ClientInfo &client)
    {
        return std::find(diff.begin(), diff.end(), client) != diff.end();
    };
    src.erase(std::remove_if(src.begin(), src.end(), is_in_diff), src.end());
}

// ====================================================================================
