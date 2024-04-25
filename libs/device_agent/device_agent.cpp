#include "device_agent.h"

ABSL_FLAG(std::string, deviceType, "", "string that identifies the device type");
ABSL_FLAG(std::string, controllerUrl, "", "string that identifies the controller url");

const int CONTAINER_BASE_PORT = 50001;
const int RECEIVER_BASE_PORT = 55001;

void msvcconfigs::to_json(json &j, const msvcconfigs::NeighborMicroserviceConfigs &val) {
    j["nb_name"] = val.name;
    j["nb_commMethod"] = val.commMethod;
    j["nb_link"] = val.link;
    j["nb_maxQueueSize"] = val.maxQueueSize;
    j["nb_classOfInterest"] = val.classOfInterest;
    j["nb_expectedShape"] = val.expectedShape;
}

void msvcconfigs::to_json(json &j, const msvcconfigs::BaseMicroserviceConfigs &val) {
    j["msvc_name"] = val.msvc_name;
    j["msvc_type"] = val.msvc_type;
    j["msvc_svcLevelObjLatency"] = val.msvc_svcLevelObjLatency;
    j["msvc_idealBatchSize"] = val.msvc_idealBatchSize;
    j["msvc_dataShape"] = val.msvc_dataShape;
    j["msvc_maxQueueSize"] = val.msvc_maxQueueSize;
    j["msvc_upstreamMicroservices"] = val.msvc_upstreamMicroservices;
    j["msvc_dnstreamMicroservices"] = val.msvc_dnstreamMicroservices;
}

DeviceAgent::DeviceAgent(const std::string &controller_url, const std::string name, DeviceType type) {
    std::string server_address = absl::StrFormat("%s:%d", "localhost", 60003);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder device_builder;
    device_builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    device_builder.RegisterService(&device_service);
    device_cq = device_builder.AddCompletionQueue();
    device_server = device_builder.BuildAndStart();

    server_address = absl::StrFormat("%s:%d", "localhost", 60002);
    ServerBuilder controller_builder;
    controller_builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    controller_builder.RegisterService(&controller_service);
    controller_cq = controller_builder.AddCompletionQueue();
    controller_server = controller_builder.BuildAndStart();
    std::string target_str = absl::StrFormat("%s:%d", controller_url, 60001);
    controller_stub = ControlCommunication::NewStub(
            grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    controller_sending_cq = new CompletionQueue();

    running = true;
    profiler = new Profiler({});
    containers = std::map<std::string, ContainerHandle>();
    threads = std::vector<std::thread>();
    threads.emplace_back(&DeviceAgent::HandleDeviceRecvRpcs, this);
    threads.emplace_back(&DeviceAgent::HandleControlRecvRpcs, this);
    threads.emplace_back(&DeviceAgent::MonitorDeviceStatus, this);
    for (auto &thread: threads) {
        thread.detach();
    }


    Ready(name, controller_url, type);

    // test code that will eventually be replaced by the controller
    CreateDataSource(0, {{"yolov5_0", CommMethod::serialized, {"localhost:55002"}, 10, -1, {{0, 0}}}}, 1, "./test.mp4",
                     dev_logPath);
    CreateYolo5Container(0, {"datasource_0", CommMethod::serialized, {"localhost:55001"}, 10, -2, {{-1, -1, -1}}},
                         {{"dummy_receiver_0", CommMethod::localGPU, {"localhost:55003"}, 10, -1, {{0, 0}}}}, 1, 10,
                         dev_logPath);
}

void DeviceAgent::CreateYolo5Container(
        int id,
        const NeighborMicroserviceConfigs &upstream,
        const std::vector<NeighborMicroserviceConfigs> &downstreams,
        const MsvcSLOType &slo,
        const BatchSizeType &batchSize,
        const std::string &logPath
) {
    std::string name = "yolov5_" + std::to_string(id);
    json j = createConfigs(
            {{name + "::receiver",                 MicroserviceType::Receiver,                 10, -1, {{-1, -1}},                    100},
             {name +
              "::PreprocessBatcher",               MicroserviceType::PreprocessBatcher,        10, -1, {{-1, -1,  -1}},               10},
             {name +
              "::TRTInferencer",                   MicroserviceType::TRTInferencer,            10, -1, {{3,  640, 640}},              10},
             {name +
              "::PostprocessorBBoxCropper",        MicroserviceType::PostprocessorBBoxCropper, 10, -1, {{1}, {100, 4}, {100}, {100}}, 100},
             {name +
              "::sender",                          MicroserviceType::Sender,                   10, -1, {{-1, -1}},                    10}},
            slo,
            batchSize,
            logPath,
            upstream,
            downstreams
    );
    // TRTConfigs config = {"./models/yolov5s_b32_dynamic_NVIDIAGeForceRTX3090_fp32_32_1.engine", MODEL_DATA_TYPE::fp32, "", 128, 1, 1, 0, true};
    finishContainer("./Container_Yolov5", name, to_string(j), CONTAINER_BASE_PORT + containers.size(),
                    RECEIVER_BASE_PORT + containers.size());
}

void DeviceAgent::CreateDataSource(
        int id,
        const std::vector<NeighborMicroserviceConfigs> &downstreams,
        const MsvcSLOType &slo,
        const std::string &video_path,
        const std::string &logPath
) {
    std::string name = "datasource_" + std::to_string(id);
    NeighborMicroserviceConfigs upstream = {"video_source", CommMethod::localCPU, {video_path}, 0, -2, {{0, 0}}};
    json j = createConfigs(
            {{name + "::data_reader", MicroserviceType::PostprocessorBBoxCropper, 10, -1, {{0, 0}}, 100},
             {name + "::sender",      MicroserviceType::Sender,                   10, -1, {{0, 0}}, 100}},
            slo,
            1,
            logPath,
            upstream,
            downstreams
    );
    finishContainer("./Container_DataSource", name, to_string(j), CONTAINER_BASE_PORT + containers.size(),
                    RECEIVER_BASE_PORT + containers.size());
}

json DeviceAgent::createConfigs(
        const std::vector<MsvcConfigTupleType> &data,
        const MsvcSLOType &slo,
        const BatchSizeType &batchSize,
        const std::string &logPath,
        const NeighborMicroserviceConfigs &prev_msvc,
        const std::vector<NeighborMicroserviceConfigs> &next_msvc
) {
    int i = 0, j = next_msvc.size() + 1;
    std::vector<BaseMicroserviceConfigs> configs;
    NeighborMicroserviceConfigs upstream = prev_msvc;
    for (auto &msvc: data) {
        std::list<NeighborMicroserviceConfigs> downstream;
        if (std::get<1>(msvc) == MicroserviceType::PostprocessorBBoxCropper) {
            while (--j > 0) {
                downstream.push_back(
                        {std::get<0>(data[i + j]), CommMethod::localGPU, {""}, std::get<2>(data[i + j]),
                         std::get<3>(data[i + j]), std::get<4>(data[i + j])});
            }
        } else if (std::get<1>(msvc) == MicroserviceType::Sender) {
            downstream.push_back(next_msvc[j++]);
        } else {
            downstream.push_back(
                    {std::get<0>(data[++i]), CommMethod::localGPU, {""}, std::get<2>(data[i]), std::get<3>(data[i]),
                     std::get<4>(data[i])});
        }
        configs.push_back(
                {std::get<0>(msvc), std::get<1>(msvc), "", slo, std::get<5>(msvc), batchSize, std::get<4>(msvc), -1,
                 logPath, RUNMODE::DEPLOYMENT, {upstream}, downstream});
        //current msvc becomes upstream for next msvc
        upstream = {std::get<0>(msvc), CommMethod::localGPU, {""}, std::get<2>(msvc), -2, std::get<4>(msvc)};
    }
    return json(configs);
}

void
DeviceAgent::finishContainer(const std::string &executable, const std::string &name, const std::string &start_string,
                             const int &control_port, const int &data_port, const std::string &trt_config) {
    runDocker(executable, name, start_string, control_port, trt_config);
    std::string target = absl::StrFormat("%s:%d", "localhost", control_port);
    containers[name] = {{},
                        InDeviceCommunication::NewStub(grpc::CreateChannel(target, grpc::InsecureChannelCredentials())),
                        {}, new CompletionQueue(), 0};
}

void DeviceAgent::StopContainer(const ContainerHandle &container) {
    EmptyMessage request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container.stub->AsyncStopExecution(&context, request, container.cq));
    rpc->Finish(&reply, &status, (void *) 1);
}

void DeviceAgent::Ready(const std::string &name, const std::string &ip, DeviceType type) {
    ConnectionConfigs request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_device_name(name);
    request.set_device_type(type);
    request.set_ip_address(ip);
    if (type == DeviceType::Server) {
        int devices = profiler->getGpuCount();
        request.set_processors(devices);
        request.set_memory(profiler->getGpuMemory(devices));
    } else {
        struct sysinfo sys_info;
        if(sysinfo(&sys_info) != 0)
        {
            std::cerr << "sysinfo call failed!" << std::endl;
            exit(1);
        }
        request.set_processors(sys_info.procs);
        request.set_memory(sys_info.totalram * sys_info.mem_unit / 1000000);
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            controller_stub->AsyncAdvertiseToController(&context, request, controller_sending_cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(controller_sending_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    if (!status.ok()) {
        std::cerr << "Ready RPC failed" << status.error_code() << ": " << status.error_message() << std::endl;
        exit(1);
    }
}

void DeviceAgent::ReportDeviceStatus() {
    LightMetricsList request;
    EmptyMessage reply;
    ClientContext context;
    Status status;

    for (auto &container: containers) {
        LightMetrics *metrics = request.add_metrics();
        metrics->set_name(container.first);
        for (auto &size: container.second.queuelengths) {
            metrics->add_queue_size(size);
        }
        metrics->set_request_rate(container.second.metrics.requestRate);
    }


    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            controller_stub->AsyncSendLightMetrics(&context, request, controller_sending_cq));
    rpc->Finish(&reply, &status, (void *) 1);
}

void DeviceAgent::ReportFullMetrics() {
    FullMetricsList request;
    EmptyMessage reply;
    ClientContext context;
    Status status;

    for (auto &container: containers) {
        FullMetrics *metrics = request.add_metrics();
        metrics->set_name(container.first);
        for (auto &size: container.second.queuelengths) {
            metrics->add_queue_size(size);
        }
        metrics->set_request_rate(container.second.metrics.requestRate);
        metrics->set_cpu_usage(container.second.metrics.cpuUsage);
        metrics->set_mem_usage(container.second.metrics.memUsage);
        metrics->set_gpu_usage(container.second.metrics.gpuUsage);
        metrics->set_gpu_mem_usage(container.second.metrics.gpuMemUsage);
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            controller_stub->AsyncSendFullMetrics(&context, request, controller_sending_cq));
    rpc->Finish(&reply, &status, (void *) 1);
}

void DeviceAgent::HandleDeviceRecvRpcs() {
    new StateUpdateRequestHandler(&device_service, device_cq.get(), this);
    new ReportStartRequestHandler(&device_service, device_cq.get(), this);
    while (running) {
        void *tag;
        bool ok;
        if (!device_cq->Next(&tag, &ok)) {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void DeviceAgent::HandleControlRecvRpcs() {
    new StartMicroserviceRequestHandler(&controller_service, controller_cq.get(), this);
    new StopMicroserviceRequestHandler(&controller_service, controller_cq.get(), this);
    while (running) {
        void *tag;
        bool ok;
        if (!device_cq->Next(&tag, &ok)) {
            break;
        }
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void DeviceAgent::MonitorDeviceStatus() {
    profiler->run();
    int i = 0;
    while (running) {
        if (i++ > 20) {
            i = 0;
            ReportFullMetrics();
        } else {
            for (auto &container: containers) {
                Profiler::sysStats stats = profiler->reportAtRuntime(container.second.pid);
                container.second.metrics.cpuUsage =
                        (1 - 1 / i) * container.second.metrics.cpuUsage + (1 / i) * stats.cpuUsage;
                container.second.metrics.memUsage =
                        (1 - 1 / i) * container.second.metrics.memUsage + (1 / i) * stats.memoryUsage;
                container.second.metrics.gpuUsage =
                        (1 - 1 / i) * container.second.metrics.memUsage + (1 / i) * stats.gpuUtilization;
                container.second.metrics.gpuMemUsage =
                        (1 - 1 / i) * container.second.metrics.memUsage + (1 / i) * stats.gpuMemoryUsage;
            }
            ReportDeviceStatus();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
}

void DeviceAgent::StateUpdateRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendState(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StateUpdateRequestHandler(service, cq, device_agent);
        device_agent->UpdateState(request.name(), request.arrival_rate(), request.queue_size());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::ReportStartRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestReportMsvcStart(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new ReportStartRequestHandler(service, cq, device_agent);
        std::cout << "Received start report from " << request.msvc_name() << std::endl;
        device_agent->containers[request.msvc_name()].pid = request.pid();
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::StartMicroserviceRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStartMicroservice(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StartMicroserviceRequestHandler(service, cq, device_agent);
        // TODO: add logic to start container
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::StopMicroserviceRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStopMicroservice(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StopMicroserviceRequestHandler(service, cq, device_agent);
        device_agent->StopContainer(device_agent->containers[request.name()]);
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    std::string name = absl::GetFlag(FLAGS_name);
    std::string type = absl::GetFlag(FLAGS_deviceType);
    std::string controller_url = absl::GetFlag(FLAGS_controllerUrl);
    DeviceType deviceType;
    if (type == "Server")
        deviceType = DeviceType::Server;
    else if (type == "Edge")
        deviceType = DeviceType::Edge;
    else {
        std::cerr << "Invalid device type" << std::endl;
        exit(1);
    }

    DeviceAgent agent(controller_url, name, deviceType);
}