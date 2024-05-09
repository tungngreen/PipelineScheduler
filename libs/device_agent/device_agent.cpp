#include "device_agent.h"

ABSL_FLAG(std::string, deviceType, "", "string that identifies the device type");
ABSL_FLAG(std::string, controllerUrl, "", "string that identifies the controller url");

const unsigned long CONTAINER_BASE_PORT = 50001;

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
    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", 60003);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder device_builder;
    device_builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    device_builder.RegisterService(&device_service);
    device_cq = device_builder.AddCompletionQueue();
    device_server = device_builder.BuildAndStart();

    server_address = absl::StrFormat("%s:%d", "0.0.0.0", 60002);
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
}

bool DeviceAgent::CreateContainer(
        ModelType model,
        std::string name,
        BatchSizeType batch_size,
        const MsvcSLOType &slo,
        const google::protobuf::RepeatedPtrField<Neighbor> &upstreams,
        const google::protobuf::RepeatedPtrField<Neighbor> &downstreams
) {
    std::ifstream file;
    std::string executable;
    switch (model) {
        case ModelType::Age:
            file.open("../jsons/age.json");
            executable = "./Container_Age";
            break;
        case ModelType::Arcface:
            file.open("../jsons/arcface.json");
            executable = "./Container_Arcface";
            break;

        case ModelType::DataSource:
            file.open("../jsons/data_source.json");
            executable = "./Container_DataSource";
            break;
        case ModelType::Emotionnet:
            file.open("../jsons/emotionnet.json");
            executable = "./Container_Emotionnet";
            break;
        case ModelType::Gender:
            file.open("../jsons/gender.json");
            executable = "./Container_Gender";
            break;
        case ModelType::Movenet:
            file.open("../jsons/movenet.json");
            executable = "./Container_Movenet";
            break;
        case ModelType::Retinaface:
            file.open("../jsons/retinaface.json");
            executable = "./Container_Retinaface";
            break;
        case ModelType::Yolov5:
            file.open("../jsons/yolov5.json");
            executable = "./Container_Yolov5";
            break;
        case ModelType::Yolov5_Plate:
            file.open("../jsons/yolov5-plate.json");
            executable = "./Container_Yolov5_Plate";
            break;
        default:
            std::cerr << "Invalid model type" << std::endl;
            return false;
    }
    json start_config = json::parse(file);
    json base_config = start_config["container"]["cont_pipeline"];
    file.close();

    //adjust configs themselves
    for (auto &j: base_config) {
        j["msvc_name"] = name + j["msvc_name"].get<std::string>();
        j["msvc_idealBatchSize"] = batch_size;
        j["msvc_svcLevelObjLatency"] = slo;
    }

    //adjust upstreams
    base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = upstreams.at(0).name();
    base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"] = upstreams.at(0).ip();
    base_config[0]["msvc_upstreamMicroservices"][0]["nb_classOfInterest"] = upstreams.at(0).class_of_interest();

    //adjust downstreams
    int i = 0;
    // BUG SUSPECT: back() is only the last sender, not all senders
    for (auto &downstream: base_config.back()["msvc_dnstreamMicroservices"]) {
        downstream["nb_name"] = downstreams.at(i).name();
        downstream["nb_link"] = downstreams.at(i).ip();
        downstream["nb_classOfInterest"] = downstreams.at(i++).class_of_interest();
    }

    start_config["container"] = base_config;
    int control_port = CONTAINER_BASE_PORT + containers.size();
    runDocker(executable, name, to_string(start_config), control_port);
    std::string target = absl::StrFormat("%s:%d", "localhost", control_port);
    containers[name] = {{},
                        InDeviceCommunication::NewStub(grpc::CreateChannel(target, grpc::InsecureChannelCredentials())),
                        {}, new CompletionQueue(), 0, (model != ModelType::DataSource)};
    return true;
}

void DeviceAgent::StopContainer(const ContainerHandle &container, bool forced) {
    indevicecommunication::Signal request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container.stub->AsyncStopExecution(&context, request, container.cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(container.cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void DeviceAgent::UpdateContainerSender(const std::string &name, const std::string &dwnstr, const std::string &ip,
                                        const int &port) {
    Connection request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_name(dwnstr);
    request.set_ip(ip);
    request.set_port(port);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            containers[name].stub->AsyncUpdateSender(&context, request, containers[name].cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(containers[name].cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
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
        if (sysinfo(&sys_info) != 0) {
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

    if (containers.empty()) {
        return;
    }
    for (auto &container: containers) {
        if (container.second.reportMetrics) {
            LightMetrics *metrics = request.add_metrics();
            metrics->set_name(container.first);
            for (auto &size: container.second.queuelengths) {
                metrics->add_queue_size(size);
            }
            metrics->set_request_rate(container.second.metrics.requestRate);
        }
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            controller_stub->AsyncSendLightMetrics(&context, request, controller_sending_cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(controller_sending_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

void DeviceAgent::ReportFullMetrics() {
    FullMetricsList request;
    EmptyMessage reply;
    ClientContext context;
    Status status;

    for (auto &container: containers) {
        if (container.second.reportMetrics) {
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
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            controller_stub->AsyncSendFullMetrics(&context, request, controller_sending_cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(controller_sending_cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
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
    new UpdateDownstreamRequestHandler(&controller_service, controller_cq.get(), this);
    new StopMicroserviceRequestHandler(&controller_service, controller_cq.get(), this);
    while (running) {
        void *tag;
        bool ok;
        if (!controller_cq->Next(&tag, &ok)) {
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
                if (container.second.reportMetrics && container.second.pid != 0) {
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
            }
            ReportDeviceStatus();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
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
        device_agent->profiler->addPid(request.pid());
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
        bool success = device_agent->CreateContainer(static_cast<ModelType>(request.model()), request.name(),
                                                     request.bath_size(), request.slo(), request.upstream(),
                                                     request.downstream());
        if (!success) {
            status = FINISH;
            responder.Finish(reply, Status::CANCELLED, this);
        } else {
            status = FINISH;
            responder.Finish(reply, Status::OK, this);
        }
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
        if (device_agent->containers.find(request.name()) == device_agent->containers.end()) {
            status = FINISH;
            responder.Finish(reply, Status::CANCELLED, this);
            return;
        }
        device_agent->StopContainer(device_agent->containers[request.name()], request.forced());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::UpdateDownstreamRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateDownstream(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StopMicroserviceRequestHandler(service, cq, device_agent);
        device_agent->UpdateContainerSender(request.name(), request.downstream_name(), request.ip(), request.port());
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
    if (type == "server")
        deviceType = DeviceType::Server;
    else if (type == "edge")
        deviceType = DeviceType::Edge;
    else {
        std::cerr << "Invalid device type" << std::endl;
        exit(1);
    }

    DeviceAgent *agent = new DeviceAgent(controller_url, name, deviceType);
    while (agent->isRunning()) {
        std::string command;
        std::cin >> command;
        if (command == "exit") {
            break;
        }
    }
    delete agent;
    return 0;
}