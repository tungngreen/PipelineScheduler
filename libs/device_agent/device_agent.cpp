#include "device_agent.h"

ABSL_FLAG(std::string, name, "", "name of the device");
ABSL_FLAG(std::string, device_type, "", "string that identifies the device type");
ABSL_FLAG(std::string, controller_url, "", "string that identifies the controller url without port!");
ABSL_FLAG(uint16_t, dev_verbose, 0, "Verbosity level of the Device Agent.");
ABSL_FLAG(uint16_t, dev_loggingMode, 0, "Logging mode of the Device Agent. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, dev_logPath, "../logs", "Path to the log dir for the Device Agent.");
ABSL_FLAG(uint16_t, dev_port_offset, 0, "port offset for starting the control communication");

std::string getHostIP() {
    struct ifaddrs *ifAddrStruct = nullptr;
    struct ifaddrs *ifa = nullptr;
    void *tmpAddrPtr = nullptr;

    getifaddrs(&ifAddrStruct);

    for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }
        // check it is IP4
        if (ifa->ifa_addr->sa_family == AF_INET) {
            tmpAddrPtr = &((struct sockaddr_in *) ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
            if (std::strcmp(ifa->ifa_name, "lo") != 0) { // exclude loopback
                freeifaddrs(ifAddrStruct);
                return {addressBuffer};
            }
        }
    }
    if (ifAddrStruct != nullptr) freeifaddrs(ifAddrStruct);
    return "";
}

DeviceAgent::DeviceAgent() {
    dev_name = absl::GetFlag(FLAGS_name);
    std::string type = absl::GetFlag(FLAGS_device_type);
    if (type == "server") {
        dev_type = SystemDeviceType::Server;
    } else if (type == "nxavier") {
        dev_type = SystemDeviceType::NXXavier;
    } else if (type == "agxavier") {
        dev_type = SystemDeviceType::AGXXavier;
    } else if (type == "orinano") {
        dev_type = SystemDeviceType::OrinNano;
    }
    else {
        std::cerr << "Invalid device type, use [server, nxavier, agxavier, orinano]" << std::endl;
        exit(1);
    }
    dev_port_offset = absl::GetFlag(FLAGS_dev_port_offset);
    dev_loggingMode = absl::GetFlag(FLAGS_dev_loggingMode);
    dev_verbose = absl::GetFlag(FLAGS_dev_verbose);
    dev_logPath = absl::GetFlag(FLAGS_dev_logPath);
    deploy_mode = absl::GetFlag(FLAGS_deploy_mode);

    containers = std::map<std::string, DevContainerHandle>();

    dev_metricsServerConfigs.from_json(json::parse(std::ifstream("../jsons/metricsserver.json")));
    dev_metricsServerConfigs.user = "device_agent";
    dev_metricsServerConfigs.password = "agent";
    dev_metricsServerConn = connectToMetricsServer(dev_metricsServerConfigs, "Device_agent");

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    std::string server_address = absl::StrFormat( "%s:%d", "0.0.0.0", DEVICE_CONTROL_PORT + dev_port_offset);
    ServerBuilder controller_builder;
    controller_builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    controller_builder.RegisterService(&controller_service);
    controller_cq = controller_builder.AddCompletionQueue();
    controller_server = controller_builder.BuildAndStart();
}

DeviceAgent::DeviceAgent(const std::string &controller_url) : DeviceAgent() {
    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", INDEVICE_CONTROL_PORT + dev_port_offset);
    ServerBuilder device_builder;
    device_builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    device_builder.RegisterService(&device_service);
    device_cq = device_builder.AddCompletionQueue();
    device_server = device_builder.BuildAndStart();

    server_address = absl::StrFormat("%s:%d", controller_url, CONTROLLER_BASE_PORT + dev_port_offset);
    controller_stub = ControlMessages::NewStub(
            grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));
    controller_sending_cq = new CompletionQueue();

    dev_profiler = new Profiler({});
    Ready(getHostIP(), dev_type);

    dev_logPath += "/" + dev_experiment_name;
    std::filesystem::create_directories(
            std::filesystem::path(dev_logPath)
    );

    dev_logPath += "/" + dev_system_name;
    std::filesystem::create_directories(
            std::filesystem::path(dev_logPath)
    );

    setupLogger(
            dev_logPath,
            "device_agent",
            dev_loggingMode,
            dev_verbose,
            dev_loggerSinks,
            dev_logger
    );

    dev_metricsServerConfigs.schema = abbreviate(dev_experiment_name + "_" + dev_system_name);
    dev_hwMetricsTableName =  dev_metricsServerConfigs.schema + "." + abbreviate(dev_experiment_name + "_" + dev_name) + "_hw";
    dev_networkTableName = dev_metricsServerConfigs.schema + "." + abbreviate(dev_experiment_name + "_" + dev_name) + "_netw";

    if (!tableExists(*dev_metricsServerConn, dev_metricsServerConfigs.schema, dev_networkTableName)) {
        std::string sql = "CREATE TABLE IF NOT EXISTS " + dev_networkTableName + " ("
                                                                                    "timestamps BIGINT NOT NULL, "
                                                                                    "sender_host TEXT NOT NULL, "
                                                                                    "p95_transfer_duration_us BIGINT NOT NULL, "
                                                                                    "p95_total_package_size_b INTEGER NOT NULL)";

        pushSQL(*dev_metricsServerConn, sql);

        sql = "GRANT ALL PRIVILEGES ON " + dev_networkTableName + " TO " + "controller, container_agent" + ";";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "SELECT create_hypertable('" + dev_networkTableName + "', 'timestamps', if_not_exists => TRUE);";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "CREATE INDEX ON " + dev_networkTableName + " (timestamps);";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "CREATE INDEX ON " + dev_networkTableName + " (sender_host);";
        pushSQL(*dev_metricsServerConn, sql);
    }

    if (!tableExists(*dev_metricsServerConn, dev_metricsServerConfigs.schema, dev_hwMetricsTableName)) {
        std::string sql = "CREATE TABLE IF NOT EXISTS " + dev_hwMetricsTableName + " ("
                                                                                    "   timestamps BIGINT NOT NULL,"
                                                                                    "   cpu_usage INT," // percentage (1-100)
                                                                                    "   mem_usage INT,"; // Megabytes
        for (auto i = 0; i < dev_numCudaDevices; i++) {
            sql += "gpu_" + std::to_string(i) + "_usage INT," // percentage (1-100)
                   "gpu_" + std::to_string(i) + "_mem_usage INT,"; // Megabytes
        };
        sql += "   PRIMARY KEY (timestamps)"
                                                                                    ");";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "GRANT ALL PRIVILEGES ON " + dev_hwMetricsTableName + " TO " + "controller, container_agent" + ";";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "SELECT create_hypertable('" + dev_hwMetricsTableName + "', 'timestamps', if_not_exists => TRUE);";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "CREATE INDEX ON " + dev_hwMetricsTableName + " (timestamps);";
        pushSQL(*dev_metricsServerConn, sql);
    }

    running = true;
    threads = std::vector<std::thread>();
    threads.emplace_back(&DeviceAgent::HandleDeviceRecvRpcs, this);
    threads.emplace_back(&DeviceAgent::HandleControlRecvRpcs, this);
    for (auto &thread: threads) {
        thread.detach();
    }
}

void DeviceAgent::collectRuntimeMetrics() {
    std::string sql;
    auto timeNow = std::chrono::high_resolution_clock::now();
    if (timeNow > dev_metricsServerConfigs.nextMetricsReportTime) {
        dev_metricsServerConfigs.nextMetricsReportTime = timeNow + std::chrono::milliseconds(
                dev_metricsServerConfigs.metricsReportIntervalMillisec);
    }

    if (timeNow > dev_metricsServerConfigs.nextHwMetricsScrapeTime) {
        dev_metricsServerConfigs.nextHwMetricsScrapeTime = timeNow + std::chrono::milliseconds(
                dev_metricsServerConfigs.hwMetricsScrapeIntervalMillisec);
    }
    while (running) {
        auto metricsStopwatch = Stopwatch();
        metricsStopwatch.start();
        auto startTime = metricsStopwatch.getStartTime();
        uint64_t scrapeLatencyMillisec = 0;
        uint64_t timeDiff;

        if (timePointCastMillisecond(startTime) >=
            timePointCastMillisecond(dev_metricsServerConfigs.nextHwMetricsScrapeTime)) {
            std::vector<Profiler::sysStats> stats = dev_profiler->reportDeviceStats();

            DeviceHardwareMetrics metrics;
            metrics.timestamp = std::chrono::high_resolution_clock::now();
            metrics.cpuUsage = stats[0].cpuUsage;
            metrics.memUsage = stats[0].memoryUsage;
            metrics.rssMemUsage = stats[0].rssMemory;
            for (unsigned int i = 0; i < stats.size(); i++) {
                metrics.gpuUsage.emplace_back(stats[i].gpuUtilization);
                metrics.gpuMemUsage.emplace_back(stats[i].gpuMemoryUsage);
            }
            dev_runtimeMetrics.emplace_back(metrics);
            for (auto &container: containers) {
                if (container.second.pid > 0) {
                    Profiler::sysStats stats = dev_profiler->reportAtRuntime(container.second.pid, container.second.pid);
                    container.second.hwMetrics = {stats.cpuUsage, stats.memoryUsage, stats.rssMemory, stats.gpuUtilization,
                                    stats.gpuMemoryUsage};
                    spdlog::get("container_agent")->trace("{0:s} SCRAPE hardware metrics. Latency {1:d}ms.",
                                                        dev_name,
                                                        scrapeLatencyMillisec);
                }
            }
            metricsStopwatch.stop();
            scrapeLatencyMillisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
            dev_metricsServerConfigs.nextHwMetricsScrapeTime = std::chrono::high_resolution_clock::now() +
                                                                        std::chrono::milliseconds(
                                                                                dev_metricsServerConfigs.hwMetricsScrapeIntervalMillisec -
                                                                                scrapeLatencyMillisec);
        }

        metricsStopwatch.reset();
        metricsStopwatch.start();
        startTime = metricsStopwatch.getStartTime();
        if (timePointCastMillisecond(startTime) >=
            timePointCastMillisecond(dev_metricsServerConfigs.nextMetricsReportTime)) {

            if (dev_runtimeMetrics.empty()) {
                spdlog::get("container_agent")->trace("{0:s} No runtime metrics to push to the database.", dev_name);
                dev_metricsServerConfigs.nextMetricsReportTime = std::chrono::high_resolution_clock::now() +
                                                                 std::chrono::milliseconds(
                                                                         dev_metricsServerConfigs.metricsReportIntervalMillisec);
                continue;
            }
            sql = "INSERT INTO " + dev_hwMetricsTableName +
                  " (timestamps, cpu_usage, mem_usage";

            for (int i = 0; i < dev_numCudaDevices; i++) {
                sql += ", gpu_" + std::to_string(i) + "_usage, gpu_" + std::to_string(i) + "_mem_usage";
            }
            sql += ") VALUES ";
            for (const auto& entry : dev_runtimeMetrics) {
                sql += absl::StrFormat("(%s, %d, %d", timePointToEpochString(entry.timestamp),
                    entry.cpuUsage, entry.memUsage);
                for (int i = 0; i < dev_numCudaDevices; i++) {
                    sql += absl::StrFormat(", %d, %d", entry.gpuUsage[i], entry.gpuMemUsage[i]);
                }
                sql += "),";
            }
            sql.pop_back();
            dev_runtimeMetrics.clear();
            pushSQL(*dev_metricsServerConn, sql);
            spdlog::get("container_agent")->trace("{0:s} pushed device hardware metrics to the database.", dev_name);

            dev_metricsServerConfigs.nextMetricsReportTime = std::chrono::high_resolution_clock::now() +
                                                             std::chrono::milliseconds(
                                                                     dev_metricsServerConfigs.metricsReportIntervalMillisec);
            if (dev_type == SystemDeviceType::Server) {
                std::thread t(&DeviceAgent::ContainersLifeCheck, this);
                t.detach();
            }
        }

        metricsStopwatch.stop();
        auto reportLatencyMillisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
        ClockType nextTime;
        nextTime = std::min(dev_metricsServerConfigs.nextMetricsReportTime,
                            dev_metricsServerConfigs.nextHwMetricsScrapeTime);
        timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextTime - std::chrono::high_resolution_clock::now()).count();
        std::chrono::milliseconds sleepPeriod(timeDiff - (reportLatencyMillisec) + 2);
        spdlog::get("container_agent")->trace("{0:s} Container Agent's Metric Reporter sleeps for {1:d} milliseconds.", dev_name, sleepPeriod.count());
        std::this_thread::sleep_for(sleepPeriod);
    }
}

void DeviceAgent::testNetwork(float min_size, float max_size, int num_loops) {
    spdlog::get("container_agent")->info("Testing network with min size: {}, max size: {}, num loops: {}",
                                         min_size, max_size, num_loops);
    ClockType timestamp;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist = std::normal_distribution<float>((min_size + max_size) / 2, (max_size - min_size) / 6);
    std::vector<char> data;
    data.reserve(static_cast<size_t>(max_size));
    for (int i = 0; i < max_size + 1; i++) {
        data.push_back('x');
    }

    for (int i = 0; i < num_loops; i++) {
        DummyMessage request;
        EmptyMessage reply;
        ClientContext context;
        Status status;
        int size = std::abs((int) dist(gen));
        timestamp = std::chrono::high_resolution_clock::now();
        request.set_origin_name(dev_name);
        request.set_gen_time(std::chrono::duration_cast<TimePrecisionType>(timestamp.time_since_epoch()).count());
        spdlog::get("container_agent")->debug("Sending data of size: {}", size);
        request.set_data(data.data(), size);
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                controller_stub->AsyncSendDummyData(&context, request, controller_sending_cq));

        rpc->Finish(&reply, &status, (void *)1);
        void *got_tag;
        bool ok = false;
        if (controller_sending_cq != nullptr) GPR_ASSERT(controller_sending_cq->Next(&got_tag, &ok));
    }
    spdlog::get("container_agent")->info("Network test completed");
}

bool DeviceAgent::CreateContainer(ContainerConfig &c) {
    spdlog::get("container_agent")->info("Creating container: {}", c.name());
    try {
        std::string command = runDocker(c.executable(), c.name(), c.json_config(), c.device(), c.control_port());
        if (command == "") {
            return false;
        }
        std::string target = absl::StrFormat("%s:%d", "localhost", c.control_port());
        if (c.name().find("sink") != std::string::npos) {
            return true;
        }
        std::lock_guard<std::mutex> lock(containers_mutex);
        containers[c.name()] = {InDeviceCommunication::NewStub(
                grpc::CreateChannel(target, grpc::InsecureChannelCredentials())),
                                new CompletionQueue(), static_cast<unsigned int>(c.control_port()), 0, command, {}};
        return true;
    } catch (std::exception &e) {
        spdlog::get("container_agent")->error("Error creating container: {}", e.what());
        return false;
    }
}

void DeviceAgent::ContainersLifeCheck() {
    std::lock_guard<std::mutex> lock(containers_mutex);
    for (auto &container: containers) {
        if (container.second.pid == 0) continue;
        EmptyMessage request;
        EmptyMessage reply;
        ClientContext context;
        Status status;
        CompletionQueue *cq = container.second.cq;
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                container.second.stub->AsyncKeepAlive(&context, request, cq));

        rpc->Finish(&reply, &status, (void *)1);
        void *got_tag;
        bool ok = false;
        if (cq != nullptr) GPR_ASSERT(cq->Next(&got_tag, &ok));
        if (!status.ok()){
            container.second.pid = 0;
            runDocker(container.second.startCommand);
        }
    }
}

void DeviceAgent::StopContainer(const DevContainerHandle &container, bool forced) {
    indevicecommunication::Signal request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container.stub->AsyncStopExecution(&context, request, container.cq));

    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (container.cq != nullptr) GPR_ASSERT(container.cq->Next(&got_tag, &ok));
}

void DeviceAgent::UpdateContainerSender(int mode, const std::string &cont_name, const std::string &dwnstr,
                                        const std::string &ip, const int &port) {
    Connection request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_mode(mode);
    request.set_name(dwnstr);
    request.set_ip(ip);
    request.set_port(port);

    //check if cont_name is in containers
    if (containers.find(cont_name) == containers.end()) {
        spdlog::get("container_agent")->error("UpdateSender: Container {} not found!", cont_name);
        return;
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            containers[cont_name].stub->AsyncUpdateSender(&context, request, containers[cont_name].cq));

    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (containers[cont_name].cq != nullptr) GPR_ASSERT(containers[cont_name].cq->Next(&got_tag, &ok));
}

void DeviceAgent::SyncDatasources(const std::string &cont_name, const std::string &dsrc) {
    indevicecommunication::Int32 request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_value(containers[dsrc].port);
    //check if cont_name is in containers
    if (containers.find(cont_name) == containers.end()) {
        spdlog::get("container_agent")->error("SyncDatasources: Container {} not found!", cont_name);
        return;
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            containers[cont_name].stub->AsyncSyncDatasources(&context, request, containers[cont_name].cq));

    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (containers[cont_name].cq != nullptr) GPR_ASSERT(containers[cont_name].cq->Next(&got_tag, &ok));
}

void DeviceAgent::Ready(const std::string &ip, SystemDeviceType type) {
    ConnectionConfigs request;
    SystemInfo reply;
    ClientContext context;
    Status status;
    int processing_units;
    request.set_device_name(dev_name);
    request.set_device_type(type);
    request.set_ip_address(ip);
    if (type == SystemDeviceType::Server) {
        processing_units = dev_profiler->getGpuCount();
        request.set_processors(processing_units);
        for (auto &mem: dev_profiler->getGpuMemory(processing_units)) {
            request.add_memory(mem);
        }
    } else {
        struct sysinfo sys_info;
        if (sysinfo(&sys_info) != 0) {
            spdlog::get("container_agent")->error("sysinfo call failed!");
            exit(1);
        }
        processing_units = 1;
        request.set_processors(processing_units);
        request.add_memory(sys_info.totalram * sys_info.mem_unit / 1000000);
    }
    dev_numCudaDevices = processing_units;

    std::unique_ptr<ClientAsyncResponseReader<SystemInfo>> rpc(
            controller_stub->AsyncAdvertiseToController(&context, request, controller_sending_cq));

    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (controller_sending_cq != nullptr) GPR_ASSERT(controller_sending_cq->Next(&got_tag, &ok));
    if (!status.ok()) {
        spdlog::error("Ready RPC failed with code: {} and message: {}", status.error_code(), status.error_message());
        exit(1);
    }
    dev_system_name = reply.name();
    dev_experiment_name = reply.experiment();
}

void DeviceAgent::HandleDeviceRecvRpcs() {
    new ReportStartRequestHandler(&device_service, device_cq.get(), this);
    void *tag;
    bool ok;
    while (running) {
        if (!device_cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void DeviceAgent::HandleControlRecvRpcs() {
    new ExecuteNetworkTestRequestHandler(&controller_service, controller_cq.get(), this);
    new StartContainerRequestHandler(&controller_service, controller_cq.get(), this);
    new UpdateDownstreamRequestHandler(&controller_service, controller_cq.get(), this);
    new UpdateBatchsizeRequestHandler(&controller_service, controller_cq.get(), this);
    new UpdateResolutionRequestHandler(&controller_service, controller_cq.get(), this);
    new UpdateTimeKeepingRequestHandler(&controller_service, controller_cq.get(), this);
    new StopContainerRequestHandler(&controller_service, controller_cq.get(), this);
    new ShutdownRequestHandler(&controller_service, controller_cq.get(), this);
    void *tag;
    bool ok;
    while (running) {
        if (!controller_cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

int getContainerProcessPid(std::string container_name_or_id) {
    std::string cmd = "docker inspect --format '{{.State.Pid}}' " + container_name_or_id;
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    try {
        return std::stoi(result);
    } catch (std::exception &e) {
        return 0;
    }
}

void DeviceAgent::ReportStartRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestReportMsvcStart(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new ReportStartRequestHandler(service, cq, device_agent);

        int pid = getContainerProcessPid(request.msvc_name());
        device_agent->containers[request.msvc_name()].pid = pid;
        device_agent->dev_profiler->addPid(pid);
        spdlog::get("container_agent")->info("Received start report from {} with pid: {}", request.msvc_name(), pid);
        reply.set_pid(pid);
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::ExecuteNetworkTestRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestExecuteNetworkTest(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new ExecuteNetworkTestRequestHandler(service, cq, device_agent);
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
        device_agent->testNetwork((float) request.min(), (float) request.max(), request.repetitions());
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::StartContainerRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStartContainer(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StartContainerRequestHandler(service, cq, device_agent);
        bool success = device_agent->CreateContainer(request);
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

void DeviceAgent::StopContainerRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStopContainer(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new StopContainerRequestHandler(service, cq, device_agent);
        if (device_agent->containers.find(request.name()) == device_agent->containers.end()) {
            if (request.name().find("sink") != std::string::npos) {
                std::string command = "docker stop " + request.name();
                int status = system(command.c_str());
                device_agent->containers.erase(request.name());
                spdlog::get("container_agent")->info("Stopped container: {} with status: {}", request.name(), status);
            } else {
                spdlog::get("container_agent")->error("Container {} not found for deletion!", request.name());
                status = FINISH;
                responder.Finish(reply, Status::CANCELLED, this);
                return;
            }
        } else {
            unsigned int pid = device_agent->containers[request.name()].pid;
            device_agent->containers[request.name()].pid = 0;
            spdlog::get("container_agent")->info("Stopping container: {}", request.name());
            DeviceAgent::StopContainer(device_agent->containers[request.name()], request.forced());
            std::lock_guard<std::mutex> lock(device_agent->containers_mutex);
            device_agent->containers.erase(request.name());
            device_agent->dev_profiler->removePid(pid);
        }
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
        new UpdateDownstreamRequestHandler(service, cq, device_agent);
        device_agent->UpdateContainerSender(request.mode(), request.name(), request.downstream_name(), request.ip(), request.port());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::SyncDatasourceRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSyncDatasource(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new SyncDatasourceRequestHandler(service, cq, device_agent);
        device_agent->SyncDatasources(request.name(), request.downstream_name());
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::UpdateBatchsizeRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateBatchSize(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateBatchsizeRequestHandler(service, cq, device_agent);
        ClientContext context;
        Status state;
        indevicecommunication::Int32 bs;
        bs.set_value(request.value().at(0));

        //check if cont_name is in containers
        if (device_agent->containers.find(request.name()) == device_agent->containers.end()) {
            spdlog::get("container_agent")->error("UpdateBatchSize: Container {} not found!", request.name());
            status = FINISH;
            responder.Finish(reply, Status::CANCELLED, this);
            return;
        }
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                device_agent->containers[request.name()].stub->AsyncUpdateBatchSize(&context, bs,
                                                                                    device_agent->containers[request.name()].cq));

        rpc->Finish(&reply, &state, (void *)1);
        void *got_tag;
        bool ok = false;
        if (device_agent->containers[request.name()].cq != nullptr) GPR_ASSERT(device_agent->containers[request.name()].cq->Next(&got_tag, &ok));
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::UpdateResolutionRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateResolution(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateResolutionRequestHandler(service, cq, device_agent);
        ClientContext context;
        Status state;
        indevicecommunication::Dimensions dims;
        dims.set_channels(request.value().at(0));
        dims.set_height(request.value().at(1));
        dims.set_width(request.value().at(2));

        //check if cont_name is in containers
        if (device_agent->containers.find(request.name()) == device_agent->containers.end()) {
            spdlog::get("container_agent")->error("UpdateResolution: Container {} not found!", request.name());
            status = FINISH;
            responder.Finish(reply, Status::CANCELLED, this);
            return;
        }
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                device_agent->containers[request.name()].stub->AsyncUpdateResolution(&context, dims,
                                                                                    device_agent->containers[request.name()].cq));

        rpc->Finish(&reply, &state, (void *)1);
        void *got_tag;
        bool ok = false;
        if (device_agent->containers[request.name()].cq != nullptr) GPR_ASSERT(device_agent->containers[request.name()].cq->Next(&got_tag, &ok));
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::UpdateTimeKeepingRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateTimeKeeping(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateTimeKeepingRequestHandler(service, cq, device_agent);
        responder.Finish(reply, Status::OK, this);

        ClientContext context;
        Status state;
        indevicecommunication::TimeKeeping tk;
        tk.set_slo(request.slo());
        tk.set_time_budget(request.time_budget());
        tk.set_start_time(request.start_time());
        tk.set_end_time(request.end_time());
        tk.set_local_duty_cycle(request.local_duty_cycle());
        tk.set_cycle_start_time(request.cycle_start_time());

        //check if cont_name is in containers
        if (device_agent->containers.find(request.name()) == device_agent->containers.end()) {
            spdlog::get("container_agent")->error("UpdateTimeKeeping: Container {} not found!", request.name());
            status = FINISH;
            return;
        }
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                device_agent->containers[request.name()].stub->AsyncUpdateTimeKeeping(&context, tk,
                                                                                     device_agent->containers[request.name()].cq));

        rpc->Finish(&reply, &state, (void *)1);
        void *got_tag;
        bool ok = false;
        if (device_agent->containers[request.name()].cq != nullptr) GPR_ASSERT(device_agent->containers[request.name()].cq->Next(&got_tag, &ok));
        status = FINISH;
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DeviceAgent::ShutdownRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestShutdown(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        device_agent->running = false;
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
        device_agent->controller_cq->Shutdown();
        device_agent->device_cq->Shutdown();
    }
}

// Function to run the bash script with parameters from a JSON file
void DeviceAgent::limitBandwidth(const std::string& scriptPath, const std::string& jsonFilePath) {
    // Read JSON file
    std::ifstream json_file(jsonFilePath);
    if (!json_file.is_open()) {
        std::cerr << "Failed to open " << jsonFilePath << std::endl;
        return;
    }

    json config;
    json_file >> config;

    std::string interface = config["interface"];
    auto bandwidth_limits = config["bandwidth_limits"];

    if (bandwidth_limits.empty()) {
        std::cerr << "No bandwidth limits found in the JSON file." << std::endl;
        return;
    }

    auto start = std::chrono::system_clock::now();

    uint64_t bwThresholdIndex = 0;

    ClockType nextThresholdSetTime = start + std::chrono::seconds(bandwidth_limits[bwThresholdIndex]["time"]); 
    while (isRunning()) {
        if (bwThresholdIndex >= bandwidth_limits.size()) {
            break;
        }
        if (std::chrono::system_clock::now() >= nextThresholdSetTime) {
            Stopwatch stopwatch;

            auto limit = bandwidth_limits[bwThresholdIndex];
            int mbps = limit["mbps"];

            // Build and execute the command
            std::string command = "sudo bash " + scriptPath + " " + interface + " " + std::to_string(mbps);
            spdlog::get("container_agent")->info("{0:s} Setting BW limit to {1:d}", dev_name, mbps);
            int result = system(command.c_str());
            spdlog::get("container_agent")->info("Command executed with result: {0:d}", result);

            if (bwThresholdIndex == bandwidth_limits.size() - 1) {
                break;
            }
            // TODO: resolve unsequenced modification and access to 'bwThresholdIndex'
            auto distanceToNext = bandwidth_limits[++bwThresholdIndex]["time"].get<int>() - bandwidth_limits[bwThresholdIndex - 1]["time"].get<int>();
            nextThresholdSetTime += std::chrono::seconds(distanceToNext);

            auto sleepTime = nextThresholdSetTime - std::chrono::system_clock::now();
            std::this_thread::sleep_for(sleepTime + std::chrono::nanoseconds(10000000));

        }
    }

    // QUANG: Remove the bandwidth limit
    std::cout << "Finished bandwidth limiting." << std::endl;
}