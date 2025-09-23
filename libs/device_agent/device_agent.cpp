#include "device_agent.h"

ABSL_FLAG(std::string, name, "", "name of the device");
ABSL_FLAG(std::string, device_type, "", "string that identifies the device type");
ABSL_FLAG(std::string, controller_url, "", "string that identifies the controller url without port!");
ABSL_FLAG(uint16_t, dev_verbose, 0, "Verbosity level of the Device Agent.");
ABSL_FLAG(uint16_t, dev_loggingMode, 0, "Logging mode of the Device Agent. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, dev_logPath, "../logs", "Path to the log dir for the Device Agent.");
ABSL_FLAG(uint16_t, dev_port_offset, 0, "port offset the deviceAgents ports to allow for co-located virtual nodes");
ABSL_FLAG(uint16_t, dev_system_port_offset, 0, "port offset of the whole system for starting the control communication");
ABSL_FLAG(uint16_t, dev_bandwidthLimitID, 0, "id at the end of bandwidth_limits{}.json file to indicate which should be used");
ABSL_FLAG(std::string, dev_networkInterface, "eth0", "optionally specify the network interface to use for bandwidth limiting");
ABSL_FLAG(int, dev_gpuID, -1, "GPU ID to use a specific GPU for containers (setting for virtual Devices)");

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
    if (type == "virtual") {
        dev_type = SystemDeviceType::Virtual;
    } else if (type == "server") {
        dev_type = SystemDeviceType::Server;
    } else if (type == "onprem") {
        dev_type = SystemDeviceType::OnPremise;
    } else if (type == "orinagx") {
        dev_type = SystemDeviceType::OrinAGX;
    } else if (type == "orinnx") {
        dev_type = SystemDeviceType::OrinNX;
    } else if (type == "orinano") {
        dev_type = SystemDeviceType::OrinNano;
    } else if (type == "agxavier") {
        dev_type = SystemDeviceType::AGXXavier;
    } else if (type == "nxavier") {
        dev_type = SystemDeviceType::NXXavier;
    } else {
        std::cerr << "Invalid device type, use [virtual, server, onprem, orinagx, orinnx, orinano, agxavier, nxavier]" << std::endl;
        exit(1);
    }
    dev_agent_port_offset = absl::GetFlag(FLAGS_dev_port_offset);
    dev_system_port_offset = absl::GetFlag(FLAGS_dev_system_port_offset) + dev_agent_port_offset;
    dev_gpuID = absl::GetFlag(FLAGS_dev_gpuID);
    dev_loggingMode = absl::GetFlag(FLAGS_dev_loggingMode);
    dev_verbose = absl::GetFlag(FLAGS_dev_verbose);
    dev_logPath = absl::GetFlag(FLAGS_dev_logPath);
    deploy_mode = absl::GetFlag(FLAGS_deploy_mode);

    containers = std::map<std::string, DevContainerHandle>();

    dev_metricsServerConfigs.from_json(json::parse(std::ifstream("../jsons/metricsserver.json")));
    dev_metricsServerConfigs.user = "device_agent";
    dev_metricsServerConfigs.password = "agent";
    dev_metricsServerConn = connectToMetricsServer(dev_metricsServerConfigs, "Device_agent");

    in_device_handlers = {
        {MSG_TYPE[MSVC_START_REPORT], std::bind(&DeviceAgent::ReceiveStartReport, this, std::placeholders::_1)},
        {MSG_TYPE[START_FL], std::bind(&DeviceAgent::ForwardFL, this, std::placeholders::_1)},
        {MSG_TYPE[BCEDGE_UPDATE], std::bind(&DeviceAgent::InferBCEdge, this, std::placeholders::_1)},
        {MSG_TYPE[CONTEXT_METRICS], std::bind(&DeviceAgent::ReceiveContainerMetrics, this, std::placeholders::_1)}
    };

    controller_handlers = {
        {MSG_TYPE[NETWORK_CHECK], std::bind(&DeviceAgent::testNetwork, this, std::placeholders::_1)},
        {MSG_TYPE[CONTAINER_START], std::bind(&DeviceAgent::CreateContainer, this, std::placeholders::_1)},
        {MSG_TYPE[ADJUST_UPSTREAM], std::bind(static_cast<void (DeviceAgent::*)(const std::string&)>
                                            (&DeviceAgent::UpdateContainerSender), this, std::placeholders::_1)},
        {MSG_TYPE[SYNC_DATASOURCES], std::bind(&DeviceAgent::SyncDatasources, this, std::placeholders::_1)},
        {MSG_TYPE[BATCH_SIZE_UPDATE], std::bind(&DeviceAgent::UpdateBatchSize, this, std::placeholders::_1)},
        {MSG_TYPE[RESOLUTION_UPDATE], std::bind(&DeviceAgent::UpdateResolution, this, std::placeholders::_1)},
        {MSG_TYPE[TIME_KEEPING_UPDATE], std::bind(&DeviceAgent::UpdateTimeKeeping, this, std::placeholders::_1)},
        {MSG_TYPE[CONTAINER_STOP], std::bind(static_cast<void (DeviceAgent::*)(const std::string&)>
                                            (&DeviceAgent::StopContainer), this, std::placeholders::_1)},
        {MSG_TYPE[RETURN_FL], std::bind(&DeviceAgent::ReturnFL, this, std::placeholders::_1)},
        {MSG_TYPE[DEVICE_SHUTDOWN], std::bind(&DeviceAgent::Shutdown, this, std::placeholders::_1)}
    };

    dev_totalBandwidthData = std::vector<BandwidthManager>();
    for (int id = 0; id <= 20; ++id) {
        BandwidthManager data;
        std::string filename = "../jsons/bandwidths/bandwidth_limits" + std::to_string(id) + ".json";
        if (!std::filesystem::exists(filename)) {
            dev_totalBandwidthData.push_back(data);
            continue;
        }
        std::ifstream json_file(filename);
        if (!json_file.is_open()) {
            dev_totalBandwidthData.push_back(data);
            continue;
        }
        json config = json::parse(json_file);
        for (auto &item : config["bandwidth_limits"]) {
            data.addLimit(item["time"].get<int>(), item["mbps"]);
        }
        data.prepare();
        dev_totalBandwidthData.push_back(data);
    }
    dev_startTime = std::chrono::high_resolution_clock::now();
}

DeviceAgent::DeviceAgent(const std::string &controller_url) : DeviceAgent() {
    in_device_ctx = context_t(dev_type == Server ? 2 : 1);
    std::string server_address = absl::StrFormat("tcp://*:%d", IN_DEVICE_RECEIVE_PORT + dev_system_port_offset);
    in_device_socket = socket_t(in_device_ctx, ZMQ_REP);
    in_device_socket.bind(server_address);
    in_device_socket.set(zmq::sockopt::rcvtimeo, 1000);
    server_address = absl::StrFormat("tcp://*:%d", IN_DEVICE_MESSAGE_QUEUE_PORT + dev_system_port_offset);
    in_device_message_queue = socket_t(in_device_ctx, ZMQ_PUB);
    in_device_message_queue.bind(server_address);
    in_device_message_queue.set(zmq::sockopt::sndtimeo, 100);

    controller_ctx = context_t(1);
    server_address = absl::StrFormat("tcp://%s:%d", controller_url, CONTROLLER_RECEIVE_PORT + dev_system_port_offset - dev_agent_port_offset);
    controller_socket = socket_t(controller_ctx, ZMQ_REQ);
    controller_socket.connect(server_address);
    server_address = absl::StrFormat("tcp://%s:%d", controller_url, CONTROLLER_MESSAGE_QUEUE_PORT + dev_system_port_offset  - dev_agent_port_offset);
    controller_message_queue = socket_t(controller_ctx, ZMQ_SUB);
    controller_message_queue.setsockopt(ZMQ_SUBSCRIBE, (dev_name + "|").c_str(), dev_name.size() + 1);
    controller_message_queue.connect(server_address);
    controller_message_queue.set(zmq::sockopt::rcvtimeo, 1000);

    dev_profiler = new Profiler({(unsigned int) getpid()}, "runtime");
    SystemInfo readyReply = Ready(getHostIP());

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
    if (dev_totalBandwidthData.size() <= absl::GetFlag(FLAGS_dev_bandwidthLimitID)) {
        spdlog::get("container_agent")->error("Invalid bandwidth limit ID, use [0, 20]! Defaulting to 0.");
        dev_bandwidthLimit = dev_totalBandwidthData[0];
    } else {
        dev_bandwidthLimit = dev_totalBandwidthData[absl::GetFlag(FLAGS_dev_bandwidthLimitID)];
    }

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
                   "gpu_" + std::to_string(i) + "_mem_usage INT," // Megabytes
                     "gpu_" + std::to_string(i) + "_power_consumption INT,"; // Milli-Watts
        };
        sql += "   PRIMARY KEY (timestamps));";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "GRANT ALL PRIVILEGES ON " + dev_hwMetricsTableName + " TO " + "controller, container_agent" + ";";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "SELECT create_hypertable('" + dev_hwMetricsTableName + "', 'timestamps', if_not_exists => TRUE);";
        pushSQL(*dev_metricsServerConn, sql);

        sql = "CREATE INDEX ON " + dev_hwMetricsTableName + " (timestamps);";
        pushSQL(*dev_metricsServerConn, sql);
    }

    if (dev_system_name == "bce") {
        if (dev_type == Virtual || dev_type == Server || dev_type == OnPremise) {
            dev_bcedge_agent = new BCEdgeAgent("/ssd0/tung/PipePlusPlus/models/bcedge/" + dev_name, 3500, torch::kF32, 0);
        } else {
            dev_bcedge_agent = new BCEdgeAgent("../../pipe/models/bcedge/" + dev_name, 3000000, torch::kF32, 0);
        }
    } else if (dev_system_name == "edvi") {
        dev_rlDecisionInterval = TimePrecisionType(200000);
        for (auto &el: readyReply.offloading_targets()) {
            edgevision_dwnstrList.push_back({el.name(), el.offloading_ip(), el.bandwidth_id()});
        }
        dev_edgevision_agent = new EdgeVisionAgent(dev_name, (int) edgevision_dwnstrList.size() - 1, torch::kF32, 0);
    }

    running = true;
    threads = std::vector<std::thread>();
    threads.emplace_back(&DeviceAgent::HandleDeviceMessages, this);
    threads.emplace_back(&DeviceAgent::HandleControlCommands, this);
    for (auto &thread: threads) {
        thread.detach();
    }
    dev_startTime = std::chrono::high_resolution_clock::now(); // update start_time compared to previous timestamp
}

void DeviceAgent::collectRuntimeMetrics() {
    std::string sql;
    std::vector<double> edgevisionBWs(std::max(1, (int) edgevision_dwnstrList.size() - 1));
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
            if (!containers.empty()) {
                metrics.timestamp = std::chrono::high_resolution_clock::now();
                metrics.cpuUsage = stats[0].cpuUsage;
                metrics.memUsage = stats[0].deviceMemoryUsage;
                for (unsigned int i = 0; i < stats.size(); i++) {
                    metrics.gpuUsage.emplace_back(stats[i].gpuUtilization);
                    metrics.gpuMemUsage.emplace_back(stats[i].gpuMemoryUsage);
                    metrics.powerConsumption.emplace_back(stats[i].energyConsumption);
                }
                dev_runtimeMetrics.emplace_back(metrics);
            }
            for (auto &container: containers) {
                if (container.second.pid > 0) {
                    Profiler::sysStats stats = dev_profiler->reportAtRuntime(container.second.pid, container.second.pid);
                    container.second.hwMetrics = {stats.cpuUsage, stats.processMemoryUsage, stats.rssMemory, stats.gpuUtilization,
                                                  stats.gpuMemoryUsage};
                    spdlog::get("container_agent")->trace("{0:s} SCRAPE hardware metrics. Latency {1:d}ms.",
                                                        dev_name, scrapeLatencyMillisec);
                }
            }
            metricsStopwatch.stop();
            scrapeLatencyMillisec = (uint64_t) std::ceil((float) metricsStopwatch.elapsed_microseconds() / 1000.f);
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
                                                                 std::chrono::milliseconds(dev_metricsServerConfigs.metricsReportIntervalMillisec);
                continue;
            }
            sql = "INSERT INTO " + dev_hwMetricsTableName +
                  " (timestamps, cpu_usage, mem_usage";

            for (int i = 0; i < dev_numCudaDevices; i++) {
                sql += ", gpu_" + std::to_string(i) + "_usage, gpu_" + std::to_string(i) + "_mem_usage" +
                       ", gpu_" + std::to_string(i) + "_power_consumption";;
            }
            sql += ") VALUES ";
            for (const auto& entry : dev_runtimeMetrics) {
                sql += absl::StrFormat("(%s, %d, %d", timePointToEpochString(entry.timestamp),
                    entry.cpuUsage, entry.memUsage);
                for (int i = 0; i < dev_numCudaDevices; i++) {
                    sql += absl::StrFormat(", %d, %d, %d", entry.gpuUsage[i], entry.gpuMemUsage[i], entry.powerConsumption[i]);
                }
                sql += "),";
            }
            sql.pop_back();
            dev_runtimeMetrics.clear();
            pushSQL(*dev_metricsServerConn, sql);
            spdlog::get("container_agent")->trace("{0:s} pushed device hardware metrics to the database.", dev_name);

            dev_metricsServerConfigs.nextMetricsReportTime = std::chrono::high_resolution_clock::now() +
                                                             std::chrono::milliseconds(dev_metricsServerConfigs.metricsReportIntervalMillisec);
            if (dev_type == SystemDeviceType::Server) {
                std::thread t(&DeviceAgent::ContainersLifeCheck, this);
                t.detach();
            }
        }

        if (dev_system_name == "edvi" && timePointCastMillisecond(startTime) >= timePointCastMillisecond(dev_nextRLDecisionTime)) {
            std::vector<DevContainerHandle*> local_downstreams = {};
            std::string roi_extractor = nullptr;
            for (auto &container: containers) {
                if (container.first.find("dnstr") != std::string::npos) {
                    local_downstreams.push_back(&container.second);
                }
            }
            EmptyMessage request;
            int i = 0, target = 0;
            for (auto *dnstr: local_downstreams) {
                if (i == 0) {
                    dev_edgevision_agent->rewardCallback(dnstr->contextMetrics.throughput(), dnstr->contextMetrics.drops(),
                                                         dnstr->contextMetrics.avg_latency());
                    int runtime = std::ceil(std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::high_resolution_clock::now() - dev_startTime).count());

                    for (auto &el: edgevision_dwnstrList) {
                        edgevisionBWs[i++] = dev_totalBandwidthData[el.bandwidth_id].getMbps(runtime);
                    }
                    dev_edgevision_agent->setState(dnstr->contextMetrics.arrival_rate(), dnstr->contextMetrics.queue_size(), 0, edgevisionBWs);
                    target = dev_edgevision_agent->runStep();
                    if (target == 0) {
                        spdlog::get("container_agent")->trace("EdgeVision decision: 0@localhost");
                    } else {
                        spdlog::get("container_agent")->trace("EdgeVision  decision: {1:d}@{2:s}", target,
                                                              edgevision_dwnstrList[target - 1].offloading_ip);
                    }
                }
                if (target == 0) {
                    UpdateContainerSender(AdjustUpstreamMode::Overwrite, "cont_people", dnstr->name, "localhost",
                                          dnstr->port + 5000, 1.0, "", std::chrono::duration_cast<TimePrecisionType>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(), 10);
                } else {
                    UpdateContainerSender(AdjustUpstreamMode::Overwrite, "cont_people", dnstr->name,
                          edgevision_dwnstrList[target - 1].offloading_ip, edgevision_dwnstrList[target - 1].offloading_port,
                          1.0, "", std::chrono::duration_cast<TimePrecisionType>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(), 10);
                }
            }
            dev_nextRLDecisionTime = std::chrono::high_resolution_clock::now() + dev_rlDecisionInterval;
        }


        metricsStopwatch.stop();
        auto reportLatencyMillisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
        ClockType nextTime;
        nextTime = std::min(dev_metricsServerConfigs.nextMetricsReportTime, dev_metricsServerConfigs.nextHwMetricsScrapeTime);
        if (dev_system_name == "edvi") {
            nextTime = std::min(nextTime, dev_nextRLDecisionTime);
        }
        timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextTime - std::chrono::high_resolution_clock::now()).count();
        std::chrono::milliseconds sleepPeriod(timeDiff - (reportLatencyMillisec) + 2);
        spdlog::get("container_agent")->trace("{0:s} Container Agent's Metric Reporter sleeps for {1:d} milliseconds.", dev_name, sleepPeriod.count());
        std::this_thread::sleep_for(sleepPeriod);
    }
}

void DeviceAgent::testNetwork(const std::string &msg) {
    LoopRange range;
    if (!range.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed start network test with msg: {}", msg);
        return;
    }
    spdlog::get("container_agent")->info("Testing network with min size: {}, max size: {}, num loops: {}",
                                         range.min(), range.max(), range.repetitions());
    ClockType timestamp;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist = std::normal_distribution<float>((range.min() + range.max()) / 2,
                                                                           (range.max() - range.min()) / 6);
    std::vector<char> data;
    data.reserve(static_cast<size_t>(range.max()));
    for (int i = 0; i < range.max() + 1; i++) {
        data.push_back('x');
    }

    for (int i = 0; i < range.repetitions(); i++) {
        DummyMessage request;
        int size = std::abs((int) dist(gen));
        timestamp = std::chrono::high_resolution_clock::now();
        request.set_origin_name(dev_name);
        request.set_gen_time(std::chrono::duration_cast<TimePrecisionType>(timestamp.time_since_epoch()).count());
        spdlog::get("container_agent")->debug("Sending data of size: {}", size);
        request.set_data(data.data(), size);
        std::string message = MSG_TYPE[DUMMY_DATA] + " " + request.SerializeAsString();
        message_t zmq_msg(message.size());
        memcpy(zmq_msg.data(), message.data(), message.size());
        if (!controller_socket.send(zmq_msg, send_flags::dontwait)) {
            continue;
        }
        message_t reply;
        if (!controller_socket.recv(reply)) { i--; }
    }
    spdlog::get("container_agent")->info("Network test completed");
}

void DeviceAgent::CreateContainer(const std::string &msg) {
    ContainerConfig c;
    if (!c.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed start container with msg: {}", msg);
        return;
    }
    spdlog::get("container_agent")->info("Creating container: {}", c.name());
    try {
        std::string command = runDocker(c.executable(), c.name(), c.json_config(), c.device(), c.control_port());
        std::string target = absl::StrFormat("%s:%d", "localhost", c.control_port());
        if (c.name().find("sink") != std::string::npos) {
            return;
        }
        std::vector<int> dims = {};
        for (auto &dim: c.input_shape()) {
            dims.push_back(dim);
        }
        std::lock_guard<std::mutex> lock(containers_mutex);
        containers[c.name()] = {c.name(), static_cast<unsigned int>(c.control_port()), 0, command,
                                static_cast<ModelType>(c.model_type()), dims, 1, {}};
        return;
    } catch (std::exception &e) {
        spdlog::get("container_agent")->error("Error creating container: {}", e.what());
        return;
    }
}

void DeviceAgent::ContainersLifeCheck() {
    std::lock_guard<std::mutex> lock(containers_mutex);
    for (auto &container: containers) {
        if (container.second.pid == 0) continue;
        if (system(("docker inspect -f '{{.State.Running}}' " + container.first).c_str()) != 0) {
            if (container.second.startCommand != "") {
                spdlog::get("container_agent")->error("Container {} is not running anymore, trying restart.", container.first);
                if (runDocker(container.second.startCommand) != 0) container.second.pid = 0;
            }
        }
    }
}

void DeviceAgent::StopContainer(const std::string &msg) {
    ContainerSignal request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed stopping container with msg: {}", msg);
        return;
    }
    StopContainer(request);
}

void DeviceAgent::StopContainer(ContainerSignal request) {
    if (containers.find(request.name()) == containers.end()) {
        if (request.name().find("sink") != std::string::npos) {
            std::string command = "docker stop " + request.name();
            int status = system(command.c_str());
            containers.erase(request.name());
            spdlog::get("container_agent")->info("Stopped container: {} with status: {}", request.name(), status);
        } else {
            spdlog::get("container_agent")->error("Container {} not found for deletion!", request.name());
            return;
        }
    } else {
        unsigned int pid = containers[request.name()].pid;
        containers[request.name()].pid = 0;
        spdlog::get("container_agent")->info("Stopping container: {}", request.name());
        sendMessageToContainer(request.name(), MSG_TYPE[CONTAINER_STOP], request.SerializeAsString());
        std::lock_guard<std::mutex> lock(containers_mutex);
        containers.erase(request.name());
        dev_profiler->removePid(pid);
    }
}

void DeviceAgent::UpdateContainerSender(const std::string &msg) {
    ContainerLink request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed update container sender with msg: {}", msg);
        return;
    }
    UpdateContainerSender(request.mode(), request.name(), request.downstream_name(), request.ip(), request.port(),
                          request.data_portion(), request.old_link(), request.timestamp(), request.offloading_duration());
}

void DeviceAgent::UpdateContainerSender(int mode, const std::string &cont_name, const std::string &dwnstr,
                                        const std::string &ip, const int &port, const float &data_portion,
                                        const std::string &old_link, const int64_t &timestamp,
                                        const int &offloading_duration) {
    Connection request;
    request.set_mode(mode);
    request.set_name(dwnstr);
    request.set_ip(ip);
    request.set_port(port);
    request.set_data_portion(data_portion);
    request.set_old_link(old_link);
    request.set_timestamp(timestamp);
    request.set_offloading_duration(offloading_duration);

    //check if cont_name is in containers
    if (containers.find(cont_name) == containers.end()) {
        spdlog::get("container_agent")->error("UpdateSender: Container {} not found!", cont_name);
        return;
    }
    sendMessageToContainer(cont_name, MSG_TYPE[UPDATE_SENDER], request.SerializeAsString());
}

void DeviceAgent::SyncDatasources(const std::string &msg) {
    ContainerLink link;
    if (!link.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed sync datasources with msg: {}", msg);
        return;
    }
    indevicemessages::Int32 request;
    request.set_value(containers[link.downstream_name()].port);
    //check if cont_name is in containers
    if (containers.find(link.name()) == containers.end()) {
        spdlog::get("container_agent")->error("SyncDatasources: Container {} not found!", link.name());
        return;
    }
    sendMessageToContainer(link.name(), MSG_TYPE[SYNC_DATASOURCES], request.SerializeAsString());
}

SystemInfo DeviceAgent::Ready(const std::string &ip) {
    ConnectionConfigs request;
    request.set_device_name(dev_name);
    request.set_device_type(dev_type);
    request.set_ip_address(ip);
    request.set_agent_port_offset(dev_agent_port_offset);
    if (dev_type == SystemDeviceType::Virtual) {
        auto processing_units = dev_profiler->getGpuCount();
        auto mem = dev_profiler->getGpuMemory(processing_units);
        dev_numCudaDevices = 1;
        request.set_processors(dev_numCudaDevices);
        request.add_memory(mem[std::max(0,dev_gpuID)]);
    } else if (dev_type == SystemDeviceType::Server) {
        dev_numCudaDevices = dev_profiler->getGpuCount();
        request.set_processors(dev_numCudaDevices);
        for (auto &mem: dev_profiler->getGpuMemory(dev_numCudaDevices)) {
            request.add_memory(mem);
        }
    } else if (dev_type == SystemDeviceType::OnPremise) {
        auto processing_units = dev_profiler->getGpuCount();
        auto mem = dev_profiler->getGpuMemory(processing_units);
        dev_numCudaDevices = 1;
        request.set_processors(dev_numCudaDevices);
        request.add_memory(mem[dev_gpuID]);
    } else {
        struct sysinfo sys_info;
        if (sysinfo(&sys_info) != 0) {
            spdlog::get("container_agent")->error("sysinfo call failed!");
            exit(1);
        }
        dev_numCudaDevices = 1;
        request.set_processors(dev_numCudaDevices);
        request.add_memory(sys_info.totalram * sys_info.mem_unit / 1000000);
    }
    std::string test = request.SerializeAsString();
    std::string msg = absl::StrFormat("%s %s", MSG_TYPE[DEVICE_ADVERTISEMENT], request.SerializeAsString());
    message_t zmq_msg(msg.size()), reply;
    memcpy(zmq_msg.data(), msg.data(), msg.size());
    if (!controller_socket.send(zmq_msg, send_flags::dontwait)){
        spdlog::error("Sending ready message failed! Is the Server running?");
        exit(1);
    }
    if (!controller_socket.recv(reply)) {
        spdlog::error("Ready message reply failed! Is the Server running correctly?");
        exit(1);
    }
    SystemInfo info;
    if (!info.ParseFromString(std::string(static_cast<char *>(reply.data()), reply.size()))) {
        spdlog::get("container_agent")->error("Failed to parse SystemInfo from reply: {}", std::string(static_cast<char *>(reply.data()), reply.size()));
        exit(1);
    }
    dev_system_name = info.name();
    dev_experiment_name = info.experiment();
    return info;
}

void DeviceAgent::HandleDeviceMessages() {
    while (running) {
        message_t message;
        if (in_device_socket.recv(message, recv_flags::none)) {
            std::string raw = message.to_string();
            std::istringstream iss(raw);
            std::string topic;
            iss >> topic;
            iss.get(); // skip the space after the topic
            std::string payload((std::istreambuf_iterator<char>(iss)),
                                std::istreambuf_iterator<char>());
            if (in_device_handlers.count(topic)) {
                in_device_handlers[topic](payload);
            } else {
                spdlog::get("container_agent")->error("Received unknown device topic: {}", topic);
            }
//        } else {
//            spdlog::get("container_agent")->trace("Device Communication Receive Timeout");
        }
    }
}

void DeviceAgent::HandleControlCommands() {
    while (running) {
        message_t message;
        if (controller_message_queue.recv(message, recv_flags::none)) {
            std::string raw = message.to_string();
            std::istringstream iss(raw);
            std::string topic, type;
            iss >> topic;
            iss >> type;
            iss.get(); // skip the space after the topic
            std::string payload((std::istreambuf_iterator<char>(iss)),
                                std::istreambuf_iterator<char>());
            if (controller_handlers.count(type)) {
                controller_handlers[type](payload);
            } else {
                spdlog::get("container_agent")->error("Received unknown controller type: {} (topic: {})", type, topic);
            }
//        } else {
//            spdlog::get("container_agent")->trace("Control Communication Receive Timeout");
        }
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

void DeviceAgent::ReceiveStartReport(const std::string &msg) {
    ProcessData request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Error receiving container start report with msg: {}", msg);
        return;
    }

    int pid = getContainerProcessPid(request.msvc_name());
    containers[request.msvc_name()].pid = pid;
    dev_profiler->addPid(pid);
    spdlog::get("container_agent")->info("Received start report from {} with pid: {}", request.msvc_name(), pid);
    ProcessData reply;
    reply.set_msvc_name(request.msvc_name());
    reply.set_pid(pid);
    in_device_socket.send(message_t(reply.SerializeAsString()), send_flags::dontwait);
}

void DeviceAgent::ForwardFL(const std::string &msg) {
    FlData request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed stopping container with msg: {}", msg);
        return;
    }
    request.set_device_name(dev_name);

    std::string message = absl::StrFormat("%s %s", MSG_TYPE[START_FL], request.SerializeAsString());
    message_t zmq_msg(message.size()), reply;
    memcpy(zmq_msg.data(), message.data(), message.size());
    if (!controller_socket.send(zmq_msg, send_flags::dontwait)) {
        spdlog::get("container_agent")->error("Failed to send FL data to controller.");
        in_device_socket.send(message_t("error"), send_flags::dontwait);
        return;
    }
    if (controller_socket.recv(reply)) {
        spdlog::get("container_agent")->info("Forwarded FL data to controller successfully.");
        in_device_socket.send(reply, send_flags::dontwait);
    } else {
        spdlog::get("container_agent")->error("Failed to receive reply from controller for FL data forwarding.");
        in_device_socket.send(message_t("error"), send_flags::dontwait);
    }
}

void DeviceAgent::InferBCEdge(const std::string &msg) {
    BCEdgeData request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed InferBCEdge with msg: {}", msg);
        return;
    }

    DevContainerHandle *node = &containers[request.msvc_name()];
    dev_bcedge_agent->rewardCallback(request.throughput(), request.latency(), request.slo(), node->hwMetrics.gpuMemUsage);
    dev_bcedge_agent->setState(node->modelType, node->dataShape, request.slo());
    int batching, scaling, memory;
    std::tie(batching, scaling, memory) = dev_bcedge_agent->runStep();

    BCEdgeConfig reply;
    reply.set_batch_size(batching);
    node->instances = scaling;
    reply.set_shared_mem_config(memory);
    in_device_socket.send(message_t(reply.SerializeAsString()), send_flags::dontwait);
}

void DeviceAgent::ReceiveContainerMetrics(const std::string &msg) {
    ContainerMetrics request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed to receive container metrics with msg: {}", msg);
        return;
    }

    //check if cont_name is in containers
    if (containers.find(request.name()) == containers.end()) {
        spdlog::get("container_agent")->error("ReceiveContainerMetrics: Container {} not found!", request.name());
        return;
    }

    containers[request.name()].contextMetrics = request;
}

void DeviceAgent::UpdateBatchSize(const std::string &msg) {
    ContainerInts request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed UpdateBatchSize container with msg: {}", msg);
        return;
    }

    indevicemessages::Int32 bs;
    bs.set_value(request.value().at(0));

    //check if cont_name is in containers
    if (containers.find(request.name()) == containers.end()) {
        spdlog::get("container_agent")->error("UpdateBatchSize: Container {} not found!", request.name());
        return;
    }
    sendMessageToContainer(request.name(), MSG_TYPE[BATCH_SIZE_UPDATE], bs.SerializeAsString());
}

void DeviceAgent::UpdateResolution(const std::string &msg) {
    ContainerInts request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed UpdateResolution container with msg: {}", msg);
        return;
    }

    indevicemessages::Dimensions dims;
    dims.set_channels(request.value().at(0));
    dims.set_height(request.value().at(1));
    dims.set_width(request.value().at(2));

    //check if cont_name is in containers
    if (containers.find(request.name()) == containers.end()) {
        spdlog::get("container_agent")->error("UpdateResolution: Container {} not found!", request.name());
        return;
    }
    sendMessageToContainer(request.name(), MSG_TYPE[RESOLUTION_UPDATE], dims.SerializeAsString());
}

void DeviceAgent::UpdateTimeKeeping(const std::string &msg) {
    TimeKeeping request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed UpdateTimeKeeping container with msg: {}", msg);
        return;
    }

    //check if cont_name is in containers
    if (containers.find(request.name()) == containers.end()) {
        spdlog::get("container_agent")->error("UpdateTimeKeeping: Container {} not found!", request.name());
        return;
    }
    sendMessageToContainer(request.name(), MSG_TYPE[TIME_KEEPING_UPDATE], msg);
}

void DeviceAgent::ReturnFL(const std::string &msg) {
    FlData request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed returning FL to container with msg: {}", msg);
        return;
    }

    //check if cont_name is in containers
    if (containers.find(request.name()) == containers.end()) {
        spdlog::get("container_agent")->error("ReturnFL: Container {} not found!", request.name());
        return;
    }
    sendMessageToContainer(request.name(), MSG_TYPE[RETURN_FL], msg);
}

void DeviceAgent::Shutdown(const std::string &msg) {
    running = false;
}

void DeviceAgent::sendMessageToContainer(const std::string &topik, const std::string &type, const std::string &content) {
    std::string msg = absl::StrFormat("%s| %s %s", topik, type, content);
    message_t zmq_msg(msg.size());
    memcpy(zmq_msg.data(), msg.data(), msg.size());
    in_device_message_queue.send(zmq_msg, send_flags::none);
}

// Function to run the bash script with parameters from a JSON file
void DeviceAgent::limitBandwidth(const std::string& scriptPath, std::string interface) {
    if (dev_bandwidthLimit.empty()) {
        return;
    }

    unsigned int bwThresholdIndex = 0;
    ClockType nextThresholdSetTime = dev_startTime + std::chrono::seconds(dev_bandwidthLimit[bwThresholdIndex].time);
    while (isRunning()) {
        if (bwThresholdIndex >= dev_bandwidthLimit.size()) {
            break;
        }
        if (std::chrono::system_clock::now() >= nextThresholdSetTime) {
            Stopwatch stopwatch;

            // Build and execute the command
            std::ostringstream command;
            command << "sudo bash " << scriptPath << " " << interface << " " << std::fixed << std::setprecision(2) << dev_bandwidthLimit[bwThresholdIndex].mbps;
            spdlog::get("container_agent")->info("{0:s} Setting BW limit to {1:f} Mbps", dev_name, dev_bandwidthLimit[bwThresholdIndex].mbps);
            int result = system(command.str().c_str());
            spdlog::get("container_agent")->info("Command executed with result: {0:d}", result);

            if (bwThresholdIndex == dev_bandwidthLimit.size() - 1) {
                break;
            }
            bwThresholdIndex++;
            auto distanceToNext = dev_bandwidthLimit[bwThresholdIndex].time - dev_bandwidthLimit[bwThresholdIndex - 1].time;
            nextThresholdSetTime += std::chrono::seconds(distanceToNext);

            auto sleepTime = nextThresholdSetTime - std::chrono::system_clock::now();
            std::this_thread::sleep_for(sleepTime + std::chrono::nanoseconds(10000000));
        }
    }
    std::cout << "Finished bandwidth limiting." << std::endl;
}
