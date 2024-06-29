#include "controller.h"

ABSL_FLAG(std::string, ctrl_configPath, "../jsons/experiments/base-experiment.json",
          "Path to the configuration file for this experiment.");
ABSL_FLAG(uint16_t, ctrl_verbose, 0, "Verbosity level of the controller.");
ABSL_FLAG(uint16_t, ctrl_loggingMode, 0, "Logging mode of the controller. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, ctrl_logPath, "../logs", "Path to the log dir for the controller.");

const int DATA_BASE_PORT = 55001;
const int CONTROLLER_BASE_PORT = 60001;
const int DEVICE_CONTROL_PORT = 60002;

void Controller::readConfigFile(const std::string &path) {
    std::ifstream file(path);
    json j = json::parse(file);

    ctrl_experimentName = j["expName"];
    ctrl_systemName = j["systemName"];
    ctrl_runtime = j["runtime"];
    ctrl_port_offset = j["port_offset"];
    initialTasks = j["initial_pipelines"];
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val) {
    j.at("pipeline_name").get_to(val.name);
    val.fullName = val.name + "_" + val.device;
    j.at("pipeline_target_slo").get_to(val.slo);
    j.at("pipeline_type").get_to(val.type);
    j.at("video_source").get_to(val.source);
    j.at("pipeline_source_device").get_to(val.device);
}

Controller::Controller(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    readConfigFile(absl::GetFlag(FLAGS_ctrl_configPath));

    ctrl_logPath = absl::GetFlag(FLAGS_ctrl_logPath);
    ctrl_logPath += "/" + ctrl_experimentName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_logPath += "/" + ctrl_systemName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_verbose = absl::GetFlag(FLAGS_ctrl_verbose);
    ctrl_loggingMode = absl::GetFlag(FLAGS_ctrl_loggingMode);

    setupLogger(
            ctrl_logPath,
            "controller",
            ctrl_loggingMode,
            ctrl_verbose,
            ctrl_loggerSinks,
            ctrl_logger
    );

    ctrl_containerLib = getContainerLib("all");

    json metricsCfgs = json::parse(std::ifstream("../jsons/metricsserver.json"));
    ctrl_metricsServerConfigs.from_json(metricsCfgs);
    ctrl_metricsServerConfigs.schema = abbreviate(ctrl_experimentName + "_" + ctrl_systemName);
    ctrl_metricsServerConfigs.user = "controller";
    ctrl_metricsServerConfigs.password = "agent";
    ctrl_metricsServerConn = connectToMetricsServer(ctrl_metricsServerConfigs, "controller");

    std::string sql = "CREATE SCHEMA IF NOT EXISTS " + ctrl_metricsServerConfigs.schema + ";";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "GRANT USAGE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "GRANT CREATE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);
    sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " GRANT SELECT, INSERT ON TABLES TO device_agent, container_agent;";
    pushSQL(*ctrl_metricsServerConn, sql);

    std::thread networkCheckThread(&Controller::checkNetworkConditions, this);
    networkCheckThread.detach();

    running = true;

    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", CONTROLLER_BASE_PORT + ctrl_port_offset);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();
}

Controller::~Controller() {
    std::unique_lock<std::mutex> lock(containers.containersMutex);
    for (auto &msvc: containers.list) {
        StopContainer(msvc.first, msvc.second.device_agent, true);
    }

    std::unique_lock<std::mutex> lock2(devices.devicesMutex);
    for (auto &device: devices.list) {
        device.second.cq->Shutdown();
        void *got_tag;
        bool ok = false;
        while (device.second.cq->Next(&got_tag, &ok));
    }
    server->Shutdown();
    cq->Shutdown();
}

void Controller::HandleRecvRpcs() {
    new DeviseAdvertisementHandler(&service, cq.get(), this);
    new DummyDataRequestHandler(&service, cq.get(), this);
    void *tag;
    bool ok;
    while (running) {
        if (!cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void Controller::Scheduling() {
    // TODO: please out your scheduling loop inside of here
    while (running) {
        // use list of devices, tasks and containers to schedule depending on your algorithm
        // put helper functions as a private member function of the controller and write them at the bottom of this file.
        std::this_thread::sleep_for(std::chrono::milliseconds(
                5000)); // sleep time can be adjusted to your algorithm or just left at 5 seconds for now
    }
}

void Controller::queryInDeviceNetworkEntries(NodeHandle *node) {
    std::string deviceTypeName = SystemDeviceTypeList[node->type];
    std::string deviceTypeNameAbbr = abbreviate(deviceTypeName);
    if (ctrl_inDeviceNetworkEntries.find(deviceTypeName) == ctrl_inDeviceNetworkEntries.end()) {
        std::string tableName = "prof_" + deviceTypeNameAbbr + "_netw";
        std::string sql = absl::StrFormat("SELECT p95_transfer_duration_us, p95_total_package_size_b "
                                    "FROM %s ", tableName);
        pqxx::result res = pullSQL(*ctrl_metricsServerConn, sql);
        if (res.empty()) {
            spdlog::get("container_agent")->error("No in-device network entries found for device type {}.", deviceTypeName);
            return;
        }
        for (pqxx::result::const_iterator row = res.begin(); row != res.end(); ++row) {
            std::pair<uint32_t, uint64_t> entry = {row["p95_total_package_size_b"].as<uint32_t>(), row["p95_transfer_duration_us"].as<uint64_t>()};
            ctrl_inDeviceNetworkEntries[deviceTypeName].emplace_back(entry);
        }
        spdlog::get("container_agent")->info("Finished querying in-device network entries for device type {}.", deviceTypeName);
    }
    std::unique_lock lock(node->nodeHandleMutex);
    node->latestNetworkEntries[deviceTypeName] = aggregateNetworkEntries(ctrl_inDeviceNetworkEntries[deviceTypeName]);
    std::cout << node->latestNetworkEntries[deviceTypeName].size() << std::endl;
}

void Controller::DeviseAdvertisementHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestAdvertiseToController(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new DeviseAdvertisementHandler(service, cq, controller);
        std::string target_str = absl::StrFormat("%s:%d", request.ip_address(), DEVICE_CONTROL_PORT + controller->ctrl_port_offset);
        std::string deviceName = request.device_name();
        NodeHandle node{deviceName,
                                     request.ip_address(),
                                     ControlCommunication::NewStub(
                                             grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials())),
                                     new CompletionQueue(),
                                     static_cast<SystemDeviceType>(request.device_type()),
                                     request.processors(), std::vector<double>(request.processors(), 0.0),
                                     std::vector<unsigned long>(request.memory().begin(), request.memory().end()),
                                     std::vector<double>(request.processors(), 0.0), DATA_BASE_PORT + controller->ctrl_port_offset, {}};
        reply.set_name(controller->ctrl_systemName);
        reply.set_experiment(controller->ctrl_experimentName);
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
        controller->devices.addDevice(deviceName, node);
        spdlog::get("container_agent")->info("Device {} is connected to the system", request.device_name());
        controller->queryInDeviceNetworkEntries(&(controller->devices.list[deviceName]));
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::DummyDataRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSendDummyData(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new DummyDataRequestHandler(service, cq, controller);
        ClockType now = std::chrono::system_clock::now();
        unsigned long diff = std::chrono::duration_cast<TimePrecisionType>(
                now - std::chrono::time_point<std::chrono::system_clock>(TimePrecisionType(request.gen_time()))).count();
        unsigned int size = request.data().size();
        controller->network_check_buffer[request.origin_name()].push_back({size, diff});
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void Controller::StartContainer(std::pair<std::string, ContainerHandle *> &container, int slo, std::string source,
                                int replica, bool easy_allocation) {
    std::cout << "Starting container: " << container.first << std::endl;
    ContainerConfig request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_pipeline_name(container.second->task->tk_name);
    request.set_model(container.second->model);
    request.set_model_file(container.second->model_file[replica -1]);
    request.set_batch_size(container.second->batch_size[replica -1]);
    request.set_replica_id(replica);
    request.set_allocation_mode(easy_allocation);
    request.set_device(container.second->cuda_device[replica - 1]);
    request.set_slo(slo);
    for (auto dim: container.second->dimensions) {
        request.add_input_dimensions(dim);
    }
    for (auto dwnstr: container.second->downstreams) {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name(dwnstr->name);
        dwn->set_ip(absl::StrFormat("%s:%d", dwnstr->device_agent->ip, dwnstr->recv_port[replica - 1]));
        dwn->set_class_of_interest(dwnstr->class_of_interest);
        if (dwnstr->model == Sink) {
            dwn->set_gpu_connection(false);
        } else {
            dwn->set_gpu_connection((container.second->device_agent == dwnstr->device_agent) &&
                                    (container.second->cuda_device == dwnstr->cuda_device));
        }
    }
    if (request.downstream_size() == 0) {
        Neighbor *dwn = request.add_downstream();
        dwn->set_name("video_sink");
        dwn->set_ip("./out.log"); //output log file
        dwn->set_class_of_interest(-1);
        dwn->set_gpu_connection(false);
    }
    if (container.second->model == DataSource || container.second->model == Yolov5nDsrc || container.second->model == RetinafaceDsrc) {
        Neighbor *up = request.add_upstream();
        up->set_name("video_source");
        up->set_ip(source);
        up->set_class_of_interest(-1);
        up->set_gpu_connection(false);
    } else {
        for (auto upstr: container.second->upstreams) {
            Neighbor *up = request.add_upstream();
            up->set_name(upstr->name);
            up->set_ip(absl::StrFormat("0.0.0.0:%d", container.second->recv_port[replica -1]));
            up->set_class_of_interest(-2);
            up->set_gpu_connection((container.second->device_agent == upstr->device_agent) &&
                                   (container.second->cuda_device == upstr->cuda_device));
        }
    }
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            container.second->device_agent->stub->AsyncStartContainer(&context, request,
                                                                      container.second->device_agent->cq));
    finishGrpc(rpc, reply, status, container.second->device_agent->cq);
    if (!status.ok()) {
        std::cout << status.error_code() << ": An error occured while sending the request" << std::endl;
    }
}

void Controller::MoveContainer(ContainerHandle *msvc, bool to_edge, int cuda_device, int replica) {
    NodeHandle *old_device = msvc->device_agent;
    NodeHandle *device;
    bool start_dsrc = false, merge_dsrc = false;
    if (to_edge) {
        device = msvc->upstreams[0]->device_agent;
        if (msvc->mergable) {
            merge_dsrc = true;
            if (msvc->model == Yolov5n) {
                msvc->model = Yolov5nDsrc;
            } else if (msvc->model == Retinaface) {
                msvc->model = RetinafaceDsrc;
            }
        }
    } else {
        device = &devices.list["server"];
        if (msvc->mergable) {
            start_dsrc = true;
            if (msvc->model == Yolov5nDsrc) {
                msvc->model = Yolov5n;
            } else if (msvc->model == RetinafaceDsrc) {
                msvc->model = Retinaface;
            }
        }
    }
    msvc->device_agent = device;
    msvc->recv_port[replica - 1] = device->next_free_port++;
    device->containers.insert({msvc->name, msvc});
    msvc->cuda_device[replica - 1] = cuda_device;
    std::pair<std::string, ContainerHandle *> pair = {msvc->name, msvc};
    StartContainer(pair, msvc->task->tk_slo, msvc->task->tk_source, replica, !(start_dsrc || merge_dsrc));
    for (auto upstr: msvc->upstreams) {
        if (start_dsrc) {
            std::pair<std::string, ContainerHandle *> dsrc_pair = {upstr->name, upstr};
            StartContainer(dsrc_pair, upstr->task->tk_slo, msvc->task->tk_source, replica, false);
            SyncDatasource(msvc, upstr);
        } else if (merge_dsrc) {
            SyncDatasource(upstr, msvc);
            StopContainer(upstr->name, old_device);
        } else {
            AdjustUpstream(msvc->recv_port[replica - 1], upstr, device, msvc->name);
        }
    }
    StopContainer(msvc->name, old_device);
    old_device->containers.erase(msvc->name);
}

void Controller::AdjustUpstream(int port, ContainerHandle *upstr, NodeHandle *new_device,
                                const std::string &dwnstr) {
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
    finishGrpc(rpc, reply, status, upstr->device_agent->cq);
}

void Controller::SyncDatasource(ContainerHandle *prev, ContainerHandle *curr) {
    ContainerLink request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(prev->name);
    request.set_downstream_name(curr->name);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            curr->device_agent->stub->AsyncSyncDatasource(&context, request, curr->device_agent->cq));
    finishGrpc(rpc, reply, status, curr->device_agent->cq);
}

void Controller::AdjustBatchSize(ContainerHandle *msvc, int new_bs, int replica) {
    msvc->batch_size[replica - 1] = new_bs;
    ContainerInts request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.add_value(new_bs);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateBatchSize(&context, request, msvc->device_agent->cq));
    finishGrpc(rpc, reply, status, msvc->device_agent->cq);
}

void AdjustResolution(ContainerHandle *msvc, std::vector<int> new_resolution, int replica = 1) {
    msvc->dimensions = new_resolution;
    ContainerInts request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(msvc->name);
    request.add_value(new_resolution[0]);
    request.add_value(new_resolution[1]);
    request.add_value(new_resolution[2]);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            msvc->device_agent->stub->AsyncUpdateResolution(&context, request, msvc->device_agent->cq));
    finishGrpc(rpc, reply, status, msvc->device_agent->cq);
}

void Controller::StopContainer(std::string name, NodeHandle *device, bool forced) {
    ContainerSignal request;
    ClientContext context;
    EmptyMessage reply;
    Status status;
    request.set_name(name);
    request.set_forced(forced);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            device->stub->AsyncStopContainer(&context, request, containers.list[name].device_agent->cq));
    finishGrpc(rpc, reply, status, device->cq);
}

/**
 * @brief 
 * 
 * @param container calculating queue sizes for the container before its official deployment.
 * @param modelType 
 */
void Controller::calculateQueueSizes(ContainerHandle &container, const ModelType modelType) {
    float preprocessRate = 1000000.f / container.expectedPreprocessLatency; // queries per second
    float postprocessRate = 1000000.f / container.expectedPostprocessLatency; // qps
    float inferRate = 1000000.f / (container.expectedInferLatency * container.batch_size[0]); // batch per second

    QueueLengthType minimumQueueSize = 30;

    // Receiver to Preprocessor
    // Utilization of preprocessor
    float preprocess_rho = container.arrival_rate / preprocessRate;
    QueueLengthType preprocess_inQueueSize = std::max((QueueLengthType) std::ceil(preprocess_rho * preprocess_rho / (2 * (1 - preprocess_rho))), minimumQueueSize);
    float preprocess_thrpt = std::min(preprocessRate, container.arrival_rate);

    // Preprocessor to Inferencer
    // Utilization of inferencer
    float infer_rho = preprocess_thrpt / container.batch_size[0] / inferRate;
    QueueLengthType infer_inQueueSize = std::max((QueueLengthType) std::ceil(infer_rho * infer_rho / (2 * (1 - infer_rho))), minimumQueueSize);
    float infer_thrpt = std::min(inferRate, preprocess_thrpt / container.batch_size[0]); // batch per second

    float postprocess_rho = (infer_thrpt * container.batch_size[0]) / postprocessRate;
    QueueLengthType postprocess_inQueueSize = std::max((QueueLengthType) std::ceil(postprocess_rho * postprocess_rho / (2 * (1 - postprocess_rho))), minimumQueueSize);
    float postprocess_thrpt = std::min(postprocessRate, infer_thrpt * container.batch_size[0]);

    QueueLengthType sender_inQueueSize = postprocess_inQueueSize * container.batch_size[0];

    container.queueSizes = {preprocess_inQueueSize, infer_inQueueSize, postprocess_inQueueSize, sender_inQueueSize};

    container.expectedThroughput = postprocess_thrpt;
}

// void Controller::optimizeBatchSizeStep(
//         const Pipeline &models,
//         std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects) {
//     ModelType candidate;
//     int max_saving = 0;
//     std::vector<ModelType> blacklist;
//     for (const auto &m: models) {
//         int saving;
//         if (max_saving == 0) {
//             saving =
//                     estimated_infer_times[m.first] - InferTimeEstimator(m.first, batch_sizes[m.first] * 2);
//         } else {
//             if (batch_sizes[m.first] == 64 ||
//                 std::find(blacklist.begin(), blacklist.end(), m.first) != blacklist.end()) {
//                 continue;
//             }
//             for (const auto &d: m.second) {
//                 if (batch_sizes[d.first] > batch_sizes[m.first]) {
//                     blacklist.push_back(d.first);
//                 }
//             }
//             saving = estimated_infer_times[m.first] -
//                      (InferTimeEstimator(m.first, batch_sizes[m.first] * 2) * (nObjects / batch_sizes[m.first] * 2));
//         }
//         if (saving > max_saving) {
//             max_saving = saving;
//             candidate = m.first;
//         }
//     }
//     batch_sizes[candidate] *= 2;
//     estimated_infer_times[candidate] -= max_saving;
// }

// double Controller::LoadTimeEstimator(const char *model_path, double input_mem_size) {
//     // Load the pre-trained model
//     BoosterHandle booster;
//     int num_iterations = 1;
//     int ret = LGBM_BoosterCreateFromModelfile(model_path, &num_iterations, &booster);

//     // Prepare the input data
//     std::vector<double> input_data = {input_mem_size};

//     // Perform inference
//     int64_t out_len;
//     std::vector<double> out_result(1);
//     ret = LGBM_BoosterPredictForMat(booster,
//                                     input_data.data(),
//                                     C_API_DTYPE_FLOAT64,
//                                     1,  // Number of rows
//                                     1,  // Number of columns
//                                     1,  // Is row major
//                                     C_API_PREDICT_NORMAL,  // Predict type
//                                     0,  // Start iteration
//                                     -1,  // Number of iterations, -1 means use all
//                                     "",  // Parameter
//                                     &out_len,
//                                     out_result.data());
//     if (ret != 0) {
//         std::cout << "Failed to perform inference!" << std::endl;
//         exit(ret);
//     }

//     // Print the predicted value
//     std::cout << "Predicted value: " << out_result[0] << std::endl;

//     // Free the booster handle
//     LGBM_BoosterFree(booster);

//     return out_result[0];
// }


/**
 * @brief
 *
 * @param model to specify model
 * @param batch_size for targeted batch size (binary)
 * @return int for inference time per full batch in nanoseconds
 */
int Controller::InferTimeEstimator(ModelType model, int batch_size) {
    return 0;
}

// std::map<ModelType, std::vector<int>> Controller::InitialRequestCount(const std::string &input, const Pipeline &models,
//                                                                       int fps) {
//     std::map<ModelType, std::vector<int>> request_counts = {};
//     std::vector<int> fps_values = {fps, fps * 3, fps * 7, fps * 15, fps * 30, fps * 60};

//     request_counts[models[0].first] = fps_values;
//     json objectCount = json::parse(std::ifstream("../jsons/object_count.json"))[input];

//     for (const auto &m: models) {
//         if (m.first == ModelType::Sink) {
//             request_counts[m.first] = std::vector<int>(6, 0);
//             continue;
//         }

//         for (const auto &d: m.second) {
//             if (d.second == -1) {
//                 request_counts[d.first] = request_counts[m.first];
//             } else {
//                 std::vector<int> objects = (d.second == 0 ? objectCount["person"]
//                                                           : objectCount["car"]).get<std::vector<int>>();

//                 for (int j: fps_values) {
//                     int count = std::accumulate(objects.begin(), objects.begin() + j, 0);
//                     request_counts[d.first].push_back(request_counts[m.first][0] * count);
//                 }
//             }
//         }
//     }
//     return request_counts;
// }

/**
 * @brief '
 * 
 * @param node 
 * @param minPacketSize bytes
 * @param maxPacketSize bytes
 * @param numLoops 
 * @return NetworkEntryType 
 */
NetworkEntryType Controller::initNetworkCheck(const NodeHandle &node, uint32_t minPacketSize, uint32_t maxPacketSize, uint32_t numLoops) {
    LoopRange request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    request.set_min(minPacketSize);
    request.set_max(maxPacketSize);
    request.set_repetitions(numLoops);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            node.stub->AsyncExecuteNetworkTest(&context, request, node.cq));
    finishGrpc(rpc, reply, status, node.cq);

    while (network_check_buffer[node.name].size() < numLoops) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    NetworkEntryType entries = network_check_buffer[node.name];
    network_check_buffer[node.name].clear();
    return entries;
};

/**
 * @brief Query the latest network entries for each device to determine the network conditions.
 * If no such entries exists, send to each device a request for network testing.
 * 
 */
void Controller::checkNetworkConditions() {
    std::this_thread::sleep_for(TimePrecisionType(5 * 1000000));
    while (running) {
        Stopwatch stopwatch;
        stopwatch.start();
        std::map<std::string, NetworkEntryType> networkEntries = {};

        for (auto &[deviceName, nodeHandle] : *(devices.getMap())) {
            if (deviceName == "server") {
                continue;
            }
            networkEntries[deviceName] = {};
        }
        std::string tableName = ctrl_metricsServerConfigs.schema + "." + abbreviate(ctrl_experimentName) + "_serv_netw";
        std::string query = absl::StrFormat("SELECT sender_host, p95_transfer_duration_us, p95_total_package_size_b "
                            "FROM %s ", tableName);

        pqxx::result res = pullSQL(*ctrl_metricsServerConn, query);
        //Getting the latest network entries into the networkEntries map
        for (pqxx::result::const_iterator row = res.begin(); row != res.end(); ++row) {
            std::string sender_host = row["sender_host"].as<std::string>();
            if (sender_host == "server" || sender_host == "serv") {
                continue;
            }
            std::pair<uint32_t, uint64_t> entry = {row["p95_total_package_size_b"].as<uint32_t>(), row["p95_transfer_duration_us"].as<uint64_t>()};
            networkEntries[sender_host].emplace_back(entry);
        }

        // Updating NodeHandle object with the latest network entries
        for (auto &[deviceName, entries] : networkEntries) {
            // If entry belongs to a device that is not in the list of devices, ignore it
            if (devices.list.find(deviceName) == devices.list.end() || deviceName != "server") {
                continue;
            }
            std::lock_guard<std::mutex> lock(devices.list[deviceName].nodeHandleMutex);
            devices.list[deviceName].latestNetworkEntries["server"] = aggregateNetworkEntries(entries);
        }

        // If no network entries exist for a device, send a request to the device to perform network testing
        for (auto &[deviceName, nodeHandle] : devices.list) {
            if (nodeHandle.latestNetworkEntries.size() == 0) {
                // TODO: Send a request to the device to perform network testing

            }
        }

        stopwatch.stop();
        std::this_thread::sleep_for(TimePrecisionType(60 * 1000000 - stopwatch.elapsed_microseconds()));
    }
}