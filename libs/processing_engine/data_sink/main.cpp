#include "receiver.h"
#include "basesink.cpp"
#include "misc.h"
#include "controlmessages.grpc.pb.h"

ABSL_FLAG(std::string, name, "", "base name of container");
ABSL_FLAG(std::string, json, "{\"experimentName\": \"none\", \"pipelineName\": \"none\", \"systemName\": \"none\","
                             "\"controllerIP\": \"none\"}",
          "json experiment configs");
ABSL_FLAG(std::string, log_dir, "../logs", "Log path for the container");
ABSL_FLAG(uint16_t, port, 0, "receiving port for the sink");
ABSL_FLAG(uint16_t, verbose, 2, "verbose level 0:trace, 1:debug, 2:info, 3:warn, 4:error, 5:critical, 6:off");
ABSL_FLAG(uint16_t, logging_mode, 0, "0:stdout, 1:file, 2:both");
ABSL_FLAG(std::optional<uint16_t>, port_offset, 0, "port offset for control communication");
//dummy flags to make command sync with other containers
ABSL_FLAG(std::optional<int16_t>, device, 0, "UNUSED FOR SINK - NO EFFECT");
ABSL_FLAG(bool, restart, 0, "UNUSED FOR SINK - NO EFFECT");

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    std::vector<spdlog::sink_ptr> loggerSinks;
    std::shared_ptr<spdlog::logger> logger;
    json j = json::parse(absl::GetFlag(FLAGS_json));

    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    logPath += "/" + j["experimentName"].get<std::string>();
    std::filesystem::create_directory(
            std::filesystem::path(logPath)
    );
    logPath += "/" + j["systemName"].get<std::string>();
    std::filesystem::create_directory(
            std::filesystem::path(logPath)
    );

    setupLogger(
            absl::GetFlag(FLAGS_log_dir),
            absl::GetFlag(FLAGS_name),
            absl::GetFlag(FLAGS_logging_mode),
            absl::GetFlag(FLAGS_verbose),
            loggerSinks,
            logger
    );
    std::string taskName = j.contains("taskName") ? j["taskName"].get<std::string>() :  j["pipelineName"].get<std::string>();
    std::string controllerIP = j.contains("controllerIP") ? j["controllerIP"].get<std::string>() : "none";
    context_t controller_ctx;
    socket_t controller_socket;
    if (controllerIP != "none") {
        controller_ctx = context_t(1);
        std::string server_address = absl::StrFormat("tcp://%s:%d", controllerIP, CONTROLLER_RECEIVE_PORT + absl::GetFlag(FLAGS_port_offset).value());
        controller_socket = socket_t(controller_ctx, ZMQ_REQ);
        controller_socket.connect(server_address);
    }

    std::string common_end = "\", \"msvc_maxBatchSize\": 64, \"msvc_allocationMode\": 1, \"msvc_numWarmUpBatches\": 0, "
                             "\"msvc_batchMode\": 0, \"msvc_dropMode\": 0, \"msvc_timeBudgetLeft\": 9999999, "
                             "\"msvc_contSLO\": 30000, \"msvc_pipelineSLO\": 9999999, \"msvc_contStartTime\": 0, "
                             "\"msvc_contEndTime\": 30000, \"msvc_localDutyCycle\": 50000, \"msvc_cycleStartTime\": 0, "
                             "\"msvc_batchTimeout\": 0}";

    json receiver_json = json::parse(std::string("{\"msvc_contName\": \"dataSink\", \"msvc_deviceIndex\": 0, "
                                                 "\"msvc_RUNMODE\": 0, \"msvc_name\": \"receiver\", \"msvc_type\": 0, "
                                                 "\"msvc_svcLevelObjLatency\": 1, "
                                                 "\"msvc_idealBatchSize\": 1, \"msvc_dataShape\": [[0, 0]], "
                                                 "\"msvc_maxQueueSize\": 100, \"msvc_dnstreamMicroservices\": [{"
                                                 "\"nb_name\": \"data_sink\", \"nb_commMethod\": 4, \"nb_link\": [\"\"], "
                                                 "\"nb_classOfInterest\": -1, \"nb_maxQueueSize\": 10, \"nb_portions\": [], "
                                                 "\"nb_expectedShape\": [[-1, -1]]}], \"msvc_upstreamMicroservices\": [{"
                                                 "\"nb_name\": \"various\", \"nb_commMethod\": 2, "
                                                 "\"nb_link\": [\"0.0.0.0:") +
                                     std::to_string(absl::GetFlag(FLAGS_port)) +
                                     std::string("\"], \"nb_classOfInterest\": -2, \"nb_portions\": [], "
                                                 "\"nb_maxQueueSize\": 10, \"nb_expectedShape\": [[-1, -1]]}],"
                                                 "\"msvc_containerLogPath\": \".") + common_end);
    receiver_json["msvc_experimentName"] = j["experimentName"];
    receiver_json["msvc_pipelineName"] = j["pipelineName"];
    receiver_json["msvc_taskName"] = "sink";
    receiver_json["msvc_hostDevice"] = "server";
    receiver_json["msvc_systemName"] = j["systemName"];
    Microservice *receiver = new Receiver(receiver_json);
    json sink_json = json::parse(std::string("{\"msvc_contName\": \"dataSink\", \"msvc_deviceIndex\": 0, "
                                             "\"msvc_RUNMODE\": 0, \"msvc_name\": \"data_sink\", \"msvc_type\": 502, "
                                             "\"msvc_svcLevelObjLatency\": 1, "
                                             "\"msvc_idealBatchSize\": 1, \"msvc_dataShape\": [[0, 0]], "
                                             "\"msvc_maxQueueSize\": 100, \"msvc_dnstreamMicroservices\": [], "
                                             "\"msvc_upstreamMicroservices\": [{\"nb_name\": \"::receiver\", "
                                             "\"nb_commMethod\": 2, \"nb_link\": [\"\"], \"nb_classOfInterest\": -2, "
                                             "\"nb_maxQueueSize\": 10, \"nb_expectedShape\": [[-1, -1]], \"nb_portions\": []}],"
                                             "\"msvc_containerLogPath\": \"") +
                                 logPath + common_end);
    sink_json["msvc_experimentName"] = j["experimentName"];
    sink_json["msvc_pipelineName"] = j["pipelineName"];
    sink_json["msvc_taskName"] = "sink";
    sink_json["msvc_hostDevice"] = "server";
    sink_json["msvc_systemName"] = j["systemName"];
    Microservice *sink = new BaseSink(sink_json);
    sink->SetInQueue(receiver->GetOutQueue());
    receiver->dispatchThread();
    sink->dispatchThread();
    sleep(1);
    receiver->unpauseThread();
    sink->unpauseThread();
    std::cout << "Start Running" << std::endl;

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        uint32_t arrival_rate = receiver->getPerSecondArrivalRecord().numRequests;
        MsvcSLOType agg_Latency = sink->getAggLatency();
        float avg_latency = (float) agg_Latency / (float) arrival_rate;
        logger->info("Arrival rate: {0:d} requests/s with avg latency: {1:.2f} ms", arrival_rate, avg_latency / 1000.f);
        if (controllerIP != "none") {
            controlmessages::SinkMetrics request;
            request.set_name(taskName);
            request.set_avg_latency(avg_latency);
            request.set_throughput(arrival_rate);
            std::string message = MSG_TYPE[SINK_METRICS] + " " + request.SerializeAsString();
            message_t zmq_msg(message.size());
            memcpy(zmq_msg.data(), message.data(), message.size());
            controller_socket.send(zmq_msg, send_flags::dontwait);
        }
    }
    return 0;
}
