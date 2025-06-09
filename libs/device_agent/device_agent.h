#ifndef DEVICE_AGENT_H
#define DEVICE_AGENT_H

#include <cstdlib>
#include <misc.h>
#include <sys/sysinfo.h>
#include "profiler.h"
#include "controller.h"
#include "bcedge.h"
#include "edgevision.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>

#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <google/protobuf/empty.pb.h>
#include <pqxx/pqxx>

using trt::TRTConfigs;

ABSL_DECLARE_FLAG(std::string, name);
ABSL_DECLARE_FLAG(std::string, device_type);
ABSL_DECLARE_FLAG(std::string, controller_url);
ABSL_DECLARE_FLAG(std::string, dev_configPath);
ABSL_DECLARE_FLAG(uint16_t, dev_verbose);
ABSL_DECLARE_FLAG(uint16_t, dev_loggingMode);
ABSL_DECLARE_FLAG(std::string, dev_logPath);
ABSL_DECLARE_FLAG(uint16_t, dev_port_offset);
ABSL_DECLARE_FLAG(uint16_t, dev_system_port_offset);
ABSL_DECLARE_FLAG(uint16_t, dev_bandwidthLimitID);
ABSL_DECLARE_FLAG(std::string, dev_networkInterface);
ABSL_DECLARE_FLAG(int, dev_gpuID);

using indevicemessages::ContainerSignal;
using indevicemessages::Connection;
using indevicemessages::TimeKeeping;
using indevicemessages::ProcessData;
using indevicemessages::BCEdgeConfig;
using indevicemessages::BCEdgeData;
using indevicemessages::ContainerMetrics;
using EmptyMessage = google::protobuf::Empty;

class DeviceAgent {
public:
    DeviceAgent();
    DeviceAgent(const std::string &controller_url);

    virtual ~DeviceAgent() {
        running = false;
        ContainerSignal message;
        message.set_forced(true);
        for (const auto &c: containers) {
            message.set_name(c.first);
            StopContainer(message);
        }

        for (std::thread &t: threads) {
            t.join();
        }

        controller_ctx.shutdown();
        controller_ctx.close();
        in_device_ctx.shutdown();
        in_device_ctx.close();
    };

    [[nodiscard]] bool isRunning() const { return running; }

    void collectRuntimeMetrics();

    void limitBandwidth(const std::string& scriptPath, std::string interface);

protected:

    /////////////////////////////////////////// PROTECTED STRUCTURES ///////////////////////////////////////////

    struct DevContainerHandle {
        std::string name;
        unsigned int port;
        unsigned int pid;
        std::string startCommand;
        ModelType modelType;
        std::vector<int> dataShape;
        int instances;
        SummarizedHardwareMetrics hwMetrics;
        ContainerMetrics contextMetrics; // only filled for EdgeVision by default
    };

    /////////////////////////////////////////// PROTECTED FUNCTIONS ///////////////////////////////////////////

    // GENERAL OPERATION
    SystemInfo Ready(const std::string &ip);
    void Shutdown(const std::string &msg);

    // CONTAINER CONTROL
    void CreateContainer(const std::string &msg);
    void ReceiveStartReport(const std::string &msg);
    void ContainersLifeCheck();
    void ReceiveContainerMetrics(const std::string &msg);
    void UpdateContainerSender(const std::string &msg);
    void UpdateContainerSender(int mode, const std::string &cont_name, const std::string &dwnstr, const std::string &ip,
                               const int &port, const float &data_portion, const std::string &old_link,
                               const int64_t &timestamp, const int &offloading_duration);
    void SyncDatasources(const std::string &msg);
    void InferBCEdge(const std::string &msg);
    void UpdateBatchSize(const std::string &msg);
    void UpdateResolution(const std::string &msg);
    void UpdateTimeKeeping(const std::string &msg);
    void ForwardFL(const std::string &msg);
    void ReturnFL(const std::string &msg);
    void StopContainer(const std::string &msg);
    void StopContainer(ContainerSignal request);

    // MESSAGING & NETWORK
    void HandleDeviceMessages();
    virtual void HandleControlCommands();
    void testNetwork(const std::string &msg);
    void sendMessageToContainer(const std::string &topik, const std::string &type, const std::string &content);

    // SYSTEM COMMANDS
    std::string runDocker(const std::string &executable, const std::string &cont_name, const std::string &start_string,
                         int device, const int &port) {
        std::string command = "docker run -d --rm --network=host --runtime nvidia --gpus all ";
        std::string docker_tag;
        if (dev_gpuID > 0) device = dev_gpuID;
        if (dev_type == Virtual || dev_type == Server || dev_type == OnPremise) {
            command += "-v /ssd0/tung/PipePlusPlus/data/:/app/data/  -v /ssd0/tung/PipePlusPlus/logs/:/app/logs/ "
                       "-v /ssd0/tung/PipePlusPlus/models/:/app/models/ "
                       "-v /ssd0/tung/PipePlusPlus/model_profiles/:/app/model_profiles/ --name " +
                       absl::StrFormat(
                               R"(%s lucasliebe/pipeplusplus:amd64-torch %s --json '%s' --device %i --port %i --port_offset %i)",
                               cont_name, executable, start_string, device, port, dev_system_port_offset);
        } else {
            if (dev_type == NanoXavier || dev_type == NXXavier || dev_type == AGXXavier) {
                docker_tag = "jp512-torch";
            } else if (dev_type == OrinNano || dev_type == OrinNX || dev_type == OrinAGX) {
                docker_tag = "jp61-torch";
            } else {
                spdlog::get("container_agent")->error("Unknown edge device type while trying to start container!");
                return "";
            }
            command += "-u 0:0 --privileged -v /home/cdsn/FCPO:/app "
                       "-v /home/cdsn/pipe/data:/app/data -v /home/cdsn/pipe/models:/app/models "
                       "-v /run/jtop.sock:/run/jtop.sock  -v /usr/bin/tegrastats:/usr/bin/tegrastats --name " +
                        absl::StrFormat(
                                R"(%s lucasliebe/pipeplusplus:%s %s --json '%s' --device %i --port %i --port_offset %i)",
                                cont_name, docker_tag, executable, start_string, device, port, dev_system_port_offset);
        }
        command += " --log_dir ../logs";
        command += (deploy_mode? " --logging_mode 1" : " --verbose 0 --logging_mode 2");

        if (dev_type == Server || dev_type == Virtual) { // since many models might start on the server we need to slow down creation to prevent errors
            std::this_thread::sleep_for(std::chrono::milliseconds(700));
        }

        if (runDocker(command) != 0) {
            spdlog::get("container_agent")->error("Failed to start Container {}!", cont_name);
            return "";
        }
        return command;
    };
    int runDocker(const std::string &command) {
        spdlog::get("container_agent")->info("Running command: {}", command);
        return system(command.c_str());
    };

    /////////////////////////////////////////// PROTECTED VARIABLES ///////////////////////////////////////////

    // BASIC INFORMATION
    std::string dev_name;
    SystemDeviceType dev_type;
    DeviceInfoType dev_deviceInfo;
    std::string dev_experiment_name;
    std::string dev_system_name;
    int dev_agent_port_offset;
    int dev_system_port_offset;
    int dev_gpuID;

    // RUNTIME VARIABLES
    std::atomic<bool> running;
    std::atomic<bool> deploy_mode = false;
    std::chrono::high_resolution_clock::time_point dev_startTime;
    std::map<std::string, DevContainerHandle> containers;
    std::mutex containers_mutex;
    std::vector<std::thread> threads;

    // LOGGING
    std::string dev_logPath = "../logs";
    uint16_t dev_loggingMode = 0;
    uint16_t dev_verbose = 0;
    std::vector<spdlog::sink_ptr> dev_loggerSinks = {};
    std::shared_ptr<spdlog::logger> dev_logger;

    // MESSAGING & NETWORK
    context_t controller_ctx;
    socket_t controller_socket;
    socket_t controller_message_queue;
    context_t in_device_ctx;
    socket_t in_device_socket;
    socket_t in_device_message_queue;
    std::unordered_map<std::string, std::function<void(const std::string&)>> in_device_handlers;
    std::unordered_map<std::string, std::function<void(const std::string&)>> controller_handlers;
    std::vector<BandwidthManager> dev_totalBandwidthData;
    BandwidthManager dev_bandwidthLimit;

    // PROFILING DATA & METRICS
    Profiler *dev_profiler;
    MetricsServerConfigs dev_metricsServerConfigs;
    std::vector<DeviceHardwareMetrics> dev_runtimeMetrics;
    uint16_t dev_numCudaDevices{};
    std::unique_ptr<pqxx::connection> dev_metricsServerConn = nullptr;
    std::string dev_hwMetricsTableName;
    std::string dev_networkTableName;

    // LOCAL OPTIMIZATION
    BCEdgeAgent *dev_bcedge_agent;
    EdgeVisionAgent *dev_edgevision_agent;
    std::vector<EdgeVisionDwnstrmInfo> edgevision_dwnstrList;
    std::map<std::string, BandwidthManager> edgevision_dwnstrMetrics;
    TimePrecisionType dev_rlDecisionInterval;
    ClockType dev_nextRLDecisionTime = std::chrono::high_resolution_clock ::now();
};

#endif //DEVICE_AGENT_H
