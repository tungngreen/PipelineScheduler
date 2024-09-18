#ifndef DEVICE_AGENT_H
#define DEVICE_AGENT_H

#include <cstdlib>
#include <misc.h>
#include <sys/sysinfo.h>
#include "container_agent.h"
#include "profiler.h"
#include "controller.h"
#include "indevicecommunication.grpc.pb.h"
#include "controlcommunication.grpc.pb.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>

using trt::TRTConfigs;

const int CONTROLLER_BASE_PORT = 60001;
const int DEVICE_CONTROL_PORT = 60002;
const int INDEVICE_CONTROL_PORT = 60003;

ABSL_DECLARE_FLAG(std::string, name);
ABSL_DECLARE_FLAG(std::string, device_type);
ABSL_DECLARE_FLAG(std::string, controller_url);
ABSL_DECLARE_FLAG(std::string, dev_configPath);
ABSL_DECLARE_FLAG(uint16_t, dev_verbose);
ABSL_DECLARE_FLAG(uint16_t, dev_loggingMode);
ABSL_DECLARE_FLAG(std::string, dev_logPath);
ABSL_DECLARE_FLAG(uint16_t, dev_port_offset);

typedef std::tuple<
    std::string, // container name
    std::string, // name
    MicroserviceType, // type
    QueueLengthType, // queue length type
    int16_t, // class of interests
    std::vector<RequestDataShapeType>, //data shape
    QueueLengthType
> MsvcConfigTupleType;

struct DevContainerHandle {
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    CompletionQueue *cq;
    unsigned int port;
    unsigned int pid;
    SummarizedHardwareMetrics hwMetrics;
};

class DeviceAgent {
public:
    DeviceAgent();
    DeviceAgent(const std::string &controller_url);

    virtual ~DeviceAgent() {
        running = false;
        for (const auto &c: containers) {
            StopContainer(c.second);
        }

        if (controller_server) {
            controller_server->Shutdown();
        }
        if (device_server) {
            device_server->Shutdown();
        }

        for (std::thread &t: threads) {
            t.join();
        }

        if (controller_cq) {
            controller_cq->Shutdown();
        }
        if (device_cq) {
            device_cq->Shutdown();
        }
    };

    bool isRunning() const {
        return running;
    }

    void collectRuntimeMetrics();

    void limitBandwidth(const std::string& scriptPath, const std::string& jsonFilePath);

protected:
    void testNetwork(float min_size, float max_size, int num_loops);

    bool CreateContainer(ContainerConfig &c);

    int runDocker(const std::string &executable, const std::string &cont_name, const std::string &start_string,
                         const int &device, const int &port) {
        std::string command;
        command =
                "docker run --network=host -v /ssd0/tung/PipePlusPlus/data/:/app/data/  "
                "-v /ssd0/tung/PipePlusPlus/logs/:/app/logs/ -v /ssd0/tung/PipePlusPlus/models/:/app/models/ "
                "-v /ssd0/tung/PipePlusPlus/model_profiles/:/app/model_profiles/ "
                "-d --rm --runtime nvidia --gpus all --name " +
                absl::StrFormat(
                        R"(%s pipeline-base-container %s --json '%s' --device %i --port %i --port_offset %i)",
                        cont_name, executable, start_string, device, port, dev_port_offset) +
                " --log_dir ../logs";

        if (!deploy_mode) {
            command += " --verbose 0 --logging_mode 2";
        } else {
            command += " --logging_mode 1";
        }
        if (dev_type == SystemDeviceType::Server) { // since many models might start on the server we need to slow down creation to prevent errors
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        spdlog::get("container_agent")->info("Running command: {}", command);
        return system(command.c_str());
    };

    static void StopContainer(const DevContainerHandle &container, bool forced = false);

    void UpdateContainerSender(const std::string &cont_name, const std::string &dwnstr, const std::string &ip,
                               const int &port);

    void SyncDatasources(const std::string &cont_name, const std::string &dsrc);

    void Ready(const std::string &ip, SystemDeviceType type);

    void HandleDeviceRecvRpcs();

    virtual void HandleControlRecvRpcs();

    class RequestHandler {
    public:
        RequestHandler(ServerCompletionQueue *cq, DeviceAgent *device) : cq(cq), status(CREATE), device_agent(device) {}

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        DeviceAgent *device_agent;
    };

    class DeviceRequestHandler : public RequestHandler {
    public:
        DeviceRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq, DeviceAgent *d)
                : RequestHandler(cq, d), service(service) {};

    protected:
        InDeviceCommunication::AsyncService *service;
    };

    class ControlRequestHandler : public RequestHandler {
    public:
        ControlRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq, DeviceAgent *d)
                : RequestHandler(cq, d), service(service) {};

    protected:
        ControlCommunication::AsyncService *service;
    };


    class ReportStartRequestHandler : public DeviceRequestHandler {
    public:
        ReportStartRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  DeviceAgent *device)
                : DeviceRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ProcessData request;
        ProcessData reply;
        grpc::ServerAsyncResponseWriter<ProcessData> responder;
    };

    class ExecuteNetworkTestRequestHandler : public ControlRequestHandler {
    public:
        ExecuteNetworkTestRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                     DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        LoopRange request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class StartContainerRequestHandler : public ControlRequestHandler {
    public:
        StartContainerRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                     DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerConfig request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class StopContainerRequestHandler : public ControlRequestHandler {
    public:
        StopContainerRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                    DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerSignal request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class UpdateDownstreamRequestHandler : public ControlRequestHandler {
    public:
        UpdateDownstreamRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerLink request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class SyncDatasourceRequestHandler : public ControlRequestHandler {
    public:
        SyncDatasourceRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerLink request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class UpdateBatchsizeRequestHandler : public ControlRequestHandler {
    public:
        UpdateBatchsizeRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerInts request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class UpdateResolutionRequestHandler : public ControlRequestHandler {
    public:
        UpdateResolutionRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                      DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerInts request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class UpdateTimeKeepingRequestHandler : public ControlRequestHandler {
    public:
        UpdateTimeKeepingRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                      DeviceAgent *device)
                : ControlRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        TimeKeeping request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    SystemDeviceType dev_type;
    DeviceInfoType dev_deviceInfo;
    std::atomic<bool> deploy_mode = false;

    // Basic information
    std::string dev_name;
    bool running;
    std::string dev_experiment_name;
    std::string dev_system_name;
    int dev_port_offset;

    // Runtime variables
    Profiler *dev_profiler;
    std::map<std::string, DevContainerHandle> containers;
    std::vector<std::thread> threads;
    std::vector<DeviceHardwareMetrics> dev_runtimeMetrics;

    // Communication
    std::unique_ptr<ServerCompletionQueue> device_cq;
    std::unique_ptr<grpc::Server> device_server;
    InDeviceCommunication::AsyncService device_service;
    std::unique_ptr<ControlCommunication::Stub> controller_stub;
    CompletionQueue *controller_sending_cq;
    std::unique_ptr<grpc::Server> controller_server;
    std::unique_ptr<ServerCompletionQueue> controller_cq;
    ControlCommunication::AsyncService controller_service;

    // This will be mounted into the container to easily collect all logs.
    std::string dev_logPath = "../logs";
    uint16_t dev_loggingMode = 0;
    uint16_t dev_verbose = 0;

    std::vector<spdlog::sink_ptr> dev_loggerSinks = {};
    std::shared_ptr<spdlog::logger> dev_logger;

    MetricsServerConfigs dev_metricsServerConfigs;
    std::unique_ptr<pqxx::connection> dev_metricsServerConn = nullptr;
    std::string dev_hwMetricsTableName;
    std::string dev_networkTableName;

    uint16_t dev_numCudaDevices{};
};

#endif //DEVICE_AGENT_H
