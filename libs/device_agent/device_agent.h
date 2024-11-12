#ifndef DEVICE_AGENT_H
#define DEVICE_AGENT_H

#include <cstdlib>
#include <misc.h>
#include <sys/sysinfo.h>
#include "profiler.h"
#include "controller.h"
#include "bcedge.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>

#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
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

using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using indevicecommands::InDeviceCommands;
using indevicecommands::ContainerSignal;
using indevicecommands::Connection;
using indevicecommands::TimeKeeping;
using indevicemessages::ProcessData;
using EmptyMessage = google::protobuf::Empty;

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
    std::unique_ptr<InDeviceCommands::Stub> stub;
    CompletionQueue *cq;
    unsigned int port;
    unsigned int pid;
    std::string startCommand;
    SummarizedHardwareMetrics hwMetrics;
};

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
            StopContainer(c.second, message);
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

    void ContainersLifeCheck();

    std::string runDocker(const std::string &executable, const std::string &cont_name, const std::string &start_string,
                         const int &device, const int &port) {
        std::string command;
        command =
                "docker run --network=host -u 0:0 --privileged -v /home/cdsn:/home/soulsaver "
                "-v /home/cdsn/pipe/data:/home/soulsaver/FCPO/data "
                "-v /home/cdsn/pipe/models:/home/soulsaver/FCPO/models "
                "-v /run/jtop.sock:/run/jtop.sock  -v /usr/bin/tegrastats:/usr/bin/tegrastats "
                "-d --rm --runtime nvidia --gpus all --name " +
                absl::StrFormat(
                        R"(%s pipeline-scheduler-orin-nano:trt-libtorch %s --json '%s' --device %i --port %i --port_offset %i)",
                        cont_name, executable, start_string, device, port, dev_port_offset) +
                " --log_dir ../logs";
        command += deploy_mode? " --logging_mode 1" : " --verbose 0 --logging_mode 2";

        if (dev_type == SystemDeviceType::Server) { // since many models might start on the server we need to slow down creation to prevent errors
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

    static void StopContainer(const DevContainerHandle &container, ContainerSignal message);

    void UpdateContainerSender(int mode, const std::string &cont_name, const std::string &dwnstr, const std::string &ip,
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
        DeviceRequestHandler(InDeviceMessages::AsyncService *service, ServerCompletionQueue *cq, DeviceAgent *d)
                : RequestHandler(cq, d), service(service) {};

    protected:
        InDeviceMessages::AsyncService *service;
    };

    class ControlRequestHandler : public RequestHandler {
    public:
        ControlRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq, DeviceAgent *d)
                : RequestHandler(cq, d), service(service), responder(&ctx) {};

    protected:
        ControlCommands::AsyncService *service;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };


    class ReportStartRequestHandler : public DeviceRequestHandler {
    public:
        ReportStartRequestHandler(InDeviceMessages::AsyncService *service, ServerCompletionQueue *cq,
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

    class StartFederatedLearningRequestHandler : public DeviceRequestHandler {
    public:
        StartFederatedLearningRequestHandler(InDeviceMessages::AsyncService *service, ServerCompletionQueue *cq,
                                             DeviceAgent *device)
                : DeviceRequestHandler(service, cq, device), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        FlData request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class ExecuteNetworkTestRequestHandler : public ControlRequestHandler {
    public:
        ExecuteNetworkTestRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                     DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        LoopRange request;
    };

    class StartContainerRequestHandler : public ControlRequestHandler {
    public:
        StartContainerRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                     DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerConfig request;
    };

    class StopContainerRequestHandler : public ControlRequestHandler {
    public:
        StopContainerRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                    DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerSignal request;
    };

    class UpdateDownstreamRequestHandler : public ControlRequestHandler {
    public:
        UpdateDownstreamRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerLink request;
    };

    class SyncDatasourceRequestHandler : public ControlRequestHandler {
    public:
        SyncDatasourceRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerLink request;
    };

    class UpdateBatchsizeRequestHandler : public ControlRequestHandler {
    public:
        UpdateBatchsizeRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerInts request;
    };

    class UpdateResolutionRequestHandler : public ControlRequestHandler {
    public:
        UpdateResolutionRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                      DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        ContainerInts request;
    };

    class UpdateTimeKeepingRequestHandler : public ControlRequestHandler {
    public:
        UpdateTimeKeepingRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                      DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        TimeKeeping request;
    };

    class ReturnFlRequestHandler : public ControlRequestHandler {
    public:
        ReturnFlRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                        DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        FlData request;
    };

    class ShutdownRequestHandler : public ControlRequestHandler {
    public:
        ShutdownRequestHandler(ControlCommands::AsyncService *service, ServerCompletionQueue *cq,
                                        DeviceAgent *device)
                : ControlRequestHandler(service, cq, device) {
            Proceed();
        }

        void Proceed() final;

    private:
        EmptyMessage request;
    };

    SystemDeviceType dev_type;
    DeviceInfoType dev_deviceInfo;
    std::atomic<bool> deploy_mode = false;

    // Basic information
    std::string dev_name;
    std::atomic<bool> running;
    std::string dev_experiment_name;
    std::string dev_system_name;
    int dev_port_offset;

    // Runtime variables
    Profiler *dev_profiler;
    std::map<std::string, DevContainerHandle> containers;
    std::mutex containers_mutex;
    std::vector<std::thread> threads;
    std::vector<DeviceHardwareMetrics> dev_runtimeMetrics;

    // Communication
    std::unique_ptr<ServerCompletionQueue> device_cq;
    std::unique_ptr<grpc::Server> device_server;
    InDeviceMessages::AsyncService device_service;
    std::unique_ptr<ControlMessages::Stub> controller_stub;
    CompletionQueue *controller_sending_cq;
    std::unique_ptr<grpc::Server> controller_server;
    std::unique_ptr<ServerCompletionQueue> controller_cq;
    ControlCommands::AsyncService controller_service;

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

    BCEdgeAgent *dev_bcedge_agent;
};

#endif //DEVICE_AGENT_H
