#ifndef DEVICE_AGENT_H
#define DEVICE_AGENT_H

#include "profiler.h"
#include <cstdlib>
#include <misc.h>
#include <sys/sysinfo.h>
#include "container_agent.h"
#include "controller.h"
#include "indevicecommunication.grpc.pb.h"
#include "controlcommunication.grpc.pb.h"

using trt::TRTConfigs;

ABSL_DECLARE_FLAG(std::string, deviceType);
ABSL_DECLARE_FLAG(std::string, controller_url);

typedef std::tuple<
    std::string, // container name
    std::string, // name
    MicroserviceType, // type
    QueueLengthType, // queue length type
    int16_t, // class of interests
    std::vector<RequestDataShapeType>, //data shape
    QueueLengthType
> MsvcConfigTupleType;

struct ContainerHandle {
    google::protobuf::RepeatedField<int32_t> queuelengths;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    Metrics metrics;
    CompletionQueue *cq;
    unsigned int pid;
    bool reportMetrics;
};

namespace msvcconfigs {
    void to_json(json &j, const NeighborMicroserviceConfigs &val);

    void to_json(json &j, const BaseMicroserviceConfigs &val);
}

class DeviceAgent {
public:
    DeviceAgent(const std::string &controller_url, const std::string name, DeviceType type);

    ~DeviceAgent() {
        running = false;
        for (const auto &c: containers) {
            StopContainer(c.second);
        }
        controller_server->Shutdown();
        controller_cq->Shutdown();
        device_server->Shutdown();
        device_cq->Shutdown();
        for (std::thread &t: threads) {
            t.join();
        }
    };

    void UpdateState(const std::basic_string<char> &container_name, const float &requestrate,
                     const google::protobuf::RepeatedField<int32_t> &queuelengths) {
        containers[container_name].queuelengths = queuelengths;
        containers[container_name].metrics.requestRate = requestrate;
    };

    bool isRunning() const {
        return running;
    }

private:
    bool CreateContainer(
            ModelType model,
            std::string name,
            BatchSizeType batch_size,
            int device,
            const MsvcSLOType &slo,
            const google::protobuf::RepeatedPtrField<Neighbor> &upstreams,
            const google::protobuf::RepeatedPtrField<Neighbor> &downstreams
    );

    static int runDocker(const std::string &executable, const std::string &name, const std::string &start_string,
                         const int &device, const int &port) {
        std::string command = absl::StrFormat(
                R"(docker run --network=host -d --runtime nvidia --gpus all pipeline-base-container %s --name="%s" --json='%s' --device=%i --port=%i)",
                executable, name, start_string, device, port);
        std::cout << command << std::endl;
        return system(command.c_str());
    };

    static void StopContainer(const ContainerHandle &container, bool forced = false);

    void UpdateContainerSender(const std::string &name, const std::string &dwnstr, const std::string &ip,
                               const int &port);

    void Ready(const std::string &name, const std::string &ip, DeviceType type);

    void ReportDeviceStatus();

    void ReportFullMetrics();

    void HandleDeviceRecvRpcs();

    void HandleControlRecvRpcs();

    void MonitorDeviceStatus();

    class RequestHandler {
    public:
        RequestHandler(ServerCompletionQueue *cq) : cq(cq), status(CREATE) {}

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
    };

    class DeviceRequestHandler : public RequestHandler {
    public:
        DeviceRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : RequestHandler(cq), service(service) {};

    protected:
        InDeviceCommunication::AsyncService *service;
    };

    class ControlRequestHandler : public RequestHandler {
    public:
        ControlRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : RequestHandler(cq), service(service) {};

    protected:
        ControlCommunication::AsyncService *service;
    };

    class StateUpdateRequestHandler : public DeviceRequestHandler {
    public:
        StateUpdateRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  DeviceAgent *device)
                : DeviceRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        State request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    class ReportStartRequestHandler : public DeviceRequestHandler {
    public:
        ReportStartRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  DeviceAgent *device)
                : DeviceRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        ProcessData request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    class StartMicroserviceRequestHandler : public ControlRequestHandler {
    public:
        StartMicroserviceRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                        DeviceAgent *device)
                : ControlRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        MicroserviceConfig request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    class StopMicroserviceRequestHandler : public ControlRequestHandler {
    public:
        StopMicroserviceRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        MicroserviceSignal request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    class UpdateDownstreamRequestHandler : public ControlRequestHandler {
    public:
        UpdateDownstreamRequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                       DeviceAgent *device)
                : ControlRequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        MicroserviceLink request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
        DeviceAgent *device_agent;
    };

    bool running;
    Profiler *profiler;
    std::map<std::string, ContainerHandle> containers;
    std::vector<std::thread> threads;

    std::unique_ptr<ServerCompletionQueue> device_cq;
    std::unique_ptr<grpc::Server> device_server;
    InDeviceCommunication::AsyncService device_service;
    std::unique_ptr<ControlCommunication::Stub> controller_stub;
    std::unique_ptr<ServerCompletionQueue> controller_cq;
    std::unique_ptr<grpc::Server> controller_server;
    CompletionQueue *controller_sending_cq;
    ControlCommunication::AsyncService controller_service;

    // This will be mounted into the container to easily collect all logs.
    std::string dev_logPath = "../logs";
};

#endif //DEVICE_AGENT_H