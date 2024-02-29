#ifndef CONTAINER_AGENT_H
#define CONTAINER_AGENT_H

#include <vector>
#include <thread>
#include "absl/strings/str_format.h"
#include "absl/flags/flag.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

#include "../json/json.h"
#include "../communicator/sender.cpp"
#include "../communicator/receiver.cpp"
#include "../protobufprotocols/indevicecommunication.grpc.pb.h"

#include "../yolov5/yolov5.h"
#include "../data_source/data_source.cpp"
#include "../trtengine/trtengine.h"

ABSL_DECLARE_FLAG(std::string, name);
ABSL_DECLARE_FLAG(std::string, json);
ABSL_DECLARE_FLAG(uint16_t, port);

using json = nlohmann::json;

using indevicecommunication::InDeviceCommunication;
using indevicecommunication::QueueSize;
using indevicecommunication::StaticConfirm;

enum TransferMethod {
    LocalCPU,
    RemoteCPU,
    GPU
};

struct ConnectionConfigs {
    std::string ip;
    int port;
};

namespace msvcconfigs {
    void from_json(const json &j, NeighborMicroserviceConfigs &val);

    void from_json(const json &j, BaseMicroserviceConfigs &val);
}

class ContainerAgent {
public:
    ContainerAgent(const std::string &name, uint16_t device_port, uint16_t own_port);

    ~ContainerAgent() {
        for (auto msvc: msvcs) {
            delete msvc;
        }
        server->Shutdown();
        server_cq->Shutdown();
        sender_cq->Shutdown();
    };

    [[nodiscard]] bool running() const {
        return run;
    }

    void SendQueueLengths();

protected:
    void ReportStart(int port);

    class RequestHandler {
    public:
        RequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : service(service), cq(cq), status(CREATE) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        InDeviceCommunication::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
    };

    class StopRequestHandler : public RequestHandler {
    public:
        StopRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                           std::atomic<bool> *run)
                : RequestHandler(service, cq), responder(&ctx), run(run) {
            Proceed();
        }

        void Proceed() final;

    private:
        StaticConfirm request;
        StaticConfirm reply;
        grpc::ServerAsyncResponseWriter<StaticConfirm> responder;
        std::atomic<bool> *run;
    };

    void HandleRecvRpcs();

    std::string name;
    std::vector<Microservice<void> *> msvcs;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<Server> server;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run{};
};

class Yolo5ContainerAgent : public ContainerAgent {
public:
    Yolo5ContainerAgent(const std::string &name, uint16_t device_port, uint16_t own_port,
                        std::vector<BaseMicroserviceConfigs> &msvc_configs);
};

class DataSourceAgent : public ContainerAgent {
public:
    DataSourceAgent(const std::string &name, uint16_t device_port, uint16_t own_port,
                    std::vector<BaseMicroserviceConfigs> &msvc_configs);
};

#endif