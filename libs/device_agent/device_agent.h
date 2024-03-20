#include "container_agent.h"
#include "indevicecommunication.grpc.pb.h"
#include <cstdlib>
#include <misc.h>

using trt::TRTConfigs;

enum ContainerType {
    DataSource,
    Yolo5,
};

struct ContainerHandle {
    google::protobuf::RepeatedField <int32_t> queuelengths;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    CompletionQueue *cq;
};

namespace msvcconfigs {
    void to_json(json &j, const NeighborMicroserviceConfigs &val) {
        j["name"] = val.name;
        j["comm"] = val.commMethod;
        j["link"] = val.link;
        j["maxqs"] = val.maxQueueSize;
        j["coi"] = val.classOfInterest;
        j["shape"] = val.expectedShape;
    }

    void to_json(json &j, const BaseMicroserviceConfigs &val) {
        j["name"] = val.msvc_name;
        j["type"] = val.msvc_type;
        j["slo"] = val.msvc_svcLevelObjLatency;
        j["bs"] = val.msvc_idealBatchSize;
        j["ds"] = val.msvc_dataShape;
        j["upstrm"] = val.upstreamMicroservices;
        j["downstrm"] = val.dnstreamMicroservices;
    }
}

class DeviceAgent {
public:
    DeviceAgent(const std::string &controller_url, uint16_t controller_port);

    ~DeviceAgent() {
        for (const auto &c: containers) {
            StopContainer(c.second);
        }
        server->Shutdown();
        server_cq->Shutdown();
    };

    void UpdateQueueLengths(const std::basic_string<char> &container_name,
                            const google::protobuf::RepeatedField <int32_t> &queuelengths) {
        containers[container_name].queuelengths = queuelengths;
    };

private:
    void CreateYolo5Container(int id, const NeighborMicroserviceConfigs &upstream,
                              const std::vector<NeighborMicroserviceConfigs> &downstreams, const MsvcSLOType &slo);

    void CreateDataSource(int id, const std::vector<NeighborMicroserviceConfigs> &downstreams, const MsvcSLOType &slo,
                          const std::string &video_path);

    static json createConfigs(
            const std::vector<std::tuple<std::string, MicroserviceType, QueueLengthType, int16_t, std::vector<RequestShapeType>>> &data,
            const MsvcSLOType &slo, const NeighborMicroserviceConfigs &prev_msvc,
            const std::vector<NeighborMicroserviceConfigs> &next_msvc);

    void finishContainer(const std::string &executable, const std::string &name, const std::string &start_string,
                         const int &control_port, const int &data_port, const std::string &trt_config = "");

    static int runDocker(const std::string &executable, const std::string &name, const std::string &start_string,
                         const int &port, const std::string &trt_config) {
        if (trt_config.empty()) {
            std::cout << absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i)",
                    executable, name, start_string, port).c_str() << std::endl;
            return system(absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i)",
                    executable, name, start_string, port).c_str());
        } else {
            std::cout << absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i --trt_json='%s')",
                    executable, name, start_string, port, trt_config).c_str() << std::endl;
            return system(absl::StrFormat(
                    R"(docker run --network=host -d --gpus 1 pipeline-base-container %s --name="%s" --json='%s' --port=%i --trt_json='%s')",
                    executable, name, start_string, port, trt_config).c_str());
        }
    };

    static void StopContainer(const ContainerHandle &container);

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

    class CounterUpdateRequestHandler : public RequestHandler {
    public:
        CounterUpdateRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                    DeviceAgent *device)
                : RequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        QueueSize request;
        StaticConfirm reply;
        grpc::ServerAsyncResponseWriter<StaticConfirm> responder;
        DeviceAgent *device_agent;
    };

    class ReportStartRequestHandler : public RequestHandler {
    public:
        ReportStartRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                  DeviceAgent *device)
                : RequestHandler(service, cq), responder(&ctx), device_agent(device) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::ConnectionConfigs request;
        StaticConfirm reply;
        grpc::ServerAsyncResponseWriter<StaticConfirm> responder;
        DeviceAgent *device_agent;
    };

    void HandleRecvRpcs();

    std::map<std::string, ContainerHandle> containers;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    std::unique_ptr<Server> server;
    CompletionQueue *sender_cq;
    std::unique_ptr<InDeviceCommunication::Stub> controller_stub;
    InDeviceCommunication::AsyncService service;
};