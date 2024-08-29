#ifndef PIPEPLUSPLUS_RECEIVER_H
#define PIPEPLUSPLUS_RECEIVER_H

#include "communicator.h"
#include <fstream>
#include <random>

using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using boost::interprocess::read_only;
using boost::interprocess::open_only;
using json = nlohmann::ordered_json;

struct ReceiverConfigs : BaseMicroserviceConfigs {
    uint8_t msvc_inputRandomizeScheme;
    std::string msvc_dataShape;
};

class Receiver : public Microservice {
public:
    Receiver(const json &jsonConfigs);

    ~Receiver() override {
        server->Shutdown();
        cq->Shutdown();
    }

    template<typename ReqDataType>
    void processInferTimeReport(Request<ReqDataType> &timeReport);

    void dispatchThread() override {
        std::thread handler(&Receiver::HandleRpcs, this);
        handler.detach();
    }

    ReceiverConfigs loadConfigsFromJson(const json &jsonConfigs);

    void loadConfigs(const json &jsonConfigs, bool isConstructing = true);
    
private:
    class RequestHandler {
    public:
        RequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount, Receiver *receiver)
                : service(service), msvc_inReqCount(msvc_inReqCount), cq(cq), OutQueue(out), status(CREATE), receiverInstance(receiver) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        std::string containerName;
        DataTransferService::AsyncService *service;
        uint64_t &msvc_inReqCount;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        ThreadSafeFixSizedDoubleQueue *OutQueue;
        CallStatus status;
        Receiver *receiverInstance;

        /**
         * @brief Check if this request is still valid or its too old and should be discarded
         * 
         * @param timestamps 
         * @return true 
         * @return false 
         */
        inline bool validateReq(ClockType &originalGenTime) {
            auto now = std::chrono::high_resolution_clock::now();
            auto diff = std::chrono::duration_cast<TimePrecisionType>(now - originalGenTime).count();
            if (receiverInstance->msvc_RUNMODE == RUNMODE::PROFILING) {
                return true;
            }
            if (diff > receiverInstance->msvc_pipelineSLO - receiverInstance->msvc_timeBudgetLeft && 
                receiverInstance->msvc_DROP_MODE == DROP_MODE::LAZY) {
                receiverInstance->msvc_droppedReqCount++;
                spdlog::get("container_agent")->trace("{0:s} drops a request with time {1:d}", containerName, diff);
                return false;
            } else if (receiverInstance->msvc_DROP_MODE == DROP_MODE::NO_DROP) {
                return true;
            }
            return true;
        }
    };

    class GpuPointerRequestHandler : public RequestHandler {
    public:
        GpuPointerRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount, Receiver *receiver);

        void Proceed() final;

    private:
        ImageDataPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class SharedMemoryRequestHandler : public RequestHandler {
    public:
        SharedMemoryRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount, Receiver *receiver);

        void Proceed() final;

        void test() {
            ImageDataPayload request;
        }

    private:
        ImageDataPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class SerializedDataRequestHandler : public RequestHandler {
    public:
        SerializedDataRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedDoubleQueue *out, uint64_t &msvc_inReqCount, Receiver *receiver);

        void Proceed() final;

    private:
        ImageDataPayload request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs();

    std::unique_ptr<ServerCompletionQueue> cq;
    DataTransferService::AsyncService service;
    std::unique_ptr<grpc::Server> server;
};

#endif //PIPEPLUSPLUS_RECEIVER_H
