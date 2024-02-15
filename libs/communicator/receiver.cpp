#include "absl/strings/str_format.h"
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

#include "microservice.h"
#include "sender.h"

class GPULoader : public Microservice<DataRequest<LocalCPUDataType>> {
public:
    GPULoader(const BaseMicroserviceConfigs &configs, ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>> *out)
            : Microservice(configs) {
        InQueue = new ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>>();
        OutQueue = out;
    }

    ~GPULoader();

    void Schedule() {
        DataRequest<LocalCPUDataType> req = InQueue->pop();
        // copy data to gpu using cuda
        std::vector<Data<LocalGPUDataType>> elements = {};
        for (auto el: req.req_data) {
            auto gpu_image = cv::cuda::GpuMat(req.req_dataShape[0], req.req_dataShape[1], CV_8UC3);
            gpu_image.upload(el.content);
            elements.push_back({req.req_dataShape, gpu_image});
        }
        OutQueue->emplace(
                {req.req_origGenTime, req.req_e2eSLOLatency, req.req_travelPath, elements});
    }

    ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *getInQueue() {
        return InQueue;
    }

protected:
    ThreadSafeFixSizedQueue<DataRequest<LocalGPUDataType>> *OutQueue;
};

class Receiver : public GPUDataMicroservice<void> {
public:
    Receiver(const BaseMicroserviceConfigs &configs, std::string url, uint16_t port)
            : GPUDataMicroservice<void>(configs) {
        std::string server_address = absl::StrFormat("%s:%d", url, port);
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service_);
        cq_ = builder.AddCompletionQueue();
        server_ = builder.BuildAndStart();
        LoadingQueue = GPULoader(BaseMicroserviceConfigs(), OutQueue).getInQueue();
        HandleRpcs();
    }

    ~Receiver() {
        server_->Shutdown();
        cq_->Shutdown();
    }

private:
    class RequestHandler {
    public:
        RequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                       ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : service_(service), cq_(cq), LoadingQueue(lq) {}

        virtual void Proceed() = 0;

        DataTransferService::AsyncService *service_;
        ServerCompletionQueue *cq_;
        ServerContext ctx_;
    protected:
        ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *LoadingQueue;
    };

    class GpuPointerRequestHandler : public RequestHandler {
    public:
        GpuPointerRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                 ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : RequestHandler(service, cq, lq), responder_(&ctx_), status_(CREATE) {
            Proceed();
        }

        void Proceed() final {
            if (status_ == CREATE) {
                status_ = PROCESS;
                service_->RequestGpuPointerTransfer(&ctx_, &request_, &responder_, cq_, cq_,
                                                    this);
            } else if (status_ == PROCESS) {
                new GpuPointerRequestHandler(service_, cq_, LoadingQueue);

                std::vector<Data<LocalGPUDataType>> elements = {};
                for (const auto& el : *request_.mutable_elements()) {
                    auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3,
                                                      (void *) (&el.data()));
                    elements.push_back({{el.width(), el.height()}, gpu_image});
                }
                DataRequest<LocalGPUDataType> req = {request_.timestamp(), request_.slo(),
                                                     request_.path(), elements};
                OutQueue->emplace(req);

                status_ = FINISH;
                responder_.Finish(reply_, Status::OK, this);
            } else {
                GPR_ASSERT(status_ == FINISH);
                delete this;
            }
        }

    private:
        GpuPointerPayload request_;
        SimpleConfirm reply_;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder_;

        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        CallStatus status_;
    };

    class SharedMemoryRequestHandler : public RequestHandler {
    public:
        SharedMemoryRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                   ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : RequestHandler(service, cq, lq), responder_(&ctx_), status_(CREATE) {
            Proceed();
        }

        void Proceed() final {
            if (status_ == CREATE) {
                status_ = PROCESS;
                service_->RequestSharedMemTransfer(&ctx_, &request_, &responder_, cq_, cq_,
                                                   this);
            } else if (status_ == PROCESS) {
                new SharedMemoryRequestHandler(service_, cq_, LoadingQueue);

                std::vector<Data<LocalCPUDataType>> elements = {};
                for (const auto& el : *request_.mutable_elements()) {
                    auto name = el.name().c_str();
                    boost::interprocess::shared_memory_object shm{open_only, name, read_only};
                    boost::interprocess::mapped_region region{shm, read_only};
                    auto image = static_cast<cv::Mat *>(region.get_address());
                    elements.push_back({{el.width(), el.height()}, *image});

                    boost::interprocess::shared_memory_object::remove(name);
                }
                DataRequest<LocalCPUDataType> req = {request_.timestamp(), request_.slo(),
                                                     request_.path(), elements};
                LoadingQueue->emplace(req);

                status_ = FINISH;
                responder_.Finish(reply_, Status::OK, this);
            } else {
                GPR_ASSERT(status_ == FINISH);
                delete this;
            }
        }

    private:
        SharedMemPayload request_;
        SimpleConfirm reply_;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder_;

        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        CallStatus status_;
    };

    class SerializedDataRequestHandler : public RequestHandler {
    public:
        SerializedDataRequestHandler(DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
                                     ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *lq)
                : RequestHandler(service, cq, lq), responder_(&ctx_), status_(CREATE) {
            service_ = service;
            cq_ = cq;
            Proceed();
        }

        void Proceed() final {
            if (status_ == CREATE) {
                status_ = PROCESS;
                service_->RequestSerializedDataTransfer(&ctx_, &request_, &responder_, cq_, cq_,
                                                        this);
            } else if (status_ == PROCESS) {
                new SerializedDataRequestHandler(service_, cq_, LoadingQueue);

                std::vector<Data<LocalCPUDataType>> elements = {};
                for (const auto& el : *request_.mutable_elements()) {
                    uint length = el.data().length();
                    if (length != el.datalen()) {
                        responder_.Finish(reply_, Status(grpc::INVALID_ARGUMENT, "Data length does not match"), this);
                    }
                    cv::Mat image = cv::Mat(el.height(), el.width(), CV_8UC3,
                                            const_cast<char *>(el.data().c_str())).clone();
                    elements.push_back({{el.width(), el.height()}, image});
                }
                DataRequest<LocalCPUDataType> req = {request_.timestamp(), request_.slo(),
                                                    request_.path(), elements};
                LoadingQueue->emplace(req);

                status_ = FINISH;
                responder_.Finish(reply_, Status::OK, this);
            } else {
                GPR_ASSERT(status_ == FINISH);
                delete this;
            }
        }

    private:
        SerializedDataPayload request_;
        SimpleConfirm reply_;
        grpc::ServerAsyncResponseWriter<SimpleConfirm> responder_;

        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        CallStatus status_;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs() {
        new GpuPointerRequestHandler(&service_, cq_.get(), LoadingQueue);
        new SharedMemoryRequestHandler(&service_, cq_.get(), LoadingQueue);
        new SerializedDataRequestHandler(&service_, cq_.get(), LoadingQueue);
        void *tag;  // uniquely identifies a request.
        bool ok;
        while (true) {
            GPR_ASSERT(cq_->Next(&tag, &ok));
            GPR_ASSERT(ok);
            static_cast<RequestHandler *>(tag)->Proceed();
        }
    }

    std::unique_ptr<ServerCompletionQueue> cq_;
    DataTransferService::AsyncService service_;
    std::unique_ptr<Server> server_;

    ThreadSafeFixSizedQueue<DataRequest<LocalCPUDataType>> *LoadingQueue;
};
