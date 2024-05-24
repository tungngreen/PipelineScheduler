#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <grpcpp/grpcpp.h>
#include "../json/json.h"
#include <thread>
#include "controlcommunication.grpc.pb.h"
#include <LightGBM/c_api.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using controlcommunication::ControlCommunication;
using controlcommunication::ConnectionConfigs;
using controlcommunication::Neighbor;
using controlcommunication::ContainerConfig;
using controlcommunication::ContainerLink;
using controlcommunication::ContainerInt;
using controlcommunication::ContainerSignal;
using EmptyMessage = google::protobuf::Empty;

enum SystemDeviceType {
    Server,
    Edge
};

enum ModelType {
    DataSource,
    Sink,
    Yolov5,
    Yolov5Datasource,
    Arcface,
    Retinaface,
    Yolov5_Plate,
    Movenet,
    Emotionnet,
    Gender,
    Age,
    CarBrand
};

extern std::map<ModelType, std::vector<std::string>> MODEL_INFO;

enum PipelineType {
    Traffic,
    Video_Call,
    Building_Security
};

struct Metrics {
    float requestRate = 0;
    double cpuUsage = 0;
    long memUsage = 0;
    unsigned int gpuUsage = 0;
    unsigned int gpuMemUsage = 0;
};

namespace TaskDescription {
    struct TaskStruct {
        std::string name;
        int slo;
        PipelineType type;
        std::string source;
        std::string device;
    };

    void to_json(nlohmann::json &j, const TaskStruct &val);

    void from_json(const nlohmann::json &j, TaskStruct &val);
}

class Controller {
public:
    Controller();

    ~Controller();

    void HandleRecvRpcs();

    void AddTask(const TaskDescription::TaskStruct &task);

    [[nodiscard]] bool isRunning() const { return running; };

    void Stop() { running = false; };

private:
    void UpdateLightMetrics();

    void UpdateFullMetrics();

    double LoadTimeEstimator(const char *model_path, double input_mem_size);
    int InferTimeEstimator(ModelType model, int batch_size);

    struct ContainerHandle;
    struct NodeHandle {
        std::string ip;
        std::shared_ptr<ControlCommunication::Stub> stub;
        CompletionQueue *cq;
        SystemDeviceType type;
        int num_processors; // number of processing units, 1 for Edge or # GPUs for server
        std::vector<double> processors_utilization; // utilization per pu
        std::vector<unsigned long> mem_size; // memory size in MB
        std::vector<double> mem_utilization; // memory utilization per pu
        int next_free_port;
        std::map<std::string, ContainerHandle *> containers;
    };

    struct TaskHandle {
        int slo;
        PipelineType type;
        std::map<std::string, ContainerHandle *> subtasks;
    };

    struct ContainerHandle {
        std::string name;
        ModelType model;
        NodeHandle *device_agent;
        TaskHandle *task;
        int batch_size;
        int replicas;
        std::vector<int> cuda_device;
        int class_of_interest;
        int recv_port;
        Metrics metrics;
        google::protobuf::RepeatedField<int32_t> queue_lengths;
        std::vector<ContainerHandle *> upstreams;
        std::vector<ContainerHandle *> downstreams;
    };

    class RequestHandler {
    public:
        RequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq, Controller *c)
                : service(service), cq(cq), status(CREATE), controller(c), responder(&ctx) {}

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        ControlCommunication::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        Controller *controller;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class DeviseAdvertisementHandler : public RequestHandler {
    public:
        DeviseAdvertisementHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
                : RequestHandler(service, cq, c) {
            Proceed();
        }

        void Proceed() final;

    private:
        ConnectionConfigs request;
    };

    void StartContainer(std::pair<std::string, ContainerHandle *> &upstr, int slo,
                        std::string source = "", int replica = 1);

    void MoveContainer(ContainerHandle *msvc, int cuda_device, bool to_edge, int replica = 1);

    static void AdjustUpstream(int port, ContainerHandle *msvc, NodeHandle *new_device, const std::string &dwnstr);

    void AdjustBatchSize(ContainerHandle *msvc, int new_bs);

    void StopContainer(std::string name, NodeHandle *device, bool forced = false);

    void optimizeBatchSizeStep(
            const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models,
            std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects);

    std::map<ModelType, int> getInitialBatchSizes(
            const std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> &models, int slo,
            int nObjects);

    std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>>
    getModelsByPipelineType(PipelineType type);

    bool running;
    std::map<std::string, NodeHandle> devices;
    std::map<std::string, TaskHandle> tasks;
    std::map<std::string, ContainerHandle> containers;

    ControlCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;
};

// ========================================================== added ================================================================

class RIM_Module{
public:
    RIM_Module(const std::string& name, int max_fps, int resource_usage) 
        : name(name), max_fps(max_fps), resource_usage(resource_usage) {}
    virtual ~RIM_Module() {}
    virtual void execute(const void* input_data, void* output_data) = 0;
    std::string getName() const { return name; }
    int getMaxFPS() const { return max_fps; }
    int getResourceUsage() const { return resource_usage; }

private:
    std::string name;
    int max_fps;
    int resource_usage;
};

class RIM_Worker {
public:
    Worker(const std::string& id, int maxCapacity)
        : id(id), maxCapacity(maxCapacity), usedCapacity(0) {}

    std::string getId() const { return id; }

    bool canAccommodate(int required_capacity) const {
        return (used_capacity + required_capacity < max_capacity);
    }

    bool assignModule(const std::shared_ptr<RIM_Module>& module, int fps) {
        int required_capacity = (fps * module)
    }

private:
    std::string id;
    int max_capacity;
    int used_capacity;
};

class RIM_Session {
public:
    RIM_Session(int id, const std::vector<std::shared_ptr<RIM_Module>>& modules, int targetFps, int targetLatency)
        : id(id), modules(modules), targetFps(targetFps), targetLatency(targetLatency) {}

    int getId() const { return id; }

    const std::vector<std::shared_ptr<RIM_Module>>& getModules() const { return modules; }

    int getTargetFps() const { return targetFps; }

    int getTargetLatency() const { return targetLatency; }

private:
    int id;
    std::vector<std::shared_ptr<RIM_Module>> modules;
    int targetFps;
    int targetLatency;
};

class RIM_Client {
public:
    void connectToMaster(std::shared_ptr<RIM_Master> master) {
        this->master = master;
    }
    int setupSession(const std::string& mdagName, int targetFps, int targetLatency) {
        return master->setupSession(mdagName, targetFps, targetLatency);
    }

private:
    std::shared_ptr<RIM_Master> master;
};

class RIM_Master {
public:
    RIM_Master() : nextSessionId(1) {}
    void registerWorker(std::shared_ptr<RIM_Worker> worker) {
        workers.push_back(worker);
    }
    void addMDAGProfile(const std::string& mdagName, const std::vector<std::shared_ptr<RIM_Module>>& modules) {
        mdagProfiles[mdagName] = modules;
    }
    int setupSession(const std::string& mdagName, int targetFps, int targetLatency);

private:
    bool placeSession(const std::shared_ptr<RIM_Session>& session);

    bool canPlaceOnSingleWorker(const std::shared_ptr<Worker>& worker, const std::shared_ptr<RIM_Session>& session);

    void placeOnSingleWorker(const std::shared_ptr<Worker>& worker, const std::shared_ptr<RIM_Session>& session);

    bool placeAcrossWorkers(const std::shared_ptr<RIM_Session>& session);

    std::vector<std::shared_ptr<Worker>> workers;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Module>>> mdagProfiles;
    int nextSessionId;
};

#endif //PIPEPLUSPLUS_CONTROLLER_H
