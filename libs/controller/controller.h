#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <grpcpp/grpcpp.h>
#include "../json/json.h"
#include <thread>
#include "controlcommunication.grpc.pb.h"
#include <LightGBM/c_api.h>
#include <vector>
#include <algorithm>

using controlcommunication::ConnectionConfigs;
using controlcommunication::ContainerConfig;
using controlcommunication::ContainerLink;
using controlcommunication::ContainerInt;
using controlcommunication::ContainerSignal;
using controlcommunication::ControlCommunication;
using controlcommunication::Neighbor;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;
using EmptyMessage = google::protobuf::Empty;

enum SystemDeviceType
{
    Server,
    Edge
};

enum ModelType
{
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

enum PipelineType
{
    Traffic,
    Video_Call,
    Building_Security
};

struct Metrics
{
    float requestRate = 0;
    double cpuUsage = 0;
    long memUsage = 0;
    unsigned int gpuUsage = 0;
    unsigned int gpuMemUsage = 0;
};

namespace TaskDescription
{
    struct TaskStruct
    {
        std::string name;
        int slo;
        PipelineType type;
        std::string source;
        std::string device;
    };

    void to_json(nlohmann::json &j, const TaskStruct &val);

    void from_json(const nlohmann::json &j, TaskStruct &val);
}

/**
 * @brief scheduling policy logic
 *
 */
class Controller
{
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
    struct NodeHandle
    {
        std::string ip;
        std::shared_ptr<ControlCommunication::Stub> stub;
        CompletionQueue *cq;
        SystemDeviceType type;
        int num_processors;                        // number of processing units, 1 for Edge or # GPUs for server
        std::vector<double> processors_utilization; // utilization per pu
        std::vector<unsigned long> mem_size;       // memory size in MB
        std::vector<double> mem_utilization;       // memory utilization per pu
        int next_free_port;
        std::map<std::string, ContainerHandle *> containers;
    };

    struct TaskHandle
    {
        int slo;
        PipelineType type;
        std::map<std::string, ContainerHandle *> subtasks;
    };

    struct ContainerHandle
    {
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

    class RequestHandler
    {
    public:
        RequestHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq, Controller *c)
            : service(service), cq(cq), status(CREATE), controller(c), responder(&ctx) {}

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus
        {
            CREATE,
            PROCESS,
            FINISH
        };
        ControlCommunication::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        Controller *controller;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };



    class DeviseAdvertisementHandler : public RequestHandler
    {
    public:
        DeviseAdvertisementHandler(ControlCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
            : RequestHandler(service, cq, c)
        {
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

/*
Jellyfish controller implementation. By Yuheng.

Compared with the original jellyfish paper, we only consider the workload distribution part which means
the following code implement 1.data adaptation, 2.client-DNN mapping 3. dynamic batching
*/

/**
 * @brief jellyfish scheduler implementation
 *
 */
class JellyfishScheduler
{
    // goal: (1) client-dnn mapping (2) interact with clients
    // steps:
    // 1.take the acc-latency profiles and client info
    // 2. periodically update the client-dnn mapping, batch size for each worker and the input size of each client
    // 3. notify the client with corresponding input size, distribute the mapping and batch size for workers
    // info exchange:
    // client: (1) inference request rate (2) estimated network bandwidth (3) SLO
    // server: (1) input size
};

/**
 * @brief jellyfish dispatcher implementation
 *
 */
class JellyfishDispatcher
{
    // goal: maintain worker pool for model deployment
    // steps:
    // 1. fetch the client-dnn mapping from scheduler
    // 2. redirect requests to the workers
};

/**
 * @brief jellyfish scheduling logic: composed of scheduler and dispatcher
 *
 */
class JellyfishController
{
    // compose scheduler and dispatcher
};

/**
 * @brief comparison of the key of ModelProfiles
 *
 */
struct ModelSetCompare
{
    bool operator()(const std::tuple<std::string, float> &lhs, const std::tuple<std::string, float> &rhs) const
    {
        return std::get<1>(lhs) < std::get<1>(rhs);
    }
};

struct ModelInfo
{
    int batch_size;
    float inferent_latency;
    int throughput;
};

class ModelProfiles
{
public:
    // key: (model type, accuracy) value: (model_info)
    std::map<std::tuple<std::string, float>, std::vector<ModelInfo>, ModelSetCompare>
        infos;

    void add(std::string model_type, float accuracy, int batch_size, float inference_latency, int throughput);
};

struct ClientInfo
{
    std::string ip;
    float budget;
    // std::tuple<int, int> input_size;
    int req_rate;

    bool operator==(const ClientInfo &other) const
    {
        return ip == other.ip &&
               budget == other.budget &&
               req_rate == other.req_rate;
    }
};

class ClientProfiles
{
public:
    std::vector<ClientInfo> infos;

    void sortBudgetDescending(std::vector<ClientInfo> &clients);
    void add(const std::string &ip, float budget, int req_rate);
};

std::vector<std::tuple<std::tuple<std::string, float>, std::vector<ClientInfo>, int>> mapClient(ClientProfiles client_profile, ModelProfiles model_profiles);
std::vector<ClientInfo> findOptimalClients(const std::vector<ModelInfo> &models, std::vector<ClientInfo> &clients);
int check_and_assign(std::vector<ModelInfo> &model, std::vector<ClientInfo> &selected_clients);

// ================ helper functions ====================

int findMaxBatchSize(const std::vector<ModelInfo> &models, ClientInfo &client);
void differenceClients(std::vector<ClientInfo> &src, const std::vector<ClientInfo> &diff);

#endif // PIPEPLUSPLUS_CONTROLLER_H
