#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <grpcpp/grpcpp.h>
#include "../utils/json.h"
#include <thread>
#include "controlcommands.grpc.pb.h"
#include "controlmessages.grpc.pb.h"
#include <pqxx/pqxx>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <random>

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using controlcommands::ControlCommands;
using controlcommands::LoopRange;
using controlcommands::ContainerConfig;
using controlcommands::TimeKeeping;
using controlcommands::ContainerLink;
using controlcommands::ContainerInts;
using controlcommands::ContainerSignal;
using controlmessages::ControlMessages;
using controlmessages::ConnectionConfigs;
using controlmessages::SystemInfo;
using controlmessages::DummyMessage;
using EmptyMessage = google::protobuf::Empty;

ABSL_DECLARE_FLAG(std::string, ctrl_configPath);
ABSL_DECLARE_FLAG(uint16_t, ctrl_verbose);
ABSL_DECLARE_FLAG(uint16_t, ctrl_loggingMode);

// typedef std::vector<std::pair<ModelType, std::vector<std::pair<ModelType, int>>>> Pipeline;


struct ContainerHandle;
struct PipelineModel;

struct GPUPortion;
struct GPUHandle;
struct NodeHandle;

struct GPUPortionList {
    GPUPortion *head = nullptr;
    std::vector<GPUPortion *> list;
};

struct GPULane {
    GPUHandle *gpuHandle;
    NodeHandle *node;
    std::uint16_t laneNum;
    std::uint64_t dutyCycle = 0;

    GPUPortionList portionList;

    GPULane() = default;
    GPULane(GPUHandle *gpuHandle, NodeHandle *node, std::uint16_t laneNum);

    bool removePortion(GPUPortion *portion);
};

struct GPUPortion {
    std::uint64_t start = 0;
    std::uint64_t end = MAX_PORTION_SIZE;
    ContainerHandle *container = nullptr;
    GPULane * lane = nullptr;
    // The next portion in the device's global sorted list
    GPUPortion* next = nullptr;
    // The prev portion in the device's global sorted list
    GPUPortion* prev = nullptr;
    // The next portion in the lane, used to quickly recover the lane's original structure
    // When a container is removed and its portion is freed
    GPUPortion* nextInLane = nullptr;
    // The prev portion in the lane, used to quickly recover the lane's original structure
    // When a container is removed and its portion is freed
    GPUPortion* prevInLane = nullptr;
    std::uint64_t getLength() const { return end - start; }

    GPUPortion() = default;
    // ~GPUPortion();
    GPUPortion(GPULane *lane) : lane(lane) {}
    GPUPortion(std::uint64_t start, std::uint64_t end, ContainerHandle *container, GPULane *lane)
        : start(start), end(end), container(container), lane(lane) {}

    bool assignContainer(ContainerHandle *container);
};

struct GPUHandle {
    std::string type;
    std::string hostName;
    std::uint16_t number;
    MemUsageType currentMemUsage = 0;
    MemUsageType memLimit = 9999999; // MB
    std::uint16_t numLanes;

    std::map<std::string, ContainerHandle *> containers = {};
    // TODO: HANDLE FREE PORTIONS WITHIN THE GPU
    // std::vector<GPUPortion *> freeGPUPortions;
    NodeHandle *node;

    GPUHandle() = default;

    GPUHandle(const std::string &type, const std::string &hostName, std::uint16_t number, MemUsageType memLimit, std::uint16_t numLanes, NodeHandle *node)
        : type(type), hostName(hostName), number(number), memLimit(memLimit), numLanes(numLanes), node(node) {}

    bool addContainer(ContainerHandle *container);
    bool removeContainer(ContainerHandle *container);
};

// Structure that whole information about the pipeline used for scheduling
typedef std::vector<PipelineModel *> PipelineModelListType;

struct TaskHandle;
struct NodeHandle {
    std::string name;
    std::string ip;
    std::shared_ptr<ControlCommands::Stub> stub;
    CompletionQueue *cq;
    SystemDeviceType type;
    int next_free_port;
    std::map<std::string, ContainerHandle *> containers;
    // The latest network entries to determine the network conditions and latencies of transferring data
    std::map<std::string, NetworkEntryType> latestNetworkEntries = {};
    // GPU Handle;
    std::vector<GPUHandle*> gpuHandles;
    //
    uint8_t numGPULanes;
    //
    std::vector<GPULane *> gpuLanes;
    GPUPortionList freeGPUPortions;

    bool initialNetworkCheck = false;
    ClockType lastNetworkCheckTime;

    std::map<std::string, PipelineModel *> modelList;

    mutable std::mutex nodeHandleMutex;
    mutable std::mutex networkCheckMutex;

    NodeHandle() = default;

    NodeHandle(const std::string& name,
               const std::string& ip,
               std::shared_ptr<ControlCommands::Stub> stub,
               grpc::CompletionQueue* cq,
               SystemDeviceType type,
               int next_free_port,
               std::map<std::string, ContainerHandle*> containers)
        : name(name),
          ip(ip),
          stub(std::move(stub)),
          cq(cq),
          type(type),
          next_free_port(next_free_port),
          containers(std::move(containers)) {}

    NodeHandle(const NodeHandle &other) {
        std::lock(nodeHandleMutex, other.nodeHandleMutex);
        std::lock_guard<std::mutex> lock1(nodeHandleMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.nodeHandleMutex, std::adopt_lock);
        name = other.name;
        ip = other.ip;
        stub = other.stub;
        cq = other.cq;
        type = other.type;
        next_free_port = other.next_free_port;
        containers = other.containers;
        latestNetworkEntries = other.latestNetworkEntries;
    }

    NodeHandle& operator=(const NodeHandle &other) {
        if (this != &other) {
            std::lock(nodeHandleMutex, other.nodeHandleMutex);
            std::lock_guard<std::mutex> lock1(nodeHandleMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.nodeHandleMutex, std::adopt_lock);
            name = other.name;
            ip = other.ip;
            stub = other.stub;
            cq = other.cq;
            type = other.type;
            next_free_port = other.next_free_port;
            containers = other.containers;
            latestNetworkEntries = other.latestNetworkEntries;
        }
        return *this;
    }
};

struct ContainerHandle {
    std::string name;
    unsigned int replica_id;
    int class_of_interest;
    ModelType model;
    bool mergable;
    std::vector<int> dimensions;
    uint64_t pipelineSLO;

    float arrival_rate;

    int batch_size;
    int recv_port;
    std::string model_file;

    NodeHandle *device_agent;
    TaskHandle *task;
    std::vector<ContainerHandle *> downstreams;
    std::vector<ContainerHandle *> upstreams;
    // Queue sizes of the model
    std::vector<QueueLengthType> queueSizes;

    // Flag to indicate whether the container is running
    // At the end of scheduling, all containerhandle marked with `running = false` object will be deleted
    bool running = false;

    // Number of microservices packed inside this container. A regular container has 5 namely
    // receiver, preprocessor, inferencer, postprocessor, sender
    uint8_t numMicroservices = 5;
    // Average latency to query to reach from the upstream
    uint64_t expectedTransferLatency = 0;
    // Average in queueing latency, subjected to the arrival rate and processing rate of preprocessor
    uint64_t expectedQueueingLatency = 0;
    // Average batching latency, subjected to the preprocessing rate, batch size and processing rate of inferencer
    uint64_t expectedBatchingLatency = 0;
    // Average post queueing latency, subjected to the processing rate of postprocessor
    uint64_t expectedPostQueueingLatency = 0;
    // Average out queueing latency, subjected to the processing rate of sender
    uint64_t expectedOutQueueingLatency = 0;
    // Average latency to preprocess each query
    uint64_t expectedPreprocessLatency = 0;
    // Average latency to process each batch running at the specified batch size
    uint64_t expectedInferLatency = 0;
    // Average latency to postprocess each query
    uint64_t expectedPostprocessLatency = 0;
    // Expected throughput
    float expectedThroughput = 0;
    //
    uint64_t startTime;
    //
    uint64_t endTime;
    //
    uint64_t localDutyCycle = 0;
    // 
    ClockType cycleStartTime;
    // GPU Handle
    GPUHandle *gpuHandle = nullptr;
    //
    GPUPortion *executionPortion = nullptr;
    // points to the pipeline model that this container is part of
    PipelineModel *pipelineModel = nullptr;

    uint64_t timeBudgetLeft = 9999999999;
    mutable std::mutex containerHandleMutex;

    ContainerHandle() = default;

        // Constructor
    ContainerHandle(const std::string& name,
                unsigned int replica_id,
                int class_of_interest,
                ModelType model,
                bool mergable,
                const std::vector<int>& dimensions = {},
                uint64_t pipelineSLO = 0,
                float arrival_rate = 0.0f,
                const int batch_size = 0,
                const int recv_port = 0,
                const std::string model_file = "",
                NodeHandle* device_agent = nullptr,
                TaskHandle* task = nullptr,
                PipelineModel* pipelineModel = nullptr,
                const std::vector<ContainerHandle*>& upstreams = {},
                const std::vector<ContainerHandle*>& downstreams = {},
                const std::vector<QueueLengthType>& queueSizes = {},
                uint64_t timeBudgetLeft = 9999999999)
    : name(name),
      replica_id(replica_id),
      class_of_interest(class_of_interest),
      model(model),
      mergable(mergable),
      dimensions(dimensions),
      pipelineSLO(pipelineSLO),
      arrival_rate(arrival_rate),
      batch_size(batch_size),
      recv_port(recv_port),
      model_file(model_file),
      device_agent(device_agent),
      task(task),
      downstreams(downstreams),
      upstreams(upstreams),
      queueSizes(queueSizes),
      pipelineModel(pipelineModel),
      timeBudgetLeft(timeBudgetLeft) {}
    
    // Copy constructor
    ContainerHandle(const ContainerHandle& other) {
        std::lock(containerHandleMutex, other.containerHandleMutex);
        std::lock_guard<std::mutex> lock1(containerHandleMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.containerHandleMutex, std::adopt_lock);

        name = other.name;
        replica_id = other.replica_id;
        class_of_interest = other.class_of_interest;
        model = other.model;
        mergable = other.mergable;
        dimensions = other.dimensions;
        pipelineSLO = other.pipelineSLO;
        arrival_rate = other.arrival_rate;
        batch_size = other.batch_size;
        recv_port = other.recv_port;
        model_file = other.model_file;
        device_agent = other.device_agent;
        task = other.task;
        upstreams = other.upstreams;
        downstreams = other.downstreams;
        queueSizes = other.queueSizes;
        running = other.running;
        numMicroservices = other.numMicroservices;
        expectedTransferLatency = other.expectedTransferLatency;
        expectedQueueingLatency = other.expectedQueueingLatency;
        expectedBatchingLatency = other.expectedBatchingLatency;
        expectedPostQueueingLatency = other.expectedPostQueueingLatency;
        expectedOutQueueingLatency = other.expectedOutQueueingLatency;
        expectedPreprocessLatency = other.expectedPreprocessLatency;
        expectedInferLatency = other.expectedInferLatency;
        expectedPostprocessLatency = other.expectedPostprocessLatency;
        expectedThroughput = other.expectedThroughput;
        startTime = other.startTime;
        endTime = other.endTime;
        localDutyCycle = other.localDutyCycle;
        gpuHandle = other.gpuHandle;
        executionPortion = other.executionPortion;
        pipelineModel = other.pipelineModel;
        timeBudgetLeft = other.timeBudgetLeft;
    }

    // Copy assignment operator
    ContainerHandle& operator=(const ContainerHandle& other) {
        if (this != &other) {
            std::lock(containerHandleMutex, other.containerHandleMutex);
            std::lock_guard<std::mutex> lock1(containerHandleMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.containerHandleMutex, std::adopt_lock);
            name = other.name;
            replica_id = other.replica_id;
            class_of_interest = other.class_of_interest;
            model = other.model;
            mergable = other.mergable;
            dimensions = other.dimensions;
            pipelineSLO = other.pipelineSLO;
            arrival_rate = other.arrival_rate;
            batch_size = other.batch_size;
            recv_port = other.recv_port;
            model_file = other.model_file;
            device_agent = other.device_agent;
            task = other.task;
            upstreams = other.upstreams;
            downstreams = other.downstreams;
            queueSizes = other.queueSizes;
            running = other.running;
            numMicroservices = other.numMicroservices;
            expectedTransferLatency = other.expectedTransferLatency;
            expectedQueueingLatency = other.expectedQueueingLatency;
            expectedBatchingLatency = other.expectedBatchingLatency;
            expectedPostQueueingLatency = other.expectedPostQueueingLatency;
            expectedOutQueueingLatency = other.expectedOutQueueingLatency;
            expectedPreprocessLatency = other.expectedPreprocessLatency;
            expectedInferLatency = other.expectedInferLatency;
            expectedPostprocessLatency = other.expectedPostprocessLatency;
            expectedThroughput = other.expectedThroughput;
            startTime = other.startTime;
            endTime = other.endTime;
            localDutyCycle = other.localDutyCycle;
            gpuHandle = other.gpuHandle;
            executionPortion = other.executionPortion;
            pipelineModel = other.pipelineModel;
            timeBudgetLeft = other.timeBudgetLeft;
        }
        return *this;
    }

    MemUsageType getExpectedTotalMemUsage() const;
};

struct PipelineModel {
    std::string name;
    TaskHandle *task;
    // Whether the upstream is on another device
    bool isSplitPoint;
    //
    ModelArrivalProfile arrivalProfiles;
    // Latency profile of preprocessor, batch inferencer and postprocessor
    PerDeviceModelProfileType processProfiles;
    // The downstream models and their classes of interest
    std::vector<std::pair<PipelineModel *, int>> downstreams;
    std::vector<std::pair<PipelineModel *, int>> upstreams;
    // The batch size of the model
    BatchSizeType batchSize;
    // The number of replicas of the model
    uint8_t numReplicas = -1;
    // The assigned cuda device for each replica
    std::vector<uint8_t> cudaDevices;
    // Average latency to query to reach from the upstream
    uint64_t expectedTransferLatency = 0;
    // Average queueing latency, subjected to the arrival rate and processing rate of preprocessor
    uint64_t expectedQueueingLatency = 0;
    // Average batching latency, subjected to the preprocessing rate, batch size and processing rate of inferencer
    uint64_t expectedBatchingLatency = 0;
    // Average post queueing latency, subjected to the processing rate of postprocessor
    uint64_t expectedPostQueueingLatency = 0;
    // Average out queueing latency, subjected to the processing rate of sender
    uint64_t expectedOutQueueingLatency = 0;
    // Average latency to process each query
    uint64_t expectedAvgPerQueryLatency = 0;
    // Maximum latency to process each query as ones that come later have to wait to be processed in batch
    uint64_t expectedMaxProcessLatency = 0;
    // Latency from the start of the pipeline until the end of this model
    uint64_t expectedStart2HereLatency = -1;
    // The estimated cost per query processed by this model
    uint64_t estimatedPerQueryCost = 0;
    // The estimated latency of the model
    uint64_t estimatedStart2HereCost = 0;
    uint64_t startTime = 0;
    uint64_t endTime = 0;
    uint64_t localDutyCycle = 0;

    std::vector<int> dimensions = {-1, -1};

    std::string device;
    std::string deviceTypeName;
    NodeHandle *deviceAgent;

    bool merged = false;
    bool toBeRun = true;
    bool gpuScheduled = false;

    std::vector<std::string> possibleDevices;
    // Manifestations are the list of containers that will be created for this model
    std::vector<ContainerHandle *> manifestations;

    // Source
    std::string datasourceName;

    uint64_t timeBudgetLeft = 9999999999;

    // The time when the last scaling or scheduling operation was performed
    ClockType lastScaleTime = std::chrono::system_clock::now();
    //
    int8_t numInstancesScaledLastTime = 0;

    mutable std::mutex pipelineModelMutex;

        // Constructor with default parameters
    PipelineModel(const std::string& device = "",
                  const std::string& name = "",
                  TaskHandle *task = nullptr,
                  bool isSplitPoint = false,
                  const ModelArrivalProfile& arrivalProfiles = ModelArrivalProfile(),
                  const PerDeviceModelProfileType& processProfiles = PerDeviceModelProfileType(),
                  const std::vector<std::pair<PipelineModel*, int>>& downstreams = {},
                  const std::vector<std::pair<PipelineModel*, int>>& upstreams = {},
                  const BatchSizeType& batchSize = BatchSizeType(),
                  uint8_t numReplicas = -1,
                  std::vector<uint8_t> cudaDevices = {},
                  uint64_t expectedTransferLatency = 0,
                  uint64_t expectedQueueingLatency = 0,
                  uint64_t expectedAvgPerQueryLatency = 0,
                  uint64_t expectedMaxProcessLatency = 0,
                  const std::string& deviceTypeName = "",
                  bool merged = false,
                  bool toBeRun = true,
                  uint64_t timeBudgetLeft = 9999999999,
                  const std::vector<std::string>& possibleDevices = {})
        :
          name(name),
          task(task),
          isSplitPoint(isSplitPoint),
          arrivalProfiles(arrivalProfiles),
          processProfiles(processProfiles),
          downstreams(downstreams),
          upstreams(upstreams),
          batchSize(batchSize),
          numReplicas(numReplicas),
          cudaDevices(cudaDevices),
          expectedTransferLatency(expectedTransferLatency),
          expectedQueueingLatency(expectedQueueingLatency),
          expectedAvgPerQueryLatency(expectedAvgPerQueryLatency),
          expectedMaxProcessLatency(expectedMaxProcessLatency),
          device(device),
          deviceTypeName(deviceTypeName),
          merged(merged),
          toBeRun(toBeRun),
          possibleDevices(possibleDevices),
          timeBudgetLeft(timeBudgetLeft) {}

    // Copy constructor
    PipelineModel(const PipelineModel& other) {
        std::lock(pipelineModelMutex, other.pipelineModelMutex);
        std::lock_guard<std::mutex> lock1(other.pipelineModelMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(pipelineModelMutex, std::adopt_lock);
        device = other.device;
        name = other.name;
        task = other.task;
        isSplitPoint = other.isSplitPoint;
        arrivalProfiles = other.arrivalProfiles;
        processProfiles = other.processProfiles;
        downstreams = other.downstreams;
        upstreams = other.upstreams;
        batchSize = other.batchSize;
        numReplicas = other.numReplicas;
        cudaDevices = other.cudaDevices;
        expectedTransferLatency = other.expectedTransferLatency;
        expectedQueueingLatency = other.expectedQueueingLatency;
        expectedBatchingLatency = other.expectedBatchingLatency;
        expectedPostQueueingLatency = other.expectedPostQueueingLatency;
        expectedOutQueueingLatency = other.expectedOutQueueingLatency;
        expectedAvgPerQueryLatency = other.expectedAvgPerQueryLatency;
        expectedMaxProcessLatency = other.expectedMaxProcessLatency;
        expectedStart2HereLatency = other.expectedStart2HereLatency;
        estimatedPerQueryCost = other.estimatedPerQueryCost;
        estimatedStart2HereCost = other.estimatedStart2HereCost;
        startTime = other.startTime;
        endTime = other.endTime;
        localDutyCycle = other.localDutyCycle;
        deviceTypeName = other.deviceTypeName;
        merged = other.merged;
        toBeRun = other.toBeRun;
        timeBudgetLeft = other.timeBudgetLeft;
        possibleDevices = other.possibleDevices;
        dimensions = other.dimensions;
        manifestations = {};
        for (auto& container : other.manifestations) {
            manifestations.push_back(new ContainerHandle(*container));
        }
        deviceAgent = other.deviceAgent;
        datasourceName = other.datasourceName;
    }

    // Assignment operator
    PipelineModel& operator=(const PipelineModel& other) {
        if (this != &other) {
            std::lock(pipelineModelMutex, other.pipelineModelMutex);
            std::lock_guard<std::mutex> lock1(pipelineModelMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.pipelineModelMutex, std::adopt_lock);
            device = other.device;
            name = other.name;
            task = other.task;
            isSplitPoint = other.isSplitPoint;
            arrivalProfiles = other.arrivalProfiles;
            processProfiles = other.processProfiles;
            downstreams = other.downstreams;
            upstreams = other.upstreams;
            batchSize = other.batchSize;
            numReplicas = other.numReplicas;
            cudaDevices = other.cudaDevices;
            expectedTransferLatency = other.expectedTransferLatency;
            expectedQueueingLatency = other.expectedQueueingLatency;
            expectedBatchingLatency = other.expectedBatchingLatency;
            expectedPostQueueingLatency = other.expectedPostQueueingLatency;
            expectedOutQueueingLatency = other.expectedOutQueueingLatency;
            expectedAvgPerQueryLatency = other.expectedAvgPerQueryLatency;
            expectedMaxProcessLatency = other.expectedMaxProcessLatency;
            expectedStart2HereLatency = other.expectedStart2HereLatency;
            estimatedPerQueryCost = other.estimatedPerQueryCost;
            estimatedStart2HereCost = other.estimatedStart2HereCost;
            startTime = other.startTime;
            endTime = other.endTime;
            localDutyCycle = other.localDutyCycle;
            deviceTypeName = other.deviceTypeName;
            merged = other.merged;
            toBeRun = other.toBeRun;
            timeBudgetLeft = other.timeBudgetLeft;
            possibleDevices = other.possibleDevices;
            dimensions = other.dimensions;
            manifestations = {};
            for (auto& container : other.manifestations) {
                manifestations.push_back(new ContainerHandle(*container));
            }
            deviceAgent = other.deviceAgent;
            datasourceName = other.datasourceName;
        }
        return *this;
    }
};

PipelineModelListType deepCopyPipelineModelList(const PipelineModelListType& original);

struct TaskHandle {
    std::string tk_name;
    std::string tk_fullName;
    PipelineType tk_type;
    std::string tk_source;
    std::string tk_src_device;
    int tk_slo;
    ClockType tk_startTime;
    int tk_lastLatency;
    std::map<std::string, std::vector<ContainerHandle*>> tk_subTasks;
    PipelineModelListType tk_pipelineModels;
    mutable std::mutex tk_mutex;

    bool tk_newlyAdded = true;

    TaskHandle() = default;

    ~TaskHandle() {
        // Ensure no other threads are using this object
        std::lock_guard<std::mutex> lock(tk_mutex);
        for (auto& model : tk_pipelineModels) {
            delete model;
        }
    }

    TaskHandle(const std::string& tk_name,
               PipelineType tk_type,
               const std::string& tk_source,
               const std::string& tk_src_device,
               int tk_slo,
               ClockType tk_startTime,
               int tk_lastLatency)
    : tk_name(tk_name),
      tk_type(tk_type),
      tk_source(tk_source),
      tk_src_device(tk_src_device),
      tk_slo(tk_slo),
      tk_startTime(tk_startTime),
      tk_lastLatency(tk_lastLatency) {}

    TaskHandle(const TaskHandle& other) {
        std::lock(tk_mutex, other.tk_mutex);
        std::lock_guard<std::mutex> lock1(other.tk_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(tk_mutex, std::adopt_lock);
        tk_name = other.tk_name;
        tk_fullName = other.tk_fullName;
        tk_type = other.tk_type;
        tk_source = other.tk_source;
        tk_src_device = other.tk_src_device;
        tk_slo = other.tk_slo;
        tk_startTime = other.tk_startTime;
        tk_lastLatency = other.tk_lastLatency;
        tk_subTasks = other.tk_subTasks;
        tk_pipelineModels = {};
        for (auto& model : other.tk_pipelineModels) {
            tk_pipelineModels.push_back(new PipelineModel(*model));
            tk_pipelineModels.back()->task = this;
        }
        for (auto& model : this->tk_pipelineModels) {
            for (auto& downstream : model->downstreams) {
                for (auto& model2 : tk_pipelineModels) {
                    if (model2->name != downstream.first->name || model2->device != downstream.first->device) {
                        continue;
                    }
                    downstream.first = model2;
                }
            }
            for (auto& upstream : model->upstreams) {
                for (auto& model2 : tk_pipelineModels) {
                    if (model2->name != upstream.first->name || model2->device != upstream.first->device) {
                        continue;
                    }
                    upstream.first = model2;
                }
            }
        }
        tk_newlyAdded = other.tk_newlyAdded;
    }

    TaskHandle& operator=(const TaskHandle& other) {
        if (this != &other) {
            std::lock(tk_mutex, other.tk_mutex);
            std::lock_guard<std::mutex> lock1(tk_mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.tk_mutex, std::adopt_lock);
            tk_name = other.tk_name;
            tk_fullName = other.tk_fullName;
            tk_type = other.tk_type;
            tk_source = other.tk_source;
            tk_src_device = other.tk_src_device;
            tk_slo = other.tk_slo;
            tk_startTime = other.tk_startTime;
            tk_lastLatency = other.tk_lastLatency;
            tk_subTasks = other.tk_subTasks;
            tk_pipelineModels = {};
            for (auto& model : other.tk_pipelineModels) {
                tk_pipelineModels.push_back(new PipelineModel(*model));
                tk_pipelineModels.back()->task = this;
            }
            for (auto& model : this->tk_pipelineModels) {
                for (auto& downstream : model->downstreams) {
                    for (auto& model2 : tk_pipelineModels) {
                        if (model2->name != downstream.first->name || model2->device != downstream.first->device) {
                            continue;
                        }
                        downstream.first = model2;
                    }
                }
                for (auto& upstream : model->upstreams) {
                    for (auto& model2 : tk_pipelineModels) {
                        if (model2->name != upstream.first->name || model2->device != upstream.first->device) {
                            continue;
                        }
                        upstream.first = model2;
                    }
                }
            }
            tk_newlyAdded = other.tk_newlyAdded;
        }
        return *this;
    }
};

namespace TaskDescription {
    struct TaskStruct {
        // Name of the task (e.g., traffic, video_call, people, etc.)
        std::string name;
        // Full name to identify the task in the task list (which is a map)
        std::string fullName;
        int slo;
        PipelineType type;
        std::string source;
        std::string device;
        bool added = false;
    };

    void from_json(const nlohmann::json &j, TaskStruct &val);
}

struct Devices {
public:
    void addDevice(const std::string &name, NodeHandle *node) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        list[name] = node;
    }

    void removeDevice(const std::string &name) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        list.erase(name);
    }

    NodeHandle *getDevice(const std::string &name) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        return list[name];
    }

    std::vector<NodeHandle *> getList() {
        std::lock_guard<std::mutex> lock(devicesMutex);
        std::vector<NodeHandle *> elements;
        for (auto &d: list) {
            elements.push_back(d.second);
        }
        return elements;
    }

    std::map<std::string, NodeHandle*> getMap() {
        std::lock_guard<std::mutex> lock(devicesMutex);
        return list;
    }

    bool hasDevice(const std::string &name) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        return list.find(name) != list.end();
    }
private:
    std::map<std::string, NodeHandle*> list = {};
    std::mutex devicesMutex;
};

struct Tasks {
public:
    void addTask(const std::string &name, TaskHandle *task) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        list[name] = task;
    }

    void removeTask(const std::string &name) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        list.erase(name);
    }

    TaskHandle *getTask(const std::string &name) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        return list[name];
    }

    std::vector<TaskHandle *> getList() {
        std::lock_guard<std::mutex> lock(tasksMutex);
        std::vector<TaskHandle *> tasks;
        for (auto &t: list) {
            tasks.push_back(t.second);
        }
        return tasks;
    }

    std::map<std::string, TaskHandle*> getMap() {
        std::lock_guard<std::mutex> lock(tasksMutex);
        return list;
    }

    bool hasTask(const std::string &name) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        return list.find(name) != list.end();
    }

    Tasks() = default;

    // Copy constructor
    Tasks(const Tasks &other) {
        std::lock(tasksMutex, other.tasksMutex);
        std::lock_guard<std::mutex> lock1(tasksMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.tasksMutex, std::adopt_lock);
        list = {};
        for (auto &t: other.list) {
            list[t.first] = new TaskHandle(*t.second);
        }
    }

    Tasks& operator=(const Tasks &other) {
        if (this != &other) {
            std::lock(tasksMutex, other.tasksMutex);
            std::lock_guard<std::mutex> lock1(tasksMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.tasksMutex, std::adopt_lock);
            list = {};
            for (auto &t: other.list) {
                list[t.first] = new TaskHandle(*t.second);
            }
        }
        return *this;
    }

private:
    std::map<std::string, TaskHandle*> list = {};
    mutable std::mutex tasksMutex;
};

struct Containers {
public:
    void addContainer(const std::string &name, ContainerHandle *container) {
        std::lock_guard<std::mutex> lock(containersMutex);
        list[name] = container;
    }

    void removeContainer(const std::string &name) {
        std::lock_guard<std::mutex> lock(containersMutex);
        list.erase(name);
    }

    ContainerHandle *getContainer(const std::string &name) {
        std::lock_guard<std::mutex> lock(containersMutex);
        return list[name];
    }

    std::vector<ContainerHandle *> getList() {
        std::lock_guard<std::mutex> lock(containersMutex);
        std::vector<ContainerHandle *> elements;
        for (auto &c: list) {
            elements.push_back(c.second);
        }
        return elements;
    }

    std::map<std::string, ContainerHandle *> getMap() {
        std::lock_guard<std::mutex> lock(containersMutex);
        return list;
    }

    bool hasContainer(const std::string &name) {
        std::lock_guard<std::mutex> lock(containersMutex);
        return list.find(name) != list.end();
    }

private:
    std::map<std::string, ContainerHandle*> list = {};
    std::mutex containersMutex;
};

class Controller {
public:
    Controller(int argc, char **argv);

    ~Controller();

    void HandleRecvRpcs();

    void Scheduling();

    void Rescaling();

    void Init() {
        for (auto &t: initialTasks) {
            if (!t.added) {
                t.added = AddTask(t);
            }
            if (!t.added) {
                remainTasks.push_back(t);
            }
        }
        isPipelineInitialised = true;
    }

    void InitRemain() {
        for (auto &t: remainTasks) {
            if (!t.added) {
                t.added = AddTask(t);
            }
            if (t.added) {
                // Remove the task from the remain list
                remainTasks.erase(std::remove_if(remainTasks.begin(), remainTasks.end(),
                                                [&t](const TaskDescription::TaskStruct &task) {
                                                    return task.name == t.name;
                                                }), remainTasks.end());
            }
        }
    }

    void addRemainTask(const TaskDescription::TaskStruct &task) {
        remainTasks.push_back(task);
    }

    bool AddTask(const TaskDescription::TaskStruct &task);

    ContainerHandle *TranslateToContainer(PipelineModel *model, NodeHandle *device, unsigned int i);

    void ApplyScheduling();

    void ScaleUp(PipelineModel *model, uint8_t numIncReps);

    void ScaleDown(PipelineModel *model, uint8_t numDecReps);

    [[nodiscard]] bool isRunning() const { return running; };

    void Stop() { running = false; };

    void readInitialObjectCount(
        const std::string& path 
    );

private:
    void initiateGPULanes(NodeHandle &node);

    NetworkEntryType initNetworkCheck(NodeHandle &node, uint32_t minPacketSize = 1000, uint32_t maxPacketSize = 1228800, uint32_t numLoops = 20);
    uint8_t incNumReplicas(const PipelineModel *model);
    uint8_t decNumReplicas(const PipelineModel *model);

    void calculateQueueSizes(ContainerHandle &model, const ModelType modelType);
    uint64_t calculateQueuingLatency(const float &arrival_rate, const float &preprocess_rate);

    void queryingProfiles(TaskHandle *task);

    void estimateModelLatency(PipelineModel *currModel);
    void estimateModelNetworkLatency(PipelineModel *currModel);
    void estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency);

    void estimateModelTiming(PipelineModel *currModel, const uint64_t start2HereDutyCycle);

    void crossDeviceWorkloadDistributor(TaskHandle *task, uint64_t slo);
    void shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string& edgeDevice);

    bool mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile, const std::string &device, const std::string &upstreamDevice);
    bool mergeProcessProfiles(
        PerDeviceModelProfileType &mergedProfile,
        float arrivalRate1,
        const PerDeviceModelProfileType &toBeMergedProfile,
        float arrivalRate2,
        const std::string &device
    );
    bool mergeModels(PipelineModel *mergedModel, PipelineModel *tobeMergedModel, const std::string &device);
    TaskHandle* mergePipelines(const std::string& taskName);
    void mergePipelines();

    bool containerColocationTemporalScheduling(ContainerHandle *container);
    bool modelColocationTemporalScheduling(PipelineModel *pipelineModel, unsigned int replica_id);
    void colocationTemporalScheduling();

    void basicGPUScheduling(std::vector<ContainerHandle *> new_containers);

    PipelineModelListType getModelsByPipelineType(PipelineType type, const std::string &startDevice, const std::string &pipelineName = "", const std::string &streamName = "");

    void checkNetworkConditions();

    void readConfigFile(const std::string &config_path);

    // double LoadTimeEstimator(const char *model_path, double input_mem_size);
    int InferTimeEstimator(ModelType model, int batch_size);
    // std::map<ModelType, std::vector<int>> InitialRequestCount(const std::string &input, const Pipeline &models,
    //                                                           int fps = 30);

    void queryInDeviceNetworkEntries(NodeHandle *node);

    struct TimingControl {
        uint64_t schedulingIntervalSec;
        uint64_t rescalingIntervalSec;
        uint64_t networkCheckIntervalSec;
        uint64_t scaleUpIntervalThresholdSec;
        uint64_t scaleDownIntervalThresholdSec;

        ClockType nextSchedulingTime = std::chrono::system_clock::time_point::min();
        ClockType currSchedulingTime = std::chrono::system_clock::time_point::min();
        ClockType nextRescalingTime = std::chrono::system_clock::time_point::max();
    };

    TimingControl ctrl_controlTimings;

    class RequestHandler {
    public:
        RequestHandler(ControlMessages::AsyncService *service, ServerCompletionQueue *cq, Controller *c)
                : service(service), cq(cq), status(CREATE), controller(c) {}

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        ControlMessages::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        Controller *controller;
    };

    class DeviseAdvertisementHandler : public RequestHandler {
    public:
        DeviseAdvertisementHandler(ControlMessages::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
                : RequestHandler(service, cq, c), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        ConnectionConfigs request;
        SystemInfo reply;
        grpc::ServerAsyncResponseWriter<SystemInfo> responder;
    };

    void initialiseGPU(NodeHandle *node, int numGPUs, std::vector<int> memLimits);

    class DummyDataRequestHandler : public RequestHandler {
    public:
        DummyDataRequestHandler(ControlMessages::AsyncService *service, ServerCompletionQueue *cq,
                                   Controller *c)
                : RequestHandler(service, cq, c), responder(&ctx) {
            Proceed();
        }

        void Proceed() final;

    private:
        DummyMessage request;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    void StartContainer(ContainerHandle *container, bool easy_allocation = true);

    void MoveContainer(ContainerHandle *container, NodeHandle *new_device);

    static void AdjustUpstream(int port, ContainerHandle *upstr, NodeHandle *new_device,
                               const std::string &dwnstr, AdjustUpstreamMode mode = AdjustUpstreamMode::Overwrite);

    static void SyncDatasource(ContainerHandle *prev, ContainerHandle *curr);

    void AdjustBatchSize(ContainerHandle *msvc, int new_bs);

    void AdjustCudaDevice(ContainerHandle *msvc, GPUHandle *new_device);

    void AdjustResolution(ContainerHandle *msvc, std::vector<int> new_resolution);

    void StopContainer(ContainerHandle *container, NodeHandle *device, bool forced = false);

    void AdjustTiming(ContainerHandle *container);

    // void optimizeBatchSizeStep(
    //         const Pipeline &models,
    //         std::map<ModelType, int> &batch_sizes, std::map<ModelType, int> &estimated_infer_times, int nObjects);

    bool running;
    std::string ctrl_experimentName;
    std::string ctrl_systemName;
    std::vector<TaskDescription::TaskStruct> initialTasks;
    std::vector<TaskDescription::TaskStruct> remainTasks;
    uint16_t ctrl_runtime;
    uint16_t ctrl_port_offset;

    std::string ctrl_logPath;
    uint16_t ctrl_loggingMode;
    uint16_t ctrl_verbose;

    ContainerLibType ctrl_containerLib;
    DeviceInfoType ctrl_sysDeviceInfo = {
        {Server, "server"},
        {AGXXavier, "agxavier"},
        {NXXavier, "nxavier"},
        {OrinNano, "orinano"}
    };
    
    Devices devices;

    Tasks ctrl_unscheduledPipelines, ctrl_savedUnscheduledPipelines, ctrl_scheduledPipelines, ctrl_pastScheduledPipelines;

    Containers containers;

    std::map<std::string, NetworkEntryType> network_check_buffer;

    ControlMessages::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<ServerCompletionQueue> cq;

    std::unique_ptr<pqxx::connection> ctrl_metricsServerConn = nullptr;
    MetricsServerConfigs ctrl_metricsServerConfigs;
    std::string ctrl_sinkNodeIP;

    std::vector<spdlog::sink_ptr> ctrl_loggerSinks = {};
    std::shared_ptr<spdlog::logger> ctrl_logger;

    std::map<std::string, NetworkEntryType> ctrl_inDeviceNetworkEntries;

    std::uint64_t ctrl_schedulingIntervalSec;
    ClockType ctrl_nextSchedulingTime = std::chrono::system_clock::now();
    ClockType ctrl_currSchedulingTime = std::chrono::system_clock::now();

    std::map<std::string, std::map<std::string, float>> ctrl_initialRequestRates;

    uint16_t ctrl_systemFPS;

    // Fixed batch sizes for SOTAs that don't provide dynamic batching
    std::map<std::string, BatchSizeType> ctrl_initialBatchSizes;

    std::atomic<bool> isPipelineInitialised = false;

    uint16_t ctrl_numGPULanes = NUM_LANES_PER_GPU * NUM_GPUS, ctrl_numGPUPortions;

    void insertFreeGPUPortion(GPUPortionList &portionList, GPUPortion *freePortion);
    bool removeFreeGPUPortion(GPUPortionList &portionList, GPUPortion *freePortion);
    std::pair<GPUPortion *, GPUPortion *> insertUsedGPUPortion(GPUPortionList &portionList, ContainerHandle *container, GPUPortion *toBeDividedFreePortion);
    bool reclaimGPUPortion(GPUPortion *toBeReclaimedPortion);
    GPUPortion* findFreePortionForInsertion(GPUPortionList &portionList, ContainerHandle *container);
    void estimatePipelineTiming();
    void estimateTimeBudgetLeft(PipelineModel *currModel);

    Tasks ctrl_mergedPipelines;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H