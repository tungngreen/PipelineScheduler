#ifndef PIPEPLUSPLUS_CONTROLLER_H
#define PIPEPLUSPLUS_CONTROLLER_H

#include "microservice.h"
#include <memory>
#include <thread>
#include <pqxx/pqxx>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <random>
#include <vector>
#include "fcpo_learning.h"
#include "bandwidth_predictor/bandwidth_predictor.h"

ABSL_DECLARE_FLAG(std::string, ctrl_configPath);
ABSL_DECLARE_FLAG(uint16_t, ctrl_verbose);
ABSL_DECLARE_FLAG(uint16_t, ctrl_loggingMode);

struct ContainerHandle;
struct PipelineModel;

struct GPUPortion;
struct GPUHandle;
struct NodeHandle;

/**
 * @brief GPUPortion is a structure that represents a portion of the GPU's memory that can be allocated to a container.
 * It contains all the free portions in the system.
 * 
 */
struct GPUPortionList {
    GPUPortion *head = nullptr;
    std::list<GPUPortion *> list;
};

/**
 * @brief LanePortionList represents the portion that belongs to a lane, all of which are present in the GPUPortionList
 * 
 */
struct LanePortionList {
    std::list<std::unique_ptr<GPUPortion>> list;
    // head is just an observation pointer, the unique_ptr in the list owns the memory
    GPUPortion *head = nullptr;
};

/**
 * @brief GPULane represents a stream of execution on a GPU. Containers in a lane are executed sequentially.
 * 
 */
struct GPULane {
    // Non-owning "Observer" pointers
    // These are safe to use as long as we ensure that the GPUHandle and NodeHandle outlive the GPULane, 
    // which is guaranteed by our current design where GPULanes are only created within NodeHandles and reference the GPUHandles within the same NodeHandle.
    GPUHandle *gpuHandle = nullptr;
    NodeHandle *node = nullptr;
    std::uint16_t laneNum;
    std::uint64_t dutyCycle = 0;

    // Lane strictly owns its portions, so we use unique_ptr to manage their memory.
    LanePortionList portionList;

    GPULane() = default;
    GPULane(GPUHandle *gpuHandle, NodeHandle *node, std::uint16_t laneNum) 
        : gpuHandle(gpuHandle), node(node), laneNum(laneNum), dutyCycle(0) {}

    // Delete copy semantics to prevent unique_ptr compiler errors and linked-list corruption
    GPULane(const GPULane&) = delete;
    GPULane& operator=(const GPULane&) = delete;

    bool removePortion(GPUPortion *portion);
};

/**
 * @brief GPUPortion represents a portion of the GPU's memory that can be allocated to a container.
 * It is owned by a lane and is part of the lane's portion list.
 * 
 */
struct GPUPortion {
    std::uint64_t start = 0;
    std::uint64_t end = MAX_PORTION_SIZE;

    // If the controller kills the container, 
    // the GPU logic won't crash when trying to access it.
    std::weak_ptr<ContainerHandle> container;

    GPULane *lane = nullptr;
    // The next portion in the device's global sorted list
    GPUPortion *next = nullptr;
    // The prev portion in the device's global sorted list
    GPUPortion *prev = nullptr;
    // The next portion in the lane, used to quickly recover the lane's original structure
    // When a container is removed and its portion is freed
    GPUPortion *nextInLane = nullptr;
    // The prev portion in the lane, used to quickly recover the lane's original structure
    // When a container is removed and its portion is freed
    GPUPortion *prevInLane = nullptr;
    std::uint64_t getLength() const { return end - start; }

    GPUPortion() = default;
    // ~GPUPortion();
    GPUPortion(GPULane *lane) : lane(lane) {}
    GPUPortion(std::uint64_t start, std::uint64_t end, std::weak_ptr<ContainerHandle> container, GPULane *lane)
        : start(start), end(end), container(container), lane(lane) {}

    bool assignContainer(std::shared_ptr<ContainerHandle> cont);
};

/**
 * @brief GPUHandle represents a GPU device in the system. It contains all the information about the GPU,
 * including its type, its memory usage, its assigned containers, its lanes, etc.
 * 
 */
struct GPUHandle {
    std::string type;
    std::string hostName;
    std::uint16_t number;
    MemUsageType currentMemUsage = 0;
    MemUsageType memLimit = 9999999; // MB
    std::uint16_t numLanes;

    // The GPU observes which containers are assigned to it,
    // but it doesn't prevent them from being deleted by the Controller.
    std::map<std::string, std::weak_ptr<ContainerHandle>> containers = {};
    // TODO: HANDLE FREE PORTIONS WITHIN THE GPU
    // std::vector<GPUPortion *> freeGPUPortions;
    NodeHandle *node;

    GPUHandle() = default;

    GPUHandle(const std::string &type, const std::string &hostName, std::uint16_t number, MemUsageType memLimit, std::uint16_t numLanes, NodeHandle *node)
        : type(type), hostName(hostName), number(number), memLimit(memLimit), numLanes(numLanes), node(node) {}

    // Delete copy semantics to prevent Node parent-pointer corruption
    GPUHandle(const GPUHandle&) = delete;
    GPUHandle& operator=(const GPUHandle&) = delete;

    bool addContainer(std::shared_ptr<ContainerHandle> container);
    bool removeContainer(std::shared_ptr<ContainerHandle> container);
};

// Structure that holds information about the pipeline used for scheduling
typedef std::vector<std::shared_ptr<PipelineModel>> PipelineModelListType;

struct TaskHandle;

/**
 * @brief NodeHandle is the main data structure that the Controller uses to track the state of each device in the system.
 * It contains all the information about the device, including its name, its IP address, its type (server, edge, on-premise), its assigned containers,
 * its GPUs and lanes, its network conditions with other devices, etc.
 * 
 */
struct NodeHandle {
    std::string name;
    std::string ip;
    SystemDeviceType type;
    int next_free_port;
    // Use weak_ptr for containers as node tracks them, but the Controller owns them.
    std::map<std::string, std::weak_ptr<ContainerHandle>> containers;
    // The latest network entries to determine the network conditions and latencies of transferring data
    std::map<std::string, NetworkEntryType> latestNetworkEntries = {};
    // Transmission Latency History is used for bandwidth prediction to forcast network conditions
    std::vector<float> transmissionLatencyHistory;
    float transmissionLatencyPrediction;

    // NodeHandle strictly owns its GPUs and Lanes.
    uint8_t numGPULanes;
    std::vector<std::unique_ptr<GPUHandle>> gpuHandles;    
    std::vector<std::unique_ptr<GPULane>> gpuLanes;

    // freeGPUPortions is just an observation of pointers.
    // It does NOT own the memory, the lanes do.
    GPUPortionList freeGPUPortions;

    // Safely track models as observers
    std::map<std::string, std::weak_ptr<PipelineModel>> modelList;

    bool initialNetworkCheck = false;
    ClockType lastNetworkCheckTime;

    mutable std::mutex nodeHandleMutex;
    mutable std::mutex networkCheckMutex;

    NodeHandle() = default;

    NodeHandle(const std::string& name,
               const std::string& ip,
               SystemDeviceType type,
               int next_free_port,
               const std::map<std::string, std::shared_ptr<ContainerHandle>>& containers_initial)
        : name(name),
          ip(ip),
          type(type),
          next_free_port(next_free_port) {
        
        for (const auto& [name_key, container_ptr] : containers_initial) {
            containers[name_key] = container_ptr; 
        }
    }

    // EXPLICITLY DELETE COPY SEMANTICS
    // Copying a Node deep-copies lanes, which breaks the intrusive linked lists
    // and causes std::unique_ptr compiler errors.
    NodeHandle(const NodeHandle &other) = delete;
    NodeHandle& operator=(const NodeHandle &other) = delete;
};

struct DownstreamAdjustment {
    std::shared_ptr<ContainerHandle> downstreamContainer;
    AdjustMode mode;
    std::string old_link = "";
};

using SingleContainerDownstreamAdjustmentList = std::vector<DownstreamAdjustment>;

using ContainersDnstreamAdjustmentMap = std::map<std::shared_ptr<ContainerHandle>, SingleContainerDownstreamAdjustmentList>;

struct ContainerEdge {
    std::weak_ptr<ContainerHandle> targetContainer;
    // The class of interest
    int classOfInterest;
    // Name of streams allowed to be transferred through this edge.
    std::unordered_set<std::string> streamNames;
};

using ContainerEdgeMap = std::map<std::string, ContainerEdge>;

void cleanUpContainerDownstreamsAdjustmentBatch(SingleContainerDownstreamAdjustmentList &downstreamAdjustmentList);

/**
 * @brief ContainerHandle is the main data structure that the Controller uses to track the state of each container in the system
 * It contains all the information about the container, including its model, its position in the pipeline, its assigned device and GPU, 
 * its neighbors in the pipeline, its expected latencies and throughput, etc.
 *
 * */
struct ContainerHandle {
    std::string name;
    int position_in_pipeline;
    int replica_id;
    int class_of_interest;
    ModelType model;
    bool mergable;
    std::vector<int> dimensions;
    uint64_t pipelineSLO;
    json fcpo_conf = "";

    float arrival_rate;

    BatchSizeType batch_size;
    int recv_port;
    std::string model_file;

    // Observer pointers, without any ownership semantics. 

    // The device/node that the container is assigned to is responsible 
    // for ensuring these pointers remain valid as long as the container exists.
    std::weak_ptr<NodeHandle> device_agent;    
    // GPU that the container is assigned to, nullptr if the container is not assigned to any GPU. Used for scheduling and accessing GPU-level information
    GPUHandle *gpuHandle = nullptr;
    // The portion of the GPU assigned to this container, nullptr if the container is not assigned to any GPU. Used for scheduling and accessing GPU-level information
    GPUPortion *executionPortion = nullptr;
    // The pipeline task that the container is part of, used for scheduling and accessing pipeline-level information
    std::weak_ptr<TaskHandle> task;
    // This container is a manifestation of the PipelineModel, so it observes the PipelineModel for scheduling and accessing pipeline-level information.
    std::weak_ptr<PipelineModel> pipelineModel;    
    // The downstream and upstream containers that this container directly communicates with, 
    // used for scheduling and accessing their information
    // Use weak_ptr for neighbors to prevent circular reference memory leaks
    ContainerEdgeMap downstreams;
    ContainerEdgeMap upstreams;

    // Queue sizes of the model
    std::vector<QueueLengthType> queueSizes;

    // Whether the container is currently running.
    // At the end of each scheduling round, the Controller will check if the container is running and update this flag accordingly.
    // If the container is not running, the Controller will remove it from the system and free up its resources. 
    bool isRunning = false;

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

    uint64_t timeBudgetLeft = 9999999999;
    mutable std::mutex containerHandleMutex;

    ContainerHandle() = default;

    ContainerHandle(const std::string& name,
                int position_in_pipeline,
                unsigned int replica_id,
                int class_of_interest,
                ModelType model,
                bool mergable,
                const std::vector<int>& dimensions = {},
                uint64_t pipelineSLO = 0,
                float arrival_rate = 0.0f,
                const BatchSizeType batch_size = 0,
                const int recv_port = 0,
                const std::string model_file = "",
                std::weak_ptr<NodeHandle> device_agent = std::weak_ptr<NodeHandle>(),
                std::weak_ptr<TaskHandle> task = std::weak_ptr<TaskHandle>(),
                std::weak_ptr<PipelineModel> pipelineModel = std::weak_ptr<PipelineModel>(),
                const ContainerEdgeMap& upstreams = {},
                const ContainerEdgeMap& downstreams = {},
                const std::vector<QueueLengthType>& queueSizes = {},
                uint64_t timeBudgetLeft = 9999999999)
    : name(name),
      position_in_pipeline(position_in_pipeline),
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
      pipelineModel(pipelineModel),
      downstreams(downstreams),
      upstreams(upstreams),
      queueSizes(queueSizes),
      timeBudgetLeft(timeBudgetLeft) {}
    
    // Copy constructor
    ContainerHandle(const ContainerHandle& other) {
        std::lock(containerHandleMutex, other.containerHandleMutex);
        std::lock_guard<std::mutex> lock2(other.containerHandleMutex, std::adopt_lock);

        name = other.name;
        position_in_pipeline = other.position_in_pipeline;
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
        isRunning = other.isRunning;
        queueSizes = other.queueSizes;
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
            position_in_pipeline = other.position_in_pipeline;
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
            isRunning = other.isRunning;
            queueSizes = other.queueSizes;
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


struct PipelineEdge {
    std::weak_ptr<PipelineModel> targetNode;
    // The class of interest for the edge. -1 for everything
    int classOfInterest;
    // Name of streams allowed to be transferred through this edge.
    std::unordered_set<std::string> streamNames;
};


/**
 * @brief PipelineModel is the logical representation of a model in the pipeline graphm,
 * which may have multiple manifestations (containers) running in the system.
 * */
struct PipelineModel {
    std::string name;
    ModelType type;
    std::weak_ptr<TaskHandle> task;
    int position_in_pipeline;
    // Whether the upstream is on another device
    bool isSplitPoint;
    // Wether the container should also send the input data to the downstream
    bool forwardInput;
    //
    ModelArrivalProfile arrivalProfiles;
    // Latency profile of preprocessor, batch inferencer and postprocessor
    PerDeviceModelProfileType processProfiles;
    std::vector<PipelineEdge> downstreams;
    std::vector<PipelineEdge> upstreams;
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
    std::weak_ptr<NodeHandle> deviceAgent;

    bool merged = false;
    bool toBeRun = true;
    bool gpuScheduled = false;
    bool canBeCombined = true;

    std::vector<std::string> possibleDevices;
    
    // Manifestations safely managed by shared_ptr
    std::vector<std::weak_ptr<ContainerHandle>> manifestations;

    // Source
    std::vector<std::string> datasourceName;

    uint64_t timeBudgetLeft = 9999999999;

    // The time when the last scaling or scheduling operation was performed
    ClockType lastScaleTime = std::chrono::system_clock::now();
    //
    int8_t numInstancesScaledLastTime = 0;

    mutable std::mutex pipelineModelMutex;

    // Constructor 
    PipelineModel(const std::string& device = "",
                  const std::string& name = "",
                  ModelType type = ModelType::DataSource,
                  std::weak_ptr<TaskHandle> task = std::weak_ptr<TaskHandle>(),
                  int position_in_pipeline = 0,
                  bool isSplitPoint = false,
                  bool forwardInput = false,
                  const ModelArrivalProfile& arrivalProfiles = ModelArrivalProfile(),
                  const PerDeviceModelProfileType& processProfiles = PerDeviceModelProfileType(),
                  const std::vector<PipelineEdge>& downstreams = {},
                  const std::vector<PipelineEdge>& upstreams = {},
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
          type(type),
          task(task),
          position_in_pipeline(position_in_pipeline),
          isSplitPoint(isSplitPoint),
          forwardInput(forwardInput),
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

    // Destructor removed! std::vector<std::shared_ptr> cleans itself up.

    // Copy constructor
    PipelineModel(const PipelineModel& other) {
        std::lock_guard<std::mutex> lock1(other.pipelineModelMutex, std::adopt_lock);
        device = other.device;
        name = other.name;
        type = other.type;
        task = other.task;
        position_in_pipeline = other.position_in_pipeline;
        isSplitPoint = other.isSplitPoint;
        forwardInput = other.forwardInput;
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
        
        manifestations.clear();
        // for (const auto& container : other.manifestations) {
        //     // Check for valid pointer before dereferencing to prevent segfaults
        //     if (container) {
        //         // Allocate a new ContainerHandle managed by a shared_ptr
        //         manifestations.push_back(std::make_shared<ContainerHandle>(*container));
        //     }
        // }
        
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
            type = other.type;
            task = other.task;
            position_in_pipeline = other.position_in_pipeline;
            isSplitPoint = other.isSplitPoint;
            forwardInput = other.forwardInput;
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
            
            // manifestations.clear() safely drops the reference counts of the old shared_ptrs!
            manifestations.clear();
            
            // for (const auto& container : other.manifestations) {
            //     // Check for valid pointer before dereferencing to prevent segfaults
            //     if (container) {
            //         // Allocate a new ContainerHandle managed by a shared_ptr
            //         manifestations.push_back(std::make_shared<ContainerHandle>(*container));
            //     }
            // }
            
            deviceAgent = other.deviceAgent;
            datasourceName = other.datasourceName;
        }
        return *this;
    }
};

PipelineModelListType deepCopyPipelineModelList(const PipelineModelListType& original);

/**
 * @brief TaskHandle is the logical representation of a pipeline which is a DAG of PipelineModels.
 * 
 */
struct TaskHandle {
    std::string tk_name;
    std::string tk_fullName;
    PipelineType tk_type;
    std::string tk_source;
    std::string tk_src_device;
    std::string tk_edge_node;
    MsvcSLOType tk_slo;
    int tk_memSlo;
    ClockType tk_startTime;
    float tk_lastLatency;
    float tk_lastThroughput;
    
    // weak_ptr to prevent dangling pointers if containers are destroyed
    std::map<std::string, std::vector<std::weak_ptr<ContainerHandle>>> tk_subTasks;
    
    PipelineModelListType tk_pipelineModels;
    mutable std::mutex tk_mutex;

    bool tk_newlyAdded = true;

    TaskHandle() = default;

    TaskHandle(const std::string& tk_name,
               PipelineType tk_type,
               const std::string& tk_source,
               const std::string& tk_src_device,
               int tk_slo,
               ClockType tk_startTime,
               const std::string& tk_edge_node = "server")
    : tk_name(tk_name),
      tk_type(tk_type),
      tk_source(tk_source),
      tk_src_device(tk_src_device),
      tk_edge_node(tk_edge_node),
      tk_slo(tk_slo),
      tk_memSlo(1),
      tk_startTime(tk_startTime),
      tk_lastLatency(0),
      tk_lastThroughput(0.0) {}

    TaskHandle(const TaskHandle& other) {
        std::lock_guard<std::mutex> lock1(other.tk_mutex, std::adopt_lock);
        tk_name = other.tk_name;
        tk_fullName = other.tk_fullName;
        tk_type = other.tk_type;
        tk_source = other.tk_source;
        tk_src_device = other.tk_src_device;
        tk_edge_node = other.tk_edge_node;
        tk_slo = other.tk_slo;
        tk_memSlo = other.tk_memSlo;
        tk_startTime = other.tk_startTime;
        tk_lastLatency = other.tk_lastLatency;
        tk_lastThroughput = other.tk_lastThroughput;
        
        tk_pipelineModels = {};
        for (const auto& model : other.tk_pipelineModels) {
            // Safely allocate new PipelineModel managed by shared_ptr
            auto newModel = std::make_shared<PipelineModel>(*model);
            tk_pipelineModels.push_back(newModel);
            
            // FIXME: Cannot assign raw 'this' to weak_ptr here. 
            // The Controller (Tasks) will assign the 'task' pointer after wrapping this TaskHandle in a shared_ptr.
        }

        // Relink the DAG edges safely using weak_ptr locks and the new PipelineEdge struct
        for (auto& model : this->tk_pipelineModels) {
            for (auto& downstream : model->downstreams) {
                auto old_downstream = downstream.targetNode.lock();
                if (!old_downstream) {
                    downstream.targetNode.reset(); // Reset to prevent dangling pointer
                    continue;
                }
                
                bool found = false;
                for (auto& model2 : tk_pipelineModels) {
                    if (model2->name != old_downstream->name || model2->device != old_downstream->device) {
                        continue;
                    }
                    downstream.targetNode = model2; // Assigns the new shared_ptr safely to the weak_ptr
                    found = true;
                    break;
                }
                if (!found) {
                    downstream.targetNode.reset(); // Reset to prevent dangling pointer if we can't find the downstream model 
                    // in the new list (which shouldn't really happen)
                }
            }
            for (auto& upstream : model->upstreams) {
                auto old_upstream = upstream.targetNode.lock();
                if (!old_upstream) {
                    upstream.targetNode.reset(); // Reset to prevent dangling pointer
                    continue;
                }

                bool found = false;
                for (auto& model2 : tk_pipelineModels) {
                    if (model2->name != old_upstream->name || model2->device != old_upstream->device) {
                        continue;
                    }
                    upstream.targetNode = model2;
                    found = true;
                    break;
                }

                if (!found) {
                    upstream.targetNode.reset(); // Reset to prevent dangling pointer if we can't find the upstream model in the new list
                }
            }
        }

        // Deep-copy tk_subTasks to map to the NEW containers, preventing dangling pointers
        tk_subTasks.clear();
        // for (const auto& kv : other.tk_subTasks) {
        //     std::vector<std::weak_ptr<ContainerHandle>> new_container_list;
        //     for (const auto& old_cont_weak : kv.second) {
        //         auto old_cont = old_cont_weak.lock();
        //         if (!old_cont) continue;

        //         // Find the matching newly created container in the new pipeline models
        //         for (auto& pm : tk_pipelineModels) {
        //             for (auto& new_cont : pm->manifestations) {
        //                 if (new_cont->name == old_cont->name) {
        //                     new_container_list.push_back(new_cont);
        //                 }
        //             }
        //         }
        //     }
        //     tk_subTasks[kv.first] = new_container_list;
        // }

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
            tk_edge_node = other.tk_edge_node;
            tk_slo = other.tk_slo;
            tk_memSlo = other.tk_memSlo;
            tk_startTime = other.tk_startTime;
            tk_lastLatency = other.tk_lastLatency;
            tk_lastThroughput = other.tk_lastThroughput;
            
            // clear() safely drops reference counts of old shared_ptrs!
            tk_pipelineModels.clear();
            for (const auto& model : other.tk_pipelineModels) {
                auto newModel = std::make_shared<PipelineModel>(*model);
                tk_pipelineModels.push_back(newModel);
            }

            // Relink the DAG edges safely using weak_ptr locks and the new PipelineEdge struct
            for (auto& model : this->tk_pipelineModels) {
                for (auto& downstream : model->downstreams) {
                    auto old_downstream = downstream.targetNode.lock();
                    if (!old_downstream) {
                        downstream.targetNode.reset(); // Reset to prevent dangling pointer
                        continue;
                    }

                    bool found = false;
                    for (auto& model2 : tk_pipelineModels) {
                        if (model2->name != old_downstream->name || model2->device != old_downstream->device) {
                            continue;
                        }
                        downstream.targetNode = model2;
                        found = true;
                        break;
                    }
                    // If we cannot find the downstream model in the new list (which shouldn't happen), reset the weak_ptr to prevent dangling pointers
                    if (!found) {
                        downstream.targetNode.reset();
                    }
                }
                for (auto& upstream : model->upstreams) {
                    auto old_upstream = upstream.targetNode.lock();
                    if (!old_upstream)  {
                        upstream.targetNode.reset(); // Reset to prevent dangling pointer
                        continue;
                    }

                    bool found = false;
                    for (auto& model2 : tk_pipelineModels) {
                        if (model2->name != old_upstream->name || model2->device != old_upstream->device) {
                            continue;
                        }
                        upstream.targetNode = model2;
                        found = true;
                        break;
                    }
                    if (!found) {
                        upstream.targetNode.reset();
                    }
                }
            }

            // Deep-copy tk_subTasks to map to the NEW containers, preventing dangling pointers
            tk_subTasks.clear();
            // for (const auto& kv : other.tk_subTasks) {
            //     std::vector<std::weak_ptr<ContainerHandle>> new_container_list;
            //     for (const auto& old_cont_weak : kv.second) {
            //         auto old_cont = old_cont_weak.lock();
            //         if (!old_cont) continue;

            //         for (auto& pm : tk_pipelineModels) {
            //             for (auto& new_cont : pm->manifestations) {
            //                 if (new_cont->name == old_cont->name) {
            //                     new_container_list.push_back(new_cont);
            //                 }
            //             }
            //         }
            //     }
            //     tk_subTasks[kv.first] = new_container_list;
            // }

            tk_newlyAdded = other.tk_newlyAdded;
        }
        return *this;
    }

    /**
     * @brief Copy the references to the containers in tk_subTasks from another TaskHandle. 
     * This is used when we want to keep the same container references to keep track of the same running instances.
     * 
     * @param other 
     */
    void copySubTasksFrom(const TaskHandle& other) {
        // FIX 1: Prevent self-assignment deadlocks
        if (this == &other) {
            return;
        }

        std::lock(tk_mutex, other.tk_mutex);
        std::lock_guard<std::mutex> lock1(tk_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.tk_mutex, std::adopt_lock);
        
        tk_subTasks.clear();

        for (auto& pModel : tk_pipelineModels) {
            {
                std::lock_guard<std::mutex> pmLock(pModel->pipelineModelMutex);
                pModel->manifestations.clear();
            }

            for (auto& otherModel : other.tk_pipelineModels) {
                if (pModel->name == otherModel->name && pModel->device == otherModel->device) {
                    std::lock(pModel->pipelineModelMutex, otherModel->pipelineModelMutex);
                    std::lock_guard<std::mutex> pmLock1(pModel->pipelineModelMutex, std::adopt_lock);
                    std::lock_guard<std::mutex> pmLock2(otherModel->pipelineModelMutex, std::adopt_lock);

                    for (auto& cont : otherModel->manifestations) {
                        tk_subTasks[pModel->name].push_back(cont);
                        pModel->manifestations.push_back(cont);
                    }
                    break;
                }
            }
        }
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
        std::string stream;
        std::string srcDevice;
        std::string edgeNode;
        bool added = false;
    };

    void from_json(const nlohmann::json &j, TaskStruct &val);
}

/**
 * @brief This is the main data structure that the Controller uses to track the device list in the system.
 * 
 */
struct Devices {
public:
    void addDevice(const std::string &name, std::shared_ptr<NodeHandle> node) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        list[name] = node;
    }

    void removeDevice(const std::string &name) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        list.erase(name);
    }

    std::shared_ptr<NodeHandle> getDevice(const std::string &name) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        auto it = list.find(name);
        return (it != list.end()) ? it->second : nullptr;
    }

    std::vector<std::shared_ptr<NodeHandle>> getList() {
        std::lock_guard<std::mutex> lock(devicesMutex);
        std::vector<std::shared_ptr<NodeHandle>> elements;
        for (auto &d: list) {
            elements.push_back(d.second);
        }
        return elements;
    }

    std::map<std::string, std::shared_ptr<NodeHandle>> getMap() {
        std::lock_guard<std::mutex> lock(devicesMutex);
        return list;
    }

    bool hasDevice(const std::string &name) {
        std::lock_guard<std::mutex> lock(devicesMutex);
        return list.find(name) != list.end();
    }
private:
    std::map<std::string, std::shared_ptr<NodeHandle>> list = {};
    std::mutex devicesMutex;
};

/**
 * @brief This is the main data structure that the Controller uses to track the task lists in the system.
 * 
 */
struct Tasks {
public:
    void addTask(const std::string &name, std::shared_ptr<TaskHandle> task) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        list[name] = task;
    }

    void removeTask(const std::string &name) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        list.erase(name);
    }

    std::shared_ptr<TaskHandle> getTask(const std::string &name) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        auto it = list.find(name);
        return (it != list.end()) ? it->second : nullptr;
    }

    std::vector<std::shared_ptr<TaskHandle>> getList() {
        std::lock_guard<std::mutex> lock(tasksMutex);
        std::vector<std::shared_ptr<TaskHandle>> tasks;
        for (auto &t: list) {
            tasks.push_back(t.second);
        }
        return tasks;
    }

    std::map<std::string, std::shared_ptr<TaskHandle>> getMap() {
        std::lock_guard<std::mutex> lock(tasksMutex);
        return list;
    }

    bool hasTask(const std::string &name) {
        std::lock_guard<std::mutex> lock(tasksMutex);
        return list.find(name) != list.end();
    }

    bool hasTasks() {
        std::lock_guard<std::mutex> lock(tasksMutex);
        return !list.empty();
    }

    Tasks() = default;

    // Copy constructor (Deep Copy via make_shared)
    Tasks(const Tasks &other) {
        std::lock(tasksMutex, other.tasksMutex);
        std::lock_guard<std::mutex> lock1(tasksMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.tasksMutex, std::adopt_lock);
        list = {};
        for (auto &t: other.list) {
            // Copy the task itself
            auto newTask = std::make_shared<TaskHandle>(*t.second);
            
            // Adding the task pointer to the new PipelineModels and their manifestations to maintain the graph structure.
            for (auto& model : newTask->tk_pipelineModels) {
                model->task = newTask; // implicitly casts to weak_ptr
            }
            
            list[t.first] = newTask;
        }
    }

    Tasks& operator=(const Tasks &other) {
        if (this != &other) {
            std::lock(tasksMutex, other.tasksMutex);
            std::lock_guard<std::mutex> lock1(tasksMutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.tasksMutex, std::adopt_lock);
            
            // list.clear() safely drops the reference counts of the old shared_ptrs, preventing memory leaks!
            list.clear();
            
            for (auto &t: other.list) {
                // Copy the task itself
                auto newTask = std::make_shared<TaskHandle>(*t.second);
                
                // Adding the task pointer to the new PipelineModels and their manifestations to maintain the graph structure.
                for (auto& model : newTask->tk_pipelineModels) {
                    model->task = newTask; // implicitly casts to weak_ptr
                }
                
                list[t.first] = newTask;
            }
        }
        return *this;
    }

    /**
     * @brief Basically a copy constructor but this also copies the references to the containers thus it takes a snapshot of the
     * current state of the system with the same running instances. This is used when we want to keep track of the same running
     * instances in the scheduling cycle.
     * 
     * @param source 
     */
    void createSnapshotFrom(const Tasks& source) {
        if (this == &source) {
            return;
        }

        std::lock(tasksMutex, source.tasksMutex);
        std::lock_guard<std::mutex> lock1(tasksMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(source.tasksMutex, std::adopt_lock);
        
        list.clear();
        
        for (const auto& t : source.list) {
            auto newTask = std::make_shared<TaskHandle>(*t.second);
            
            for (auto& model : newTask->tk_pipelineModels) {
                model->task = newTask; 
            }
            
            newTask->copySubTasksFrom(*t.second);
            
            list[t.first] = newTask;
        }
    }

private:
    std::map<std::string, std::shared_ptr<TaskHandle>> list = {};
    mutable std::mutex tasksMutex;
};

/**
 * @brief This is the main data structure that the Controller uses to track the containers running in the system
 * 
 */
struct Containers {
public:
    void addContainer(const std::string &name, std::shared_ptr<ContainerHandle> container) {
        std::lock_guard<std::mutex> lock(containersMutex);
        list[name] = container;
    }

    void removeContainer(const std::string &name) {
        std::lock_guard<std::mutex> lock(containersMutex);
        list.erase(name);
    }

    std::shared_ptr<ContainerHandle> getContainer(const std::string &name) {
        std::lock_guard<std::mutex> lock(containersMutex);
        auto it = list.find(name);
        return (it != list.end()) ? it->second : nullptr;
    }

    std::vector<std::shared_ptr<ContainerHandle>> getList() {
        std::lock_guard<std::mutex> lock(containersMutex);
        std::vector<std::shared_ptr<ContainerHandle>> elements;
        for (auto &c: list) {
            elements.push_back(c.second);
        }
        return elements;
    }

    std::vector<std::tuple<std::string, std::string, nlohmann::json>> getFLConnections() {
        std::lock_guard<std::mutex> lock(containersMutex);
        std::vector<std::tuple<std::string, std::string, nlohmann::json>> elements;
        for (auto &c: list) {
            // Safely lock the weak_ptr before accessing the device agent name
            auto agent = c.second->device_agent.lock();
            std::string agentName = agent ? agent->name : "unassigned";
            
            elements.push_back(std::make_tuple(c.first, agentName, c.second->fcpo_conf));
        }
        return elements;
    }

    std::map<std::string, std::shared_ptr<ContainerHandle>> getMap() {
        std::lock_guard<std::mutex> lock(containersMutex);
        return list;
    }

    bool hasContainer(const std::string &name) {
        std::lock_guard<std::mutex> lock(containersMutex);
        return list.find(name) != list.end();
    }

private:
    std::map<std::string, std::shared_ptr<ContainerHandle>> list = {};
    std::mutex containersMutex;
};

class Controller {
public:
    Controller(int argc, char **argv);
    ~Controller();

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
        remainTasks.erase(std::remove_if(remainTasks.begin(), remainTasks.end(),
        [this](TaskDescription::TaskStruct &t) {
            if (!t.added) { t.added = this->AddTask(t); }
            return t.added; // Remove it if it was successfully added
        }), remainTasks.end());
    }

    void AddDevice(const std::string name);
    bool AddTask(const TaskDescription::TaskStruct &task);
    void AddRemainTask(const TaskDescription::TaskStruct &task) {remainTasks.push_back(task);}

    void Scheduling();
    void HandleControlMessages();
    [[nodiscard]] bool isRunning() const { return running; };
    void Stop() { running = false; };
private:

    /////////////////////////////////////////// PRIVATE STRUCTURES ///////////////////////////////////////////

    struct TimingControl {
        uint64_t schedulingIntervalSec;
        uint64_t rescalingIntervalSec;
        uint64_t networkCheckIntervalSec;
        uint64_t scaleUpIntervalThresholdSec;
        uint64_t scaleDownIntervalThresholdSec;

        std::map<std::string, std::chrono::system_clock::time_point> nextSchedulingtime = std::map<std::string, std::chrono::system_clock::time_point>();
        std::map<std::string, std::chrono::system_clock::time_point> nextRescalingTime = std::map<std::string, std::chrono::system_clock::time_point>();
        // ClockType nextSchedulingTime = std::chrono::system_clock::time_point::min();
        ClockType currSchedulingTime = std::chrono::system_clock::time_point::min();
        // ClockType nextRescalingTime = std::chrono::system_clock::time_point::max();
    };

    /////////////////////////////////////////// PRIVATE FUNCTIONS ///////////////////////////////////////////

    // CONFIGS
    void readConfigFile(const std::string &config_path);
    void readInitialObjectCount(const std::string& path);
    PipelineModelListType getModelsByPipelineType(PipelineType type, const std::string &startDevice,
                                                  const std::string &pipelineName = "", const std::string &streamName = "", const std::string &edgeNode = "server");
    std::vector<std::string> getPipelineNames();
    
    std::shared_ptr<TaskHandle> CreatePipelineFromMessage(TaskDesc msg);
    void UpdatePipelineFromMessage(std::shared_ptr<TaskHandle> task, TaskDesc msg);

    // STARTUP
    void ApplyScheduling();
    std::shared_ptr<ContainerHandle> TranslateToContainer(std::shared_ptr<PipelineModel> model, std::shared_ptr<NodeHandle> device, unsigned int i);

    // CWD
    void crossDeviceWorkloadDistributor(TaskHandle *task, uint64_t slo);
    void shiftModelToEdge(PipelineModelListType &pipeline, PipelineModel *currModel, uint64_t slo, const std::string& edgeDevice);

    // PIPELINE SCALING
    void Rescaling();
    void ScaleUp(PipelineModel *model, uint8_t numIncReps);
    void ScaleDown(PipelineModel *model, uint8_t numDecReps);
    uint8_t incNumReplicas(const PipelineModel *model);
    uint8_t decNumReplicas(const PipelineModel *model);

    // GPU HANDLING
    void basicGPUScheduling(std::vector<std::shared_ptr<ContainerHandle>> new_containers);
    void initialiseGPU(NodeHandle *node, int numGPUs, std::vector<int> memLimits);
    void initiateGPULanes(NodeHandle &node);
    void insertFreeGPUPortion(GPUPortionList &portionList, GPUPortion *freePortion);
    bool removeFreeGPUPortion(GPUPortionList &portionList, GPUPortion *freePortion);
    std::pair<GPUPortion *, GPUPortion *> insertUsedGPUPortion(GPUPortionList &portionList,  std::shared_ptr<ContainerHandle> container, GPUPortion *toBeDividedFreePortion);
    bool reclaimGPUPortion(GPUPortion *toBeReclaimedPortion);
    GPUPortion* findFreePortionForInsertion(GPUPortionList &portionList, ContainerHandle *container);

    // CORAL
    bool containerColocationTemporalScheduling(std::shared_ptr<ContainerHandle> container);
    bool modelColocationTemporalScheduling(PipelineModel *pipelineModel, int replica_id);
    void colocationTemporalScheduling();

    // CONTAINER MANAGEMENT
    void StartContainer(std::shared_ptr<ContainerHandle> container, bool easy_allocation = true);
    void MoveContainer(std::shared_ptr<ContainerHandle> container, 
                       NodeHandle *new_device);
    void StopContainer(std::shared_ptr<ContainerHandle> container, NodeHandle *device, bool forced = false);
    void AdjustContainerDownstreams(std::shared_ptr<ContainerHandle> cont, std::shared_ptr<ContainerHandle> upstr, NodeHandle *new_device,
                               AdjustMode mode, const std::string &old_link = "");
    void AdjustContainerDownstreamsInBatch(const std::shared_ptr<ContainerHandle> container,
                                    const SingleContainerDownstreamAdjustmentList &downstreamAdjustmentList);
    void SyncDatasource(std::shared_ptr<ContainerHandle> prev, std::shared_ptr<ContainerHandle> curr);
    void AdjustBatchSize(std::shared_ptr<ContainerHandle> msvc, int new_bs);
    void AdjustCudaDevice(std::shared_ptr<ContainerHandle> msvc, GPUHandle *new_device);
    void AdjustResolution(std::shared_ptr<ContainerHandle> msvc, std::vector<int> new_resolution);
    void AdjustTiming(std::shared_ptr<ContainerHandle> container);

    // PROFILING DATA & MONITORING
    void queryingProfiles(TaskHandle *task);
    void queryInDeviceNetworkEntries(std::shared_ptr<NodeHandle> node);
    void calculateQueueSizes(std::shared_ptr<ContainerHandle> container, const ModelType modelType);
    uint64_t calculateQueuingLatency(float &arrival_rate, const float &preprocess_rate);
    NetworkEntryType initNetworkCheck(NodeHandle &node, uint32_t minPacketSize = 1000, uint32_t maxPacketSize = 1228800, uint32_t numLoops = 20);
    void checkNetworkConditions();

    // PERFORMANCE ESTIMATION
    void estimateModelLatency(PipelineModel *currModel);
    void estimateModelNetworkLatency(PipelineModel *currModel);
    void estimatePipelineLatency(PipelineModel *currModel, const uint64_t start2HereLatency);
    void estimateModelTiming(PipelineModel *currModel, const uint64_t start2HereDutyCycle);
    void estimatePipelineTiming();
    void estimatePipelineTiming(TaskHandle *task);
    void estimateTimeBudgetLeft(PipelineModel *currModel);

    // PIPELINE MERGING
    bool mergeArrivalProfiles(ModelArrivalProfile &mergedProfile, const ModelArrivalProfile &toBeMergedProfile, const std::string &device, const std::string &upstreamDevice);
    bool mergeProcessProfiles(
            PerDeviceModelProfileType &mergedProfile,
            float arrivalRate1,
            const PerDeviceModelProfileType &toBeMergedProfile,
            float arrivalRate2,
            const std::string &device);
    bool mergeModels(PipelineModel *mergedModel, PipelineModel *tobeMergedModel, const std::string &device);
    
    std::shared_ptr<TaskHandle> mergePipelines(const std::string& taskName);
    void mergePipelines();

    // CONTROL MESSAGING
    void handleDeviseAdvertisement(const std::string &msg);
    void handleDummyDataRequest(const std::string &msg);
    void handleForwardFLRequest(const std::string &msg);
    void handleSinkMetrics(const std::string &msg);
    void sendMessageToDevice(const std::string &topik, const std::string &type, const std::string &content);

    // API CALLS
    void ScheduleSingleTask(std::shared_ptr<TaskHandle> task);
    void HandleStartTask(const std::string &msg);
    void StopSingleTask(const std::string &msg);

    /////////////////////////////////////////// PRIVATE VARIABLES ///////////////////////////////////////////

    // RUNTIME VARIABLES
    bool running;
    std::atomic<bool> isPipelineInitialised = false;
    ClockType startTime = std::chrono::system_clock::time_point();
    std::uint64_t ctrl_schedulingIntervalSec;
    ClockType ctrl_nextSchedulingTime = std::chrono::system_clock::now();
    ClockType ctrl_currSchedulingTime = std::chrono::system_clock::now();
    Tasks ctrl_unscheduledPipelines, ctrl_savedUnscheduledPipelines, ctrl_scheduledPipelines,
          ctrl_pastScheduledPipelines, ctrl_mergedPipelines;
    Devices devices;
    Containers containers;
    std::vector<spdlog::sink_ptr> ctrl_loggerSinks = {};
    std::shared_ptr<spdlog::logger> ctrl_logger;
    uint16_t ctrl_numGPULanes = NUM_LANES_PER_GPU * NUM_GPUS, ctrl_numGPUPortions;

    // EXPERIMENT CONFIG
    std::string ctrl_experimentName;
    std::string ctrl_systemName;
    json ctrl_clusterInfo;
    std::vector<TaskDescription::TaskStruct> initialTasks;
    std::vector<TaskDescription::TaskStruct> remainTasks;
    uint16_t ctrl_systemFPS;
    uint16_t ctrl_runtime;
    uint16_t ctrl_clusterID, ctrl_clusterCount;
    uint16_t ctrl_port_offset;
    std::map<std::string, BatchSizeType> ctrl_initialBatchSizes;
    TimingControl ctrl_controlTimings;
    ContainerLibType ctrl_containerLib;

    // LOGGING
    std::string ctrl_logPath;
    uint16_t ctrl_loggingMode;
    uint16_t ctrl_verbose;

    // MESSAGING & NETWORK
    context_t api_ctx, system_ctx;
    socket_t api_socket, server_socket, message_queue;
    std::unordered_map<std::string, std::function<void(const std::string&)>> api_handlers, system_handlers;

    // PROFILING DATA & METRICS
    std::unique_ptr<pqxx::connection> ctrl_metricsServerConn = nullptr;
    MetricsServerConfigs ctrl_metricsServerConfigs;
    std::map<std::string, NetworkEntryType> network_check_buffer;
    std::map<std::string, NetworkEntryType> ctrl_inDeviceNetworkEntries;
    std::map<std::string, std::map<std::string, float>> ctrl_initialRequestRates;

    // FCPO
    std::unique_ptr<FCPOServer> ctrl_fcpo_server;
    json ctrl_fcpo_config;

    BandwidthPredictor ctrl_bandwidth_predictor;
};


#endif //PIPEPLUSPLUS_CONTROLLER_H