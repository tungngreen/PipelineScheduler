#include <chrono>
#include <queue>
#include <deque>
#include <list>
#include <opencv4/opencv2/opencv.hpp>
#include <mutex>
#include <misc.h>
#include <condition_variable>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include "../utils/json.h"
#include <vector>

#include <chrono>
#include <ctime>

#ifndef MICROSERVICE_H
#define MICROSERVICE_H

using json = nlohmann::ordered_json;


template<typename DataType>
struct RequestData {
    RequestDataShapeType shape;
    DataType data;

    RequestData(RequestDataShapeType s, DataType d) : data(d) {
        shape = s;
    }

    RequestData() {}

    ~RequestData() {
        data.release();
        shape.clear();
    }
};

/**
 * @brief 
 * 
 * @tparam RequestData
 */
template<typename DataType>
struct Request {
    // The moment this request was generated at the begining of the pipeline.
    BatchTimeType req_origGenTime;
    // The end-to-end service level latency objective to which this request is subject
    RequestSLOType req_e2eSLOLatency;
    /**
     * @brief The path that this request and its ancestors have travelled through.
     * Template `[hostDeviceName|microserviceID|inReqNumber|totalNumberOfOutputs|NumberInOutputs|outPackageSize (in byte)]`
     * For instance, for a request from container with id of `YOLOv5_01` to container with id of `retinaface_02`
     * we may have a path that looks like this `[edge|YOLOv5_01|5|10|8|212072][server|retinaface_02|09|2|2|2343]`
     */

    RequestPathType req_travelPath;

    // Batch size
    BatchSizeType req_batchSize = 0;

    // The Inter-container GPU data of that this request carries.
    std::vector<RequestData<DataType>> req_data = {};
    // To carry the data of the upstream microservice in case we need them for further processing.
    // For instance, for cropping we need both the original image (`upstreamReq_data`) and the output
    // of the inference engine, which is a result of `req_data`.
    // If there is nothing to carry, it is a blank vector.
    std::vector<RequestData<DataType>> upstreamReq_data = {};

    Request() {};

    Request(
            BatchTimeType genTime,
            RequestSLOType latency,
            RequestPathType path,
            BatchSizeType batchSize,
            std::vector<RequestData<DataType>> data,
            std::vector<RequestData<DataType>> upstream_data

    ) : req_origGenTime(genTime),
        req_e2eSLOLatency(latency),
        req_travelPath(std::move(path)),
        req_batchSize(batchSize) {
        req_data = data;
        upstreamReq_data = upstream_data;
    }

    // df
    Request(
            BatchTimeType genTime,
            RequestSLOType latency,
            RequestPathType path,
            BatchSizeType batchSize,
            std::vector<RequestData<DataType>> data
    ) : req_origGenTime(genTime),
        req_e2eSLOLatency(latency),
        req_travelPath(std::move(path)),
        req_batchSize(batchSize) {
        req_data = data;
    }

    /**
     * @brief making our request 
     * 
     * @param other 
     * @return Request& 
     */
    Request &operator=(const Request &other) {
        if (this != &other) {
            req_origGenTime = other.req_origGenTime;
            req_e2eSLOLatency = other.req_e2eSLOLatency;
            req_travelPath = other.req_travelPath;
            req_batchSize = other.req_batchSize;
            req_data = other.req_data;
            upstreamReq_data = other.upstreamReq_data;
        }
        return *this;
    }

    ~Request() {
        req_data.clear();
        upstreamReq_data.clear();
    }
};

//template<int MaxSize=100>
class ThreadSafeFixSizedDoubleQueue {
private:
    std::string q_name;
    std::queue<Request<LocalCPUReqDataType>> q_cpuQueue;
    std::queue<Request<LocalGPUReqDataType>> q_gpuQueue;
    std::mutex q_mutex;
    std::condition_variable q_condition_producers;
    std::condition_variable q_condition_consumer;
    std::uint8_t activeQueueIndex;
    QueueLengthType q_MaxSize = 100;
    std::int16_t class_of_interest;
    bool isEmpty;
    bool isEncoded = false;
    std::atomic<unsigned int> dropedCount = 0;

    // Fairness control variables
    std::atomic<uint32_t> producer_ticket{0}; // Ticket number for producers
    std::atomic<uint32_t> current_ticket{0};  // Current ticket being served
    std::atomic<bool> consumer_waiting{false}; // To give consumer priority

    static constexpr uint32_t MAX_TICKET_VALUE = UINT32_MAX - 1000; // Maximum ticket value before wraparound

public:
    ThreadSafeFixSizedDoubleQueue(QueueLengthType size, int16_t coi, std::string name) :  q_name(name), q_MaxSize(size), class_of_interest(coi) {}

    ~ThreadSafeFixSizedDoubleQueue() {
        std::queue<Request<LocalGPUReqDataType>>().swap(q_gpuQueue);
        std::queue<Request<LocalCPUReqDataType>>().swap(q_cpuQueue);
    }

    std::string getName() const {
        return q_name;
    }

    /**
     * @brief Emplacing Type 1 requests with fairness for producers and ticket wraparound
     * 
     * @param request 
     */
    void emplace(Request<LocalCPUReqDataType> request) {
        // Get a ticket for this producer and handle overflow if necessary
        uint32_t my_ticket = producer_ticket.fetch_add(1);
        handleTicketOverflow();

        // Wait until it's this producer's turn and the consumer is not waiting
        while (current_ticket.load() != my_ticket || consumer_waiting.load()) {
            std::this_thread::yield();
        }

        // Enter critical section only for queue operations
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            if (q_cpuQueue.size() == q_MaxSize) {
                dropedCount += q_cpuQueue.front().req_batchSize;
                spdlog::get("container_agent")->warn("Queue {0:s} is full, dropping request", q_name);
                q_cpuQueue.pop();
            }
            q_cpuQueue.emplace(std::move(request)); // Use move semantics for efficiency
        } // Mutex is released here

        // Move to the next producer and notify consumer, outside the critical section
        current_ticket.fetch_add(1);
        q_condition_consumer.notify_one();
    }

    /**
     * @brief Emplacing Type 2 requests with fairness for producers and ticket wraparound
     * 
     * @param request 
     */
    void emplace(Request<LocalGPUReqDataType> request) {
        uint32_t my_ticket = producer_ticket.fetch_add(1); // Get a ticket for this producer
        handleTicketOverflow(); // Handle overflow if needed

        while (current_ticket.load() != my_ticket) {
            std::this_thread::yield(); // Wait for the producer's turn or yield to the consumer
        }

        // Lock only to modify the queue safely
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            if (q_gpuQueue.size() == q_MaxSize) {
                dropedCount += q_gpuQueue.front().req_batchSize;
                spdlog::get("container_agent")->warn("Queue {0:s} is full, dropping request", q_name);
                q_gpuQueue.pop();
            }
            q_gpuQueue.emplace(std::move(request)); // Use move semantics if possible for efficiency
        } // Unlock as soon as queue operations are done

        // Move to the next producer and notify consumer, outside the critical section
        current_ticket.fetch_add(1);
        q_condition_consumer.notify_one();
    }

    /**
     * @brief popping Type 1 requests with priority for the consumer
     * 
     * @param request 
     */
    Request<LocalCPUReqDataType> pop1(uint32_t timeout = 100000) { // 100ms
        Request<LocalCPUReqDataType> request;
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            isEmpty = !q_condition_consumer.wait_for(
                lock,
                TimePrecisionType(timeout),
                [this]() { return !q_cpuQueue.empty(); }
            );

            // Only proceed if the queue is not empty
            if (!isEmpty) {
                request = q_cpuQueue.front();
                q_cpuQueue.pop();
            }
        }

        // If the queue was empty, set a default value outside the critical section
        if (isEmpty) {
            request.req_travelPath = {"empty"};
        }

    return request;
}

    /**
     * @brief popping Type 2 requests with priority for the consumer
     * 
     * @param request 
     */
    Request<LocalGPUReqDataType> pop2(uint32_t timeout = 100000) { // 100ms
        Request<LocalGPUReqDataType> request;
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            isEmpty = !q_condition_consumer.wait_for(
                    lock,
                    TimePrecisionType(timeout),
                    [this]() { return !q_gpuQueue.empty(); }
            );

            if (!isEmpty) {
                request = q_gpuQueue.front();
                q_gpuQueue.pop();
            }
        }
        if (isEmpty) {
            request.req_travelPath = {"empty"};
        }
        return request;
    }

    void setQueueSize(uint32_t queueSize) {
        std::unique_lock<std::mutex> lock(q_mutex);
        q_MaxSize = queueSize;
    }

    int32_t size() {
        std::unique_lock<std::mutex> lock(q_mutex);
        if (activeQueueIndex == 1) {
            return q_cpuQueue.size();
        }
        return q_gpuQueue.size();
    }

    int32_t size(uint8_t queueIndex) {
        std::unique_lock<std::mutex> lock(q_mutex);
        if (queueIndex == 1) {
            return q_cpuQueue.size();
        }
        return q_gpuQueue.size();
    }

    unsigned int drops() {
        return dropedCount.exchange(0);
    }

    void setActiveQueueIndex(uint8_t index) {
        std::unique_lock<std::mutex> lock(q_mutex);
        activeQueueIndex = index;
    }

    uint8_t getActiveQueueIndex() {
        std::unique_lock<std::mutex> lock(q_mutex);
        return activeQueueIndex;
    }

    void setClassOfInterest(int16_t classOfInterest) {
        std::unique_lock<std::mutex> lock(q_mutex);
        class_of_interest = classOfInterest;
    }

    int16_t getClassOfInterest() {
        std::unique_lock<std::mutex> lock(q_mutex);
        return class_of_interest;
    }

    void setEncoded(bool isEncoded) {
        std::unique_lock<std::mutex> lock(q_mutex);
        this->isEncoded = isEncoded;
    }

    bool getEncoded() {
        std::unique_lock<std::mutex> lock(q_mutex);
        return isEncoded;
    }

private:
    /**
     * @brief Handle ticket overflow by resetting tickets when they reach a high value.
     */
    void handleTicketOverflow() {
        std::unique_lock<std::mutex> lock(q_mutex); // Ensure no concurrent access to tickets
        if (producer_ticket.load() >= MAX_TICKET_VALUE && current_ticket.load() >= MAX_TICKET_VALUE) {
            // Reset both producer_ticket and current_ticket
            producer_ticket.store(0);
            current_ticket.store(0);
        }
    }
};


/**
 * @brief 
 * 
 */
enum class CommMethod {
    sharedMemory = 0,
    GpuAddress = 1,
    serialized = 2,
    localGPU = 3,
    localCPU = 4,
    encodedCPU = 5
};

enum class NeighborType {
    Upstream,
    Downstream,
};

enum RUNMODE {
    DEPLOYMENT,
    PROFILING,
    EMPTY_PROFILING
};

enum class AllocationMode {
    //Conservative mode: the microservice will allocate only the amount of memory calculated based on the ideal batch size
    Conservative, 
    //Aggressive mode: the microservice will allocate the maximum amount of memory possible
    Aggressive
};

namespace msvcconfigs {
    /**
     * @brief Descriptions of up and downstream microservices neighboring this current microservice.
     *
     *
     */
    struct NeighborMicroserviceConfigs {
        // Name of the up/downstream microservice
        std::string name;
        // The communication method for the microservice to
        CommMethod commMethod;
        //
        std::vector<std::string> link;
        //
        QueueLengthType maxQueueSize;
        // For a Downstream Microservice, this is the data class (defined by the current microservice's model) to be sent this neighbor.
        // For instance, if the model is trained on coco and this neighbor microservice expects coco human, then the value is `0`.
        // Value `-1` denotes all classes.
        // Value `-2` denotes Upstream Microservice.
        int16_t classOfInterest;
        // The shape of data this neighbor microservice expects from the current microservice.
        std::vector<RequestDataShapeType> expectedShape;
    };

    /**
     * @brief Have to match the the type here and the type in the json config file
     *
     */
    enum class MicroserviceType {
        // Receiver should have number smaller than 500
        Receiver = 0,
        // DataProcessor should have number between 500 and 1000
        DataReader = 500,
        ProfileGenerator = 501,
        DataSink = 502,
        // Preprocessor should have number between 1000 and 2000
        Preprocessor = 1000,
        // Batcher should have number between 1500 and 2000
        Batcher = 1500,
        // Inferencer should have number larger than 2000
        TRTInferencer = 2000,
        // Postprocessor should have number larger than 3000
        Postprocessor = 3000,
        PostprocessorBBoxCropper = 3001,
        PostProcessorClassifer = 3002,
        PostProcessorSMClassifier = 3003,
        PostProcessorBBoxCropperVerifier = 3004,
        PostProcessorKPointExtractor = 3005,
        // Sender should have number larger than 4000
        Sender = 4000,
        LocalCPUSender = 4001,
        SerializedCPUSender = 4002,
        GPUSender = 4003
    };

    /**
     * @brief
     *
     */
    struct BaseMicroserviceConfigs {
        // Name of the container
        std::string msvc_contName;
        // Name of the microservice
        std::string msvc_name;
        // Type of microservice data receiver, data processor, or data sender
        MicroserviceType msvc_type;
        // The acceptable latency for each individual request processed by this microservice, in `microsecond`
        MsvcSLOType msvc_pipelineSLO;
        // 
        QueueLengthType msvc_maxQueueSize;
        // Ideal batch size for this microservice, runtime batch size could be smaller though
        BatchSizeType msvc_idealBatchSize;
        // Shape of data produced by this microservice
        std::vector<RequestDataShapeType> msvc_dataShape;
        // GPU index, -1 means CPU
        int8_t msvc_deviceIndex = 0;
        // Log dir
        std::string msvc_containerLogPath;
        // Run mode
        RUNMODE msvc_RUNMODE = RUNMODE::DEPLOYMENT;
        // List of upstream microservices
        std::list<NeighborMicroserviceConfigs> msvc_upstreamMicroservices;
        std::list<NeighborMicroserviceConfigs> msvc_dnstreamMicroservices;
    };


    /**
     * @brief 
     * 
     */
    struct NeighborMicroservice : NeighborMicroserviceConfigs {
        NumQueuesType queueNum;

        NeighborMicroservice(const NeighborMicroserviceConfigs &configs, NumQueuesType queueNum)
                : NeighborMicroserviceConfigs(configs),
                  queueNum(queueNum) {}
    };


    void from_json(const json &j, NeighborMicroserviceConfigs &val);

    void from_json(const json &j, BaseMicroserviceConfigs &val);

    void to_json(json &j, const NeighborMicroserviceConfigs &val);

    void to_json(json &j, const BaseMicroserviceConfigs &val);
}

using msvcconfigs::NeighborMicroserviceConfigs;
using msvcconfigs::BaseMicroserviceConfigs;
using msvcconfigs::MicroserviceType;

class ArrivalReqRecords {
public:
    ArrivalReqRecords(uint64_t keepLength = 60000) {
        this->keepLength = std::chrono::milliseconds(keepLength);
    }
    ~ArrivalReqRecords() = default;


    /**
     * @brief Add a new arrival to the records. There are 3 timestamps to keep be kept.
     * 1. The time the request is processed by the upstream postprocessor and placed onto the outqueue. (SECOND_TIMESTAMP)
     * 2. The time the request is sent out by upstream sender. (THIRD_TIMESTAMP)
     * 3. The time the request is placed onto the outqueue of receiver. (FOURTH_TIMESTAMP)
     * 4. The time the request is received by the preprocessor. (FIFTH_TIMESTAMP)
     *
     * @param timestamps
     */
    void addRecord(
        RequestTimeType timestamps,
        uint32_t rpcBatchSize,
        uint32_t totalPkgSize,
        uint32_t requestSize,
        uint32_t reqNumber,
        std::string reqOriginStream,
        std::string originDevice
    ) {
        for (size_t i = 0; i < timestamps.size() - 1; ++i) {
            if (timestamps[i] > timestamps[i + 1]) return;
        }


        auto outQueueingDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[2] - timestamps[1]).count();
        auto transferDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[3] - timestamps[2]).count();
        auto inQueueingDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[4] - timestamps[3]).count();
    
        lastTransferDuration = transferDuration;

        std::unique_lock<std::mutex> lock(mutex);
        ArrivalRecord * record = &records[{reqOriginStream, originDevice}];
        // If transfer latency is 0 or negative, which only happens when time between devices are not properly synchronized
        record->transferDuration.emplace_back(transferDuration);
        record->outQueueingDuration.emplace_back(outQueueingDuration);
        record->queueingDuration.emplace_back(inQueueingDuration);
        record->queueingDuration.emplace_back(inQueueingDuration);
        
        record->arrivalTime.emplace_back(timestamps[2]);
        record->totalPkgSize.emplace_back(totalPkgSize); //Byte
        record->reqSize.emplace_back(requestSize); //Byte
        currNumEntries++;
        totalNumEntries++;
    }


    ArrivalRecordType getRecords() {
        std::unique_lock<std::mutex> lock(mutex);
        ArrivalRecordType temp;
        for (auto &record: records) {
            temp[record.first] = record.second;
        }
        records.clear();
        currNumEntries = 0;
        return temp;
    }

    /**
     * @brief Get records into a combined record from multiple microservices
     * 
     * @param overallRecords 
     */
    void getRecords(ArrivalRecordType &overallRecords) {
        std::unique_lock<std::mutex> lock(mutex);
        // make deep copy of records

        for (auto &record: records) {
            if (overallRecords.find(record.first) == overallRecords.end()) {
                overallRecords[record.first] = record.second;
            } else {
                overallRecords[record.first].transferDuration.insert(
                    overallRecords[record.first].transferDuration.end(),
                    record.second.transferDuration.begin(),
                    record.second.transferDuration.end()
                );
                overallRecords[record.first].outQueueingDuration.insert(
                    overallRecords[record.first].outQueueingDuration.end(),
                    record.second.outQueueingDuration.begin(),
                    record.second.outQueueingDuration.end()
                );
                overallRecords[record.first].queueingDuration.insert(
                    overallRecords[record.first].queueingDuration.end(),
                    record.second.queueingDuration.begin(),
                    record.second.queueingDuration.end()
                );
                overallRecords[record.first].arrivalTime.insert(
                    overallRecords[record.first].arrivalTime.end(),
                    record.second.arrivalTime.begin(),
                    record.second.arrivalTime.end()
                );
                overallRecords[record.first].totalPkgSize.insert(
                    overallRecords[record.first].totalPkgSize.end(),
                    record.second.totalPkgSize.begin(),
                    record.second.totalPkgSize.end()
                );
                overallRecords[record.first].reqSize.insert(
                    overallRecords[record.first].reqSize.end(),
                    record.second.reqSize.begin(),
                    record.second.reqSize.end()
                );
            }
        }
        records.clear();
        currNumEntries = 0;
    }

    void setKeepLength(uint64_t keepLength) {
        std::unique_lock<std::mutex> lock(mutex);
        this->keepLength = std::chrono::milliseconds(keepLength);
    }

private:
    std::mutex mutex;
    int64_t lastTransferDuration = -1;
    ArrivalRecordType records;
    std::chrono::milliseconds keepLength;
    uint64_t totalNumEntries = 0, currNumEntries = 0;
};

class ProcessReqRecords {
public:
    ProcessReqRecords(uint64_t keepLength = 60000) {
        this->keepLength = std::chrono::milliseconds(keepLength);
    }
    ~ProcessReqRecords() = default;


    /**
     * @brief Add new process records to the records. There are 6 timestamps to keep be considered.
     * 1. When the request was received by the preprocessor (FIFTH_TIMESTAMP)
     * 2. When the request was done preprocessing by the preprocessor (SIXTH_TIMESTAMP)
     * 3. When the request, along with all others in the batch, was batched together and sent to the inferencer (SEVENTH_TIMESTAMP)
     * 4. When the batch was popped by the inferencer (EIGHTH_TIMESTAMP)
     * 5. When the batch inferencer was completed by the inferencer (NINTH_TIMESTAMP)
     * 6. When the batch was received by the postprocessor (TENTH_TIMESTAMP)
     * 7. When each request starts to be processed by the postprocessor (ELEVENTH_TIMESTAMP)
     * 8. When each request was completed by the postprocessor (TWELFTH_TIMESTAMP)
     *
     * @param timestamps
     */
    void addRecord(
        RequestTimeType timestamps,
        BatchSizeType inferBatchSize,
        uint32_t inputSize,
        uint32_t outputSize,
        uint32_t encodedOutputSize,
        uint32_t reqNumber,
        std::string reqOrigin = "stream"
    ) {
        for (size_t i = 0; i < timestamps.size() - 1; ++i) {
            if (timestamps[i] >= timestamps[i + 1]) return;
        }
        auto prepDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[6] - timestamps[5]).count();
        auto batchDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[7] - timestamps[6]).count();
        auto inferQueueingDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[8] - timestamps[7]).count();
        auto inferDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[9] - timestamps[8]).count();
        auto postDuration = std::chrono::duration_cast<TimePrecisionType>(timestamps[11] - timestamps[10]).count();

        std::unique_lock<std::mutex> lock(mutex);
        processRecords[{reqOrigin, inferBatchSize}].prepDuration.emplace_back(prepDuration);
        processRecords[{reqOrigin, inferBatchSize}].batchDuration.emplace_back(batchDuration);
        processRecords[{reqOrigin, inferBatchSize}].inferQueueingDuration.emplace_back(inferQueueingDuration);
        // We consider the time during which the batch inference results were unloaded from the inferencer to the postprocessor as the inference duration
        processRecords[{reqOrigin, inferBatchSize}].inferDuration.emplace_back(inferDuration);
        processRecords[{reqOrigin, inferBatchSize}].postDuration.emplace_back(postDuration);
        processRecords[{reqOrigin, inferBatchSize}].inferBatchSize.emplace_back(inferBatchSize);
        processRecords[{reqOrigin, inferBatchSize}].postEndTime.emplace_back(timestamps[11]);
        processRecords[{reqOrigin, inferBatchSize}].inputSize.emplace_back(inputSize);
        processRecords[{reqOrigin, inferBatchSize}].outputSize.emplace_back(outputSize);
        processRecords[{reqOrigin, inferBatchSize}].encodedOutputSize.emplace_back(encodedOutputSize);

        batchInferRecords[{reqOrigin, inferBatchSize}].inferDuration.emplace_back(inferDuration);

        currNumEntries++;
        totalNumEntries++;
    }

    ProcessRecordType getRecords() {
        std::unique_lock<std::mutex> lock(mutex);
        ProcessRecordType temp;
        for (auto &record: processRecords) {
            temp[record.first] = record.second;
        }

        processRecords.clear();
        currNumEntries = 0;
        return temp;
    }

    /**
     * @brief Get records into a combined record from multiple microservices
     * 
     * @param overallRecords 
     */
    void getRecords(ProcessRecordType &overallRecords) {
        std::unique_lock<std::mutex> lock(mutex);
        for (auto &record: processRecords) {
            if (overallRecords.find(record.first) == overallRecords.end()) {
                overallRecords[record.first] = record.second;
            } else {
                overallRecords[record.first].prepDuration.insert(
                    overallRecords[record.first].prepDuration.end(),
                    record.second.prepDuration.begin(),
                    record.second.prepDuration.end()
                );
                overallRecords[record.first].batchDuration.insert(
                    overallRecords[record.first].batchDuration.end(),
                    record.second.batchDuration.begin(),
                    record.second.batchDuration.end()
                );
                overallRecords[record.first].inferQueueingDuration.insert(
                    overallRecords[record.first].inferQueueingDuration.end(),
                    record.second.inferQueueingDuration.begin(),
                    record.second.inferQueueingDuration.end()
                );
                overallRecords[record.first].inferDuration.insert(
                    overallRecords[record.first].inferDuration.end(),
                    record.second.inferDuration.begin(),
                    record.second.inferDuration.end()
                );
                overallRecords[record.first].postDuration.insert(
                    overallRecords[record.first].postDuration.end(),
                    record.second.postDuration.begin(),
                    record.second.postDuration.end()
                );
                overallRecords[record.first].inferBatchSize.insert(
                    overallRecords[record.first].inferBatchSize.end(),
                    record.second.inferBatchSize.begin(),
                    record.second.inferBatchSize.end()
                );
                overallRecords[record.first].postEndTime.insert(
                    overallRecords[record.first].postEndTime.end(),
                    record.second.postEndTime.begin(),
                    record.second.postEndTime.end()
                );
                overallRecords[record.first].inputSize.insert(
                    overallRecords[record.first].inputSize.end(),
                    record.second.inputSize.begin(),
                    record.second.inputSize.end()
                );
                overallRecords[record.first].outputSize.insert(
                    overallRecords[record.first].outputSize.end(),
                    record.second.outputSize.begin(),
                    record.second.outputSize.end()
                );
                overallRecords[record.first].encodedOutputSize.insert(
                    overallRecords[record.first].encodedOutputSize.end(),
                    record.second.encodedOutputSize.begin(),
                    record.second.encodedOutputSize.end()
                );
            }
        }

        processRecords.clear();
        currNumEntries = 0;
    }

    BatchInferRecordType getBatchInferRecords() {
        std::unique_lock<std::mutex> lock(mutex);
        BatchInferRecordType temp;
        for (auto &record: batchInferRecords) {
            temp[record.first] = record.second;
        }

        batchInferRecords.clear();
        currNumEntries = 0;
        return temp;
    }

    void getBatchInferRecords(BatchInferRecordType &overallRecords) {
        std::unique_lock<std::mutex> lock(mutex);
        for (auto &record: batchInferRecords) {
            if (overallRecords.find(record.first) == overallRecords.end()) {
                overallRecords[record.first] = record.second;
            } else {
                overallRecords[record.first].inferDuration.insert(
                    overallRecords[record.first].inferDuration.end(),
                    record.second.inferDuration.begin(),
                    record.second.inferDuration.end()
                );
            }
        }

        batchInferRecords.clear();
        currNumEntries = 0;
    }

    void setKeepLength(uint64_t keepLength) {
        std::unique_lock<std::mutex> lock(mutex);
        this->keepLength = std::chrono::milliseconds(keepLength);
    }

private:
    std::mutex mutex;
    ProcessRecordType processRecords;
    BatchInferRecordType batchInferRecords;
    std::chrono::milliseconds keepLength;
    uint64_t totalNumEntries = 0, currNumEntries = 0;
};

enum class BATCH_MODE {
    FIXED,
    DYNAMIC, // Lazy batching
    OURS
};

enum class DROP_MODE {
    NO_DROP,
    LAZY
};

/**
 * @brief 
 * 
 */
class Microservice {
friend class ContainerAgent;
public:
    // Constructor that loads a struct args
    explicit Microservice(const json &jsonConfigs);

    explicit Microservice(const Microservice &other);

    virtual ~Microservice() {
        waitStop();
        spdlog::get("container_agent")->info("Microservice {} has stopped", msvc_name);
    }

    void waitStop () {
        uint8_t attempts = 0;
        const uint8_t maxAttempts = 100;

        while (!STOPPED) {
            if (attempts == 0) {
                spdlog::get("container_agent")->warn("Waiting for Microservice {} to stop...", msvc_name);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            if (++attempts > maxAttempts) {
                spdlog::get("container_agent")->error("Microservice {} failed to stop naturally after {} attempts", msvc_name, maxAttempts);
                break;
            }
        }
    }

    std::string msvc_experimentName;
    // Name Identifier assigned to the microservice in the format of `type_of_msvc-number`.
    // For instance, an object detector could be named `YOLOv5s-01`.
    // Another example is the
    std::string msvc_name;

    //
    std::string msvc_pipelineName;

    // Name of the contianer that holds this microservice
    std::string msvc_containerName;

    //
    std::string msvc_taskName;

    //
    std::string msvc_hostDevice;

    // Name of the system (e.g., ours, SOTA1, SOTA2, etc.)
    std::string msvc_systemName;

    void SetCurrFrameID(int id) {
        msvc_currFrameID = id;
    }

    void SetInQueue(std::vector<ThreadSafeFixSizedDoubleQueue *> queue) {
        msvc_InQueue = std::move(queue);
    };

    std::vector<ThreadSafeFixSizedDoubleQueue *> GetInQueue() {
        return msvc_InQueue;
    };

    std::vector<ThreadSafeFixSizedDoubleQueue *> GetOutQueue() {
        return msvc_OutQueue;
    };

    ThreadSafeFixSizedDoubleQueue *GetOutQueue(int queueIndex) {
        return msvc_OutQueue[queueIndex];
    };

    ThreadSafeFixSizedDoubleQueue *GetOutQueue(std::string queueName) {
        for (const auto &queue : msvc_OutQueue) {
            if (queue->getName() == queueName) {
                return queue;
            }
        }
        return nullptr;
    };

    RUNMODE getRUNMODE() {
        return msvc_RUNMODE;
    }

    virtual QueueLengthType GetOutQueueSize(int i) { return msvc_OutQueue[i]->size(); };

    unsigned int GetDroppedReqCount() {
        return msvc_droppedReqCount.exchange(0);
    };

    unsigned int GetTotalReqCount() {
        return msvc_totalReqCount.exchange(0);
    };

    unsigned int GetQueueDrops() {
        unsigned int val = 0;
        for (auto &queue : msvc_OutQueue) {
            val += queue->drops();
        }
        return val;
    };

    virtual PerSecondArrivalRecord getPerSecondArrivalRecord() {
        return {};
    }

    void stopThread() {
        STOP_THREADS = true;
    }

    void pauseThread() {
        PAUSE_THREADS = true;
        READY = false;
    }

    void unpauseThread() {
        PAUSE_THREADS = false;
    }

    bool checkReady() {
        return READY;
    }

    bool checkPause() {
        return PAUSE_THREADS;
    }

    void setRELOAD() {
        RELOADING = true;
    }

    void setReady() {
        READY = true;
    }

    /**
     * @brief Set the Device index
     * should be called at least once for each thread
     * 
     */
    void setDevice() {
        setDevice(msvc_deviceIndex);
    }

    /**
     * @brief Set the Device index
     * should be called at least once for each thread (except when the above function is already called)
     * 
     */
    void setDevice(int8_t deviceIndex) {
        if (deviceIndex >= 0) {
            int currentDevice = cv::cuda::getDevice();
            if (currentDevice != deviceIndex) {
                cv::cuda::resetDevice();
                cv::cuda::setDevice(deviceIndex);
                checkCudaErrorCode(cudaSetDevice(deviceIndex), __func__);
            }
            cudaFree(0);
        }
    }

    inline bool warmupCompleted() {
        if (msvc_RUNMODE == RUNMODE::PROFILING) {
            return msvc_profWarmupCompleted;
        } else if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
            return msvc_batchCount >= msvc_numWarmupBatches;
        }
        return false;
    }

    virtual void dispatchThread() {};

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false);

    virtual void updateCycleTiming() {};

    bool RELOADING = true;

    std::ofstream msvc_logFile;

    bool PAUSE_THREADS = false;

    virtual std::string getModelName() {return "model";}

protected:
    mutable std::mutex msvc_overallMutex;

    AllocationMode msvc_allocationMode = AllocationMode::Conservative;

    std::vector<ThreadSafeFixSizedDoubleQueue *> msvc_InQueue, msvc_OutQueue;
    //
    std::vector<uint8_t> msvc_activeInQueueIndex = {}, msvc_activeOutQueueIndex = {};

    // Used to signal to thread when not to run and to bring thread to a natural end.
    bool STOP_THREADS = false;
    bool STOPPED = false;
    bool READY = false;

    json msvc_configs;
    bool msvc_toReloadConfigs = true;
    /**
     * @brief Running mode of the container, globally set for all microservices inside the container
     * Default to be deployment.
     */
    RUNMODE msvc_RUNMODE = RUNMODE::DEPLOYMENT;

    DROP_MODE msvc_DROP_MODE = DROP_MODE::NO_DROP;

    BATCH_MODE msvc_BATCH_MODE = BATCH_MODE::FIXED;
    // In case `msvc_DROP_MODE` is `LAZY`, this is the time budget left for the current frame
    uint64_t msvc_timeBudgetLeft = 99999999; 

    // GPU index, -1 means CPU
    int8_t msvc_deviceIndex = -1;

    //type
    MicroserviceType msvc_type;

    // in microseconds
    MsvcSLOType msvc_pipelineSLO;
    // in microseconds
    MsvcSLOType msvc_contSLO;
    // 
    uint64_t msvc_contStartTime;
    //
    uint64_t msvc_contEndTime;
    //
    uint64_t msvc_localDutyCycle;
    //
    ClockType msvc_cycleStartTime;
    
    //
    MsvcSLOType msvc_interReqTime = 1;

    //
    uint64_t msvc_overallTotalReqCount = 0;
    //
    uint64_t msvc_outReqCount = 0;
    //
    uint64_t msvc_batchCount = 0;

    std::atomic<unsigned int> msvc_droppedReqCount = 0;
    std::atomic<unsigned int> msvc_avgBatchSize = 0;
    std::atomic<unsigned int> msvc_miniBatchCount = 0;
    std::atomic<unsigned int> msvc_totalReqCount = 0;

    //
    NumMscvType nummsvc_upstreamMicroservices = 0;
    //
    NumMscvType nummsvc_dnstreamMicroservices = 0;

    // The expected shape of the data for the next microservice
    std::vector<std::vector<RequestDataShapeType>> msvc_outReqShape;
    // The shape of the data to be processed by this microservice
    std::vector<RequestDataShapeType> msvc_dataShape;

    RequestShapeType msvc_inferenceShape;

    // Ideal batch size for this microservice, runtime batch size could be smaller though
    BatchSizeType msvc_idealBatchSize;

    // Maximum batch size, only used during profiling
    BatchSizeType msvc_maxBatchSize;

    //
    bool msvc_profWarmupCompleted = false;

    //
    uint16_t msvc_numWarmupBatches = 100;

    // Frame ID, only used during profiling
    int64_t msvc_currFrameID = -1;

    //
    MODEL_DATA_TYPE msvc_modelDataType = MODEL_DATA_TYPE::fp32;

    //
    std::vector<msvcconfigs::NeighborMicroservice> upstreamMicroserviceList;
    //
    std::vector<msvcconfigs::NeighborMicroservice> dnstreamMicroserviceList;
    //
    std::vector<std::pair<int16_t, NumQueuesType>> classToDnstreamMap;

    //
    virtual bool isTimeToBatch() { return true; };

    //
    virtual void updateReqRate(ClockType lastInterReqDuration);

    // Get the frame ID from the path of travel of this request
    inline uint64_t getFrameID(const std::string &path) {
        std::string temp = splitString(path, "]")[0];
        temp = splitString(temp, "|")[2];
        return std::stoull(temp);
    }

    /**
     * @brief Try increasing the batch size by 1, if the batch size is already at the maximum:
     * (1) if in deployment mode, then we keep the batch size at the maximum
     * (2) if in profiling mode, then we stop the thread
     * 
     * @return true if **batch size has been increased**
     * @return false if **otherwise**
     */
    inline bool increaseBatchSize() {
        // If we already have the max batch size, then we can stop the thread
        if (!msvc_profWarmupCompleted) {
            msvc_profWarmupCompleted = true;
            spdlog::get("container_agent")->info("{0:s} Warmup completed, starting profiling.", msvc_name);
            msvc_OutQueue[0]->emplace(
                Request<LocalGPUReqDataType>{
                        {},
                        {},
                        {"WARMUP_COMPLETED"},
                        0,
                        {},
                        {}
                }
            );
            return true;
        }
        if (++msvc_idealBatchSize > msvc_maxBatchSize) {
            if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
                msvc_idealBatchSize = msvc_maxBatchSize;
            }
            return false;
        }
        spdlog::get("container_agent")->info("Batch size increased to {}", msvc_idealBatchSize);
        return true;
    }

    /**
     * @brief ONLY IN PROFILING MODE
     * Check if the frame index of the incoming stream is reset, which is a signal to change the batch size.
     * 
     * @param req_currFrameID 
     * @return true 
     * @return false 
     */
    inline bool checkProfileFrameReset(const int64_t &req_currFrameID) {
        bool reset = false;
        if (msvc_RUNMODE == RUNMODE::PROFILING) {
            // This case the video has been reset, which means the profiling for this current batch size is completed
            if (msvc_currFrameID > req_currFrameID && req_currFrameID == 1) {
                reset = true;
            }
            msvc_currFrameID = req_currFrameID;
        }
        return reset;
    }

    /**
     * @brief ONLY IN PROFILING MODE
     * Check if the frame index of the incoming stream is reset, which is a signal to change the batch size.
     * If the batch size exceeds the value of maximum batch size, then the microservice should stop processing requests
     * 
     */
    inline bool checkProfileEnd(const std::string &path) {
        if (msvc_RUNMODE != RUNMODE::PROFILING) {
            return false;
        }

        bool frameReset = checkProfileFrameReset(getFrameID(path));
        if (frameReset) {
            // The invert operator means if batch size cannot be increased, then we should stop the thread
            return !increaseBatchSize();
        }
        return false;
    }

    inline std::string getOriginStream(const std::string &path) {
        std::string temp = splitString(path, "]")[0];
        temp = splitString(temp, "[").back();
        temp = splitString(temp, "|")[1];
        return temp;
    }

    inline std::string getSenderHost(const std::string &path) {
        auto parts = splitString(path, "[");
        std::string temp = (parts.size() > 1)? *(++parts.rbegin()) : parts.front();
        temp = splitString(temp, "]").front();
        temp = splitString(temp, "|")[0];
        return temp;
    }

    // Logging file path, where each microservice is supposed to log in running metrics
    std::string msvc_microserviceLogPath;
};


RequestData<LocalGPUReqDataType> uploadReqData(
        const RequestData<LocalCPUReqDataType> &cpuData,
        void *cudaPtr = NULL,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);

Request<LocalGPUReqDataType> uploadReq(
        const Request<LocalCPUReqDataType> &cpuReq,
        std::vector<void *> cudaPtr = {},
        cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);


#endif
