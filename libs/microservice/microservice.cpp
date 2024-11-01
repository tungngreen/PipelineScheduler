#include<microservice.h>

using namespace msvcconfigs;

void msvcconfigs::from_json(const json &j, msvcconfigs::NeighborMicroserviceConfigs &val) {
    j.at("nb_name").get_to(val.name);
    j.at("nb_commMethod").get_to(val.commMethod);
    j.at("nb_link").get_to(val.link);
    j.at("nb_maxQueueSize").get_to(val.maxQueueSize);
    j.at("nb_classOfInterest").get_to(val.classOfInterest);
    j.at("nb_expectedShape").get_to(val.expectedShape);
}

void msvcconfigs::from_json(const json &j, msvcconfigs::BaseMicroserviceConfigs &val) {
    j.at("msvc_contName").get_to(val.msvc_contName);
    j.at("msvc_name").get_to(val.msvc_name);
    j.at("msvc_type").get_to(val.msvc_type);
    j.at("msvc_pipelineSLO").get_to(val.msvc_pipelineSLO);
    j.at("msvc_idealBatchSize").get_to(val.msvc_idealBatchSize);
    j.at("msvc_maxQueueSize").get_to(val.msvc_maxQueueSize);
    j.at("msvc_dataShape").get_to(val.msvc_dataShape);
    j.at("msvc_deviceIndex").get_to(val.msvc_deviceIndex);
    j.at("msvc_containerLogPath").get_to(val.msvc_containerLogPath);
    j.at("msvc_RUNMODE").get_to(val.msvc_RUNMODE);
    j.at("msvc_upstreamMicroservices").get_to(val.msvc_upstreamMicroservices);
    j.at("msvc_dnstreamMicroservices").get_to(val.msvc_dnstreamMicroservices);
}

void msvcconfigs::to_json(json &j, const msvcconfigs::NeighborMicroserviceConfigs &val) {
    j["nb_name"] = val.name;
    j["nb_commMethod"] = val.commMethod;
    j["nb_link"] = val.link;
    j["nb_maxQueueSize"] = val.maxQueueSize;
    j["nb_classOfInterest"] = val.classOfInterest;
    j["nb_expectedShape"] = val.expectedShape;
}

void msvcconfigs::to_json(json &j, const msvcconfigs::BaseMicroserviceConfigs &val) {
    j["msvc_name"] = val.msvc_name;
    j["msvc_type"] = val.msvc_type;
    j["msvc_pipelineSLO"] = val.msvc_pipelineSLO;
    j["msvc_idealBatchSize"] = val.msvc_idealBatchSize;
    j["msvc_dataShape"] = val.msvc_dataShape;
    j["msvc_maxQueueSize"] = val.msvc_maxQueueSize;
    j["msvc_upstreamMicroservices"] = val.msvc_upstreamMicroservices;
    j["msvc_dnstreamMicroservices"] = val.msvc_dnstreamMicroservices;
}

void Microservice::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    BaseMicroserviceConfigs configs = jsonConfigs.get<BaseMicroserviceConfigs>();

    // Identifiers
    msvc_name = configs.msvc_name;
    msvc_containerName = configs.msvc_contName;
    msvc_experimentName = jsonConfigs.at("msvc_experimentName");
    msvc_pipelineName = jsonConfigs.at("msvc_pipelineName");
    msvc_taskName = jsonConfigs.at("msvc_taskName");
    msvc_hostDevice = jsonConfigs.at("msvc_hostDevice");
    msvc_systemName = jsonConfigs.at("msvc_systemName");
    msvc_BATCH_MODE = jsonConfigs.at("msvc_batchMode");
    msvc_DROP_MODE = jsonConfigs.at("msvc_dropMode");
    msvc_timeBudgetLeft = jsonConfigs.at("msvc_timeBudgetLeft");
    msvc_pipelineSLO = jsonConfigs.at("msvc_pipelineSLO");
    msvc_contStartTime = jsonConfigs.at("msvc_contStartTime");
    msvc_contEndTime = jsonConfigs.at("msvc_contEndTime");
    msvc_contSLO = msvc_contEndTime - msvc_contStartTime;
    msvc_localDutyCycle = jsonConfigs.at("msvc_localDutyCycle");
    msvc_cycleStartTime = ClockType(TimePrecisionType(jsonConfigs.at("msvc_cycleStartTime")));
    msvc_idealBatchSize = configs.msvc_idealBatchSize;

    // Configurations
    msvc_dataShape = configs.msvc_dataShape;
    msvc_type = configs.msvc_type;
    PAUSE_THREADS = true;
    msvc_deviceIndex = configs.msvc_deviceIndex;
    msvc_RUNMODE = configs.msvc_RUNMODE;

    if (msvc_taskName != "dsrc" && msvc_taskName != "datasource") {
        msvc_maxBatchSize = jsonConfigs.at("msvc_maxBatchSize");
        msvc_allocationMode = static_cast<AllocationMode>(jsonConfigs.at("msvc_allocationMode"));

        if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
            msvc_numWarmupBatches = jsonConfigs.at("msvc_numWarmUpBatches");
        } else if (msvc_RUNMODE == RUNMODE::PROFILING) {
            msvc_numWarmupBatches = jsonConfigs.at("profile_numWarmUpBatches");
        }
        // During profiling, we want to have at least 120 requests for warming ups
        // Results before warming up are not reliable
        if ((msvc_numWarmupBatches * msvc_idealBatchSize) < 120) {
            msvc_numWarmupBatches = std::ceil(120 / msvc_idealBatchSize) + 1;
        }
    }
    if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
        msvc_microserviceLogPath = configs.msvc_containerLogPath + "/" + msvc_pipelineName + "_" + msvc_name + ".txt";
    } else {
        msvc_microserviceLogPath = configs.msvc_containerLogPath + "/" + msvc_pipelineName + "_" + msvc_name + "_" + getTimestampString() + ".txt";
    }

    // Initialize the queues    
    if (isConstructing) {
        msvc_InQueue = {};
        msvc_OutQueue = {};
        msvc_outReqShape = {};

        for (auto it = configs.msvc_dnstreamMicroservices.begin(); it != configs.msvc_dnstreamMicroservices.end(); ++it) {
            msvc_OutQueue.emplace_back(new ThreadSafeFixSizedDoubleQueue(configs.msvc_maxQueueSize, it->classOfInterest, it->name));
            // Create downstream neigbor config and push that into a list for information later
            // Local microservice supposedly has only 1 downstream but `receiver` microservices could have multiple senders.
            NeighborMicroservice dnStreamMsvc = NeighborMicroservice(*it, nummsvc_dnstreamMicroservices);
            dnstreamMicroserviceList.emplace_back(dnStreamMsvc);
            // This maps the data class to be sent to this downstream microservice and the microservice's index.
            std::pair<int16_t, uint16_t> map = {dnStreamMsvc.classOfInterest, nummsvc_dnstreamMicroservices++};
            classToDnstreamMap.emplace_back(map);
            msvc_outReqShape.emplace_back(it->expectedShape); // This is a dummy value for now
            if (it->commMethod == CommMethod::localGPU) {
                msvc_activeOutQueueIndex.emplace_back(2);
            } else {
                msvc_activeOutQueueIndex.emplace_back(1);
                if (it->commMethod == CommMethod::encodedCPU) {
                    msvc_OutQueue.back()->setEncoded(true);
                }
            }
        }

        for (auto it = configs.msvc_upstreamMicroservices.begin(); it != configs.msvc_upstreamMicroservices.end(); ++it) {
            NeighborMicroservice upStreamMsvc = NeighborMicroservice(*it, nummsvc_upstreamMicroservices++);
            upstreamMicroserviceList.emplace_back(upStreamMsvc);
            if (it->commMethod == CommMethod::localGPU) {
                msvc_activeInQueueIndex.emplace_back(2);
            } else {
                msvc_activeInQueueIndex.emplace_back(1);
            }
        }
    }
}

/**
 * @brief Construct a new Microservice< In Type>:: Microservice object
 * 
 * @tparam InType 
 * @param configs 
 */
Microservice::Microservice(const json &jsonConfigs) {
    Microservice::loadConfigs(jsonConfigs, true);
    msvc_configs = jsonConfigs;
}

Microservice::Microservice(const Microservice &other) {
    // Lock other mutexes down for correctness
    std::lock(this->msvc_overallMutex, other.msvc_overallMutex);
    std::lock_guard<std::mutex> lock1(other.msvc_overallMutex, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(this->msvc_overallMutex, std::adopt_lock);

    // identifiers
    msvc_name = other.msvc_name;
    msvc_experimentName = other.msvc_experimentName;
    msvc_pipelineName = other.msvc_pipelineName;
    msvc_containerName = other.msvc_containerName;
    msvc_taskName = other.msvc_taskName;
    msvc_hostDevice = other.msvc_hostDevice;
    msvc_systemName = other.msvc_systemName;
    msvc_type = other.msvc_type;

    // Modes
    msvc_RUNMODE = other.msvc_RUNMODE;
    msvc_DROP_MODE = other.msvc_DROP_MODE;
    msvc_DROP_MODE = other.msvc_DROP_MODE;
    msvc_allocationMode = other.msvc_allocationMode;

    // Batching and Timing
    msvc_idealBatchSize = other.msvc_idealBatchSize;
    msvc_timeBudgetLeft = other.msvc_timeBudgetLeft;
    msvc_pipelineSLO = other.msvc_pipelineSLO;
    msvc_contStartTime = other.msvc_contStartTime;
    msvc_contEndTime = other.msvc_contEndTime;
    msvc_contSLO = other.msvc_contSLO;
    msvc_localDutyCycle = other.msvc_localDutyCycle;
    msvc_cycleStartTime = other.msvc_cycleStartTime;
    msvc_maxBatchSize = other.msvc_maxBatchSize;

    // IOs
    msvc_dataShape = other.msvc_dataShape;
    msvc_outReqShape = other.msvc_outReqShape;
    msvc_activeInQueueIndex = other.msvc_activeInQueueIndex;
    msvc_activeOutQueueIndex = other.msvc_activeOutQueueIndex;
    upstreamMicroserviceList = other.upstreamMicroserviceList;
    dnstreamMicroserviceList = other.dnstreamMicroserviceList;
    msvc_InQueue = other.msvc_InQueue;
    msvc_OutQueue = other.msvc_OutQueue;
    classToDnstreamMap = other.classToDnstreamMap;

    msvc_modelDataType = other.msvc_modelDataType;

    // logging
    msvc_microserviceLogPath = other.msvc_microserviceLogPath;

    // GPU
    msvc_deviceIndex = other.msvc_deviceIndex;

    // Current status
    msvc_currFrameID = other.msvc_currFrameID;

    // Profiling and Warmup
    msvc_numWarmupBatches = other.msvc_numWarmupBatches;
    msvc_profWarmupCompleted = other.msvc_profWarmupCompleted;
    msvc_numWarmupBatches = other.msvc_numWarmupBatches;
}

/**
 * @brief 
 * 
 * @tparam InType 
 * @param lastInterReqDuration 
 */
void Microservice::updateReqRate(ClockType lastInterReqDuration) {
    msvc_interReqTime = 1;
}

RequestData<LocalGPUReqDataType> uploadReqData(
    const RequestData<LocalCPUReqDataType>& cpuData,
    void * cudaPtr,
    cv::cuda::Stream &stream
) {
    RequestData<LocalGPUReqDataType> gpuData;

    gpuData.shape = cpuData.shape;
    if (cudaPtr != NULL) {
        gpuData.data = cv::cuda::GpuMat(
            cpuData.data.rows,
            cpuData.data.cols,
            cpuData.data.type(),
            cudaPtr
        );
    } else {
        gpuData.data = cv::cuda::GpuMat(
            cpuData.data.rows,
            cpuData.data.cols,
            cpuData.data.type()
        );
    }

    gpuData.data.upload(cpuData.data);

    return gpuData;
}


Request<LocalGPUReqDataType> uploadReq(
    
    const Request<LocalCPUReqDataType>& cpuReq,
    std::vector<void *> cudaPtr,
    cv::cuda::Stream &stream
) {
    Request<LocalGPUReqDataType> gpuReq;
    gpuReq.req_origGenTime = cpuReq.req_origGenTime;
    gpuReq.req_e2eSLOLatency = cpuReq.req_e2eSLOLatency;
    gpuReq.req_travelPath = cpuReq.req_travelPath;
    gpuReq.req_batchSize = cpuReq.req_batchSize;

    for (uint16_t i = 0; i < cpuReq.req_data.size(); i++) {
        RequestData<LocalGPUReqDataType> req_data;
        if (cudaPtr.size() > 0) {
            req_data = uploadReqData(cpuReq.req_data[i], cudaPtr[i]);
        } else {
            req_data = uploadReqData(cpuReq.req_data[i]);
        }
        gpuReq.req_data.emplace_back(req_data);
    }
    return gpuReq;
}
