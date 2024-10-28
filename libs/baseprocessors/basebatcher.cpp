#include "baseprocessor.h"

using namespace spdlog;

void BaseBatcher::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOADING configs...", __func__);

    if (!isConstructing) { // If this is not called from the constructor, then we are loading configs from a file for Microservice class
        Microservice::loadConfigs(jsonConfigs);
    }

    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);

    bool readProfile = readModelProfile(jsonConfigs["msvc_modelProfile"]);

    if (!readProfile && msvc_RUNMODE == RUNMODE::DEPLOYMENT && msvc_taskName != "dsrc" && msvc_taskName != "datasource") {
        spdlog::get("container_agent")->error("{0:s} No model profile found.", __func__);
        exit(1);
    }
}

BaseBatcher::BaseBatcher(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name);

    oldestReqTime = std::chrono::high_resolution_clock::time_point::max();
}

bool BaseBatcher::readModelProfile(const json &profile) {
    const uint16_t NUM_NUMBERS_PER_BATCH = 4;
    if (profile == nullptr) {
        return false;
    }
    if (profile.size() < NUM_NUMBERS_PER_BATCH) {
        return false;
    }
    if (profile.size() % NUM_NUMBERS_PER_BATCH != 0) {
        spdlog::get("container_agent")->warn("{0:s} profile size is not a multiple of {1:d}.", __func__, NUM_NUMBERS_PER_BATCH);
    }
    uint16_t i = 0;
    do {
        uint16_t numElementsLeft = profile.size() - i;
        if (numElementsLeft / NUM_NUMBERS_PER_BATCH <= 0) {
            if (numElementsLeft % NUM_NUMBERS_PER_BATCH != 0) {
                spdlog::get("container_agent")->warn("{0:s} skips the rest as they do not constitue an expected batch profile {1:d}.", __func__, NUM_NUMBERS_PER_BATCH);
            }
            break;
        }
        BatchSizeType batch = profile[i].get<BatchSizeType>();
        msvc_batchInferProfileList[batch].p95prepLat = profile[i + 1].get<BatchSizeType>();
        msvc_batchInferProfileList[batch].p95inferLat = profile[i + 2].get<BatchSizeType>();
        msvc_batchInferProfileList[batch].p95postLat = profile[i + 3].get<BatchSizeType>();

        i += NUM_NUMBERS_PER_BATCH;
    } while (true);
    return true;
}

void BaseBatcher::updateCycleTiming() {
    // The number of cycles since the beginning of  this scheduling round, which is chosen to be the start of the first cycle
    uint64_t numCyclesSince;
    if (msvc_localDutyCycle == 0) {
        numCyclesSince = 0;
    } else {
        numCyclesSince = std::chrono::duration_cast<TimePrecisionType>(
            std::chrono::high_resolution_clock::now() - msvc_cycleStartTime).count() / msvc_localDutyCycle;
    }

    if (msvc_BATCH_MODE != BATCH_MODE::FIXED) {

        // The time when the last cycle started
        ClockType lastCycleStartTime = msvc_cycleStartTime + TimePrecisionType((int) numCyclesSince * msvc_localDutyCycle);
        // The time when the next cycle should start
        ClockType nextCycleStartTime = lastCycleStartTime + TimePrecisionType(msvc_localDutyCycle);

        // The time when the next batch should be batched for execution
        msvc_nextIdealBatchTime = nextCycleStartTime + TimePrecisionType(msvc_contEndTime) -
                            TimePrecisionType(
                                (uint64_t)((msvc_batchInferProfileList.at(msvc_idealBatchSize).p95inferLat +
                                            msvc_batchInferProfileList.at(msvc_idealBatchSize).p95postLat) * 
                                            msvc_idealBatchSize * 1.3)
                            );
    }
    timeout = 100000; //microseconds
}

/**
 * @brief 
 * 
 * @param genTime 
 * @param slo 
 * @param path 
 * @param bufferData 
 * @param prevData 
 */
inline void BaseBatcher::executeBatching(BatchTimeType &genTime, RequestSLOType &slo, RequestPathType &path,
                                  std::vector<RequestData<LocalGPUReqDataType>> &bufferData,
                                  std::vector<RequestData<LocalGPUReqDataType>> &prevData) {
    // if (time < oldestReqTime) {
    //     oldestReqTime = time;
    // }

    // If true, copy the buffer data into the out queue
    ClockType timeNow = std::chrono::high_resolution_clock::now();

    // Moment of batching
    // This is the FOURTH TIMESTAMP
    for (auto &req_genTime: genTime) {
        req_genTime.emplace_back(timeNow);
    }

    Request<LocalGPUReqDataType> outReq = {
            genTime,
            slo,
            path,
            msvc_onBufferBatchSize,
            bufferData,
            prevData
    };

    msvc_batchCount++;

    spdlog::get("container_agent")->trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name,
                                           msvc_onBufferBatchSize);
    msvc_OutQueue[0]->emplace(outReq);
    msvc_onBufferBatchSize = 0;
    genTime.clear();
    path.clear();
    slo.clear();
    bufferData.clear();
    prevData.clear();
    oldestReqTime = std::chrono::high_resolution_clock::time_point::max();

    // spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
    // std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
}

/**
 * @brief Check if it's time to batch the requests in the buffer.
 * If the batch mode is FIXED, it's always wait for the buffer to be filled.
 * If the batch mode is OURS, then we try to be a bit more clever.
 * When the batch is full, then there's nothing else but to batch it.
 * But if its only partially filled, there are two moments to consider:
 * the ideal moment and the must-batch moment which is explained earlier.
 * 
 * @return true True if its time to batch
 * @return false if otherwise
 */
inline bool BaseBatcher::isTimeToBatch() {
    // timeout for the pop() function
    timeout = 100000;
    if ((msvc_RUNMODE == RUNMODE::PROFILING || 
         msvc_BATCH_MODE == BATCH_MODE::FIXED) && 
        msvc_onBufferBatchSize == msvc_idealBatchSize) {
        return true;
    }

    // OURS BATCH MODE
    if (msvc_BATCH_MODE != BATCH_MODE::OURS) {
        return false;
    }
    //First of all, whenever the batch is full, then it's time to batch
    if (msvc_onBufferBatchSize == 0) {
        return false;
    // If the batch is empty, then it doesn't really matter if it's time to batch or not
    } else if (msvc_onBufferBatchSize == msvc_idealBatchSize) {
        spdlog::get("container_agent")->trace("{0:s} got the ideal batch.", msvc_name);
        updateCycleTiming();
        return true;
    }
    // nextIdealBatchTime assumes that the batch is filled with the ideal batch size
    // nextMustBatchTime is to make sure that the oldest request in the buffer is not late
    // If either of the two times is less than the current time, then it's time to batch
    auto timeNow = std::chrono::high_resolution_clock::now();
    if (timeNow > msvc_nextMustBatchTime) {
        spdlog::get("container_agent")->trace("{0:s} must batch.", msvc_name);
        updateCycleTiming();
        return true;
    }
    if (timeNow > msvc_nextIdealBatchTime) {
        spdlog::get("container_agent")->trace("{0:s} reaches ideal batch time.", msvc_name);
        updateCycleTiming();
        return true;
    }

    // Time out until the next batch time calculated by duty cycle
    timeout = std::chrono::duration_cast<TimePrecisionType>(
        msvc_nextIdealBatchTime - timeNow).count();
    timeout = std::max(timeout, (uint64_t)0);
    
    uint64_t lastReqWaitTime = std::chrono::duration_cast<TimePrecisionType>(
            timeNow - oldestReqTime).count();

    // This is the timeout till the moment the oldest request has to be processed
    // 1.2 is to make sure the request is not late
    // Since this calculation is before the preprocessing in the preprocessor function, we add one preprocessing time unit
    // into the total reserved time for the requests already in batch.
    // If this preprocessing doesnt happen (as the next request doesn't come as expectd), then the batcher will just batch
    // the next time as this timer is expired
    uint64_t timeOutByLastReq = msvc_contSLO - lastReqWaitTime - 
                            (msvc_batchInferProfileList.at(msvc_onBufferBatchSize).p95inferLat +
                            msvc_batchInferProfileList.at(msvc_onBufferBatchSize).p95postLat) * msvc_onBufferBatchSize * 1.2 -
                            msvc_batchInferProfileList.at(msvc_onBufferBatchSize).p95prepLat * 1.2;
    timeOutByLastReq = std::max((uint64_t) 0, timeOutByLastReq);
    msvc_nextMustBatchTime = timeNow + TimePrecisionType(timeOutByLastReq);
    // Ideal batch size is calculated based on the profiles so its always confined to the cycle,
    // So we ground must batch time to the ideal batch time to make sure it is so as well.
    if (msvc_nextMustBatchTime > msvc_nextIdealBatchTime) {
        msvc_nextMustBatchTime = msvc_nextIdealBatchTime;
    }
    
    timeout = std::min(timeout, timeOutByLastReq);
    // If the timeout is less than 100 microseconds, then it's time to batch
    if (timeout < 100 || timeOutByLastReq < 100) { //microseconds
        updateCycleTiming();
        return true;
    }
    return false;
}


void BaseBatcher::batchRequests() {
    // Batch reqs' gen time
    BatchTimeType outBatch_genTime;

    // Batch reqs' slos
    RequestSLOType outBatch_slo;

    // Batch reqs' paths
    RequestPathType outReq_path;

    // Batch reqs' paths
    RequestPathType outBatch_path;

    Request<LocalGPUReqDataType> currReq;

    // Buffer memory for each batch
    std::vector<RequestData<LocalGPUReqDataType>> bufferData;

    // Data carried from upstream microservice to be processed at a downstream
    std::vector<RequestData<LocalGPUReqDataType>> prevData;

    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name);

    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                if (msvc_toReloadConfigs) {
                    loadConfigs(msvc_configs, true);
                    msvc_toReloadConfigs = false;
                }

                updateCycleTiming();

                setDevice();
                outBatch_genTime.clear();
                outReq_path.clear();
                outBatch_slo.clear();
                bufferData.clear();
                prevData.clear();

                spdlog::get("container_agent")->info("{0:s} is RELOADED.", msvc_name);

                RELOADING = false;
                READY = true;
            }
            continue;
        }
        // Processing the next incoming request
        // even if a valid request is not popped, if it's time to batch, we should batch the requests
        // as it doesn't take much time and otherwise, we are running the risk of the whole batch being late.
        // if (isTimeToBatch()) {
        //     executeBatch(outBatch_genTime, outBatch_slo, outBatch_path, bufferData, prevData);
        // }

        // Processing the next incoming request
        // Current incoming equest and request to be sent out to the next
        Request<LocalGPUReqDataType> currReq = msvc_InQueue.at(0)->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(currReq.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        
        /**
         * @brief ONLY IN PROFILING MODE
         * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
         */
        } else if (strcmp(currReq.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(currReq);
            continue;
        }  else if (strcmp(currReq.req_travelPath[0].c_str(), "WARMUP_COMPLETED") == 0) {
            msvc_profWarmupCompleted = true;
            spdlog::get("container_agent")->info("{0:s} received the signal that the warmup is completed.", msvc_name);
            msvc_OutQueue[0]->emplace(currReq);
            continue;
        }

        if (msvc_onBufferBatchSize == 0 && msvc_BATCH_MODE != BATCH_MODE::FIXED) {
            oldestReqTime = std::chrono::high_resolution_clock::now();
            // We update the oldest request time and the must batch time for this request
            // We try to account for its inference and postprocessing time
            uint64_t timeOutByLastReq = msvc_contSLO - 
                            (msvc_batchInferProfileList.at(1).p95inferLat +
                            msvc_batchInferProfileList.at(1).p95postLat) * 1.2;
            msvc_nextMustBatchTime = oldestReqTime + TimePrecisionType(timeOutByLastReq);
        }

        msvc_overallTotalReqCount++;
        msvc_onBufferBatchSize++;
        outBatch_genTime.insert(outBatch_genTime.end(), currReq.req_origGenTime.begin(), currReq.req_origGenTime.end());
        outBatch_slo.insert(outBatch_slo.end(), currReq.req_e2eSLOLatency.begin(), currReq.req_e2eSLOLatency.end());
        outBatch_path.insert(outBatch_path.end(), currReq.req_travelPath.begin(), currReq.req_travelPath.end());
        bufferData.insert(bufferData.end(), currReq.req_data.begin(), currReq.req_data.end());
        prevData.insert(prevData.end(), currReq.upstreamReq_data.begin(), currReq.upstreamReq_data.end());

        if (isTimeToBatch()) {
            executeBatching(outBatch_genTime, outBatch_slo, outBatch_path, bufferData, prevData);
        }
    }

}