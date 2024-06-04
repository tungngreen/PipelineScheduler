#include "baseprocessor.h"

using namespace spdlog;

BaseSoftmaxClassifier::BaseSoftmaxClassifier(const json& jsonConfigs) : BaseClassifier(jsonConfigs) {
    info("{0:s} is created.", msvc_name); 
}

inline uint16_t maxIndex(float* arr, size_t size) {
    float* max_ptr = std::max_element(arr, arr + size);
    return max_ptr - arr;
}

inline void softmax(const float* logits, float* probabilities, int size) {
    float max_logit = logits[0]; // Initialize max_logit with the first element
    for (int i = 1; i < size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i]; // Update max_logit if a larger element is found
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_exp += exp(logits[i] - max_logit);
    }

    for (int i = 0; i < size; ++i) {
        probabilities[i] = exp(logits[i] - max_logit) / sum_exp;
    }
}

void BaseSoftmaxClassifier::classify() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    std::vector<RequestData<LocalGPUReqDataType>> currReq_data;

    // Current incoming equest and request to be sent out to the next
    Request<LocalGPUReqDataType> currReq, outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    // TODO: remove potentially unused variable
    NumQueuesType queueIndex;

    size_t bufferSize;
    RequestDataShapeType shape;

    float *predictedProbs, *predictedLogits;
    // TODO: remove potentially unused variables
    uint16_t *predictedClass;

    auto timeNow = std::chrono::high_resolution_clock::now();

    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            if (RELOADING) {
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                setDevice();
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);
                
                predictedProbs = new float[msvc_idealBatchSize * msvc_numClasses];
                predictedLogits = new float[msvc_idealBatchSize * msvc_numClasses];
                predictedClass = new uint16_t[msvc_idealBatchSize];
                info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }

        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(currReq.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        }
        
        msvc_inReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }

        currReq_batchSize = currReq.req_batchSize;
        trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        currReq_data = currReq.req_data;

        bufferSize = this->msvc_modelDataType * (size_t)currReq_batchSize;
        shape = currReq_data[0].shape;
        for (uint8_t j = 0; j < shape.size(); ++j) {
            bufferSize *= shape[j];
        }
        checkCudaErrorCode(cudaMemcpyAsync(
            (void *) predictedProbs,
            currReq_data[0].data.cudaPtr(),
            bufferSize,
            cudaMemcpyDeviceToHost,
            postProcStream
        ), __func__);

        cudaStreamSynchronize(postProcStream);

        for (uint8_t i = 0; i < currReq_batchSize; ++i) {
            // We consider this when the request was received by the postprocessor
            currReq.req_origGenTime[i].emplace_back(std::chrono::high_resolution_clock::now());

            uint32_t totalInMem = currReq.upstreamReq_data[i].data.rows * currReq.upstreamReq_data[i].data.cols * currReq.upstreamReq_data[i].data.channels() * CV_ELEM_SIZE1(currReq.upstreamReq_data[i].data.type());

            softmax(predictedLogits + i * msvc_numClasses, predictedProbs + i * msvc_numClasses, msvc_numClasses);
            predictedClass[i] = maxIndex(predictedProbs + i * msvc_numClasses, msvc_numClasses);

            if (this->msvc_activeOutQueueIndex.at(queueIndex) == 1) { //Local CPU
                cv::Mat out(currReq.upstreamReq_data[i].data.size(), currReq.upstreamReq_data[i].data.type());
                checkCudaErrorCode(cudaMemcpyAsync(
                    out.ptr(),
                    currReq.upstreamReq_data[i].data.cudaPtr(),
                    currReq.upstreamReq_data[i].data.rows * currReq.upstreamReq_data[i].data.cols * currReq.upstreamReq_data[i].data.channels() * CV_ELEM_SIZE1(currReq.upstreamReq_data[i].data.type()),
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
                checkCudaErrorCode(cudaStreamSynchronize(postProcStream), __func__);
                msvc_OutQueue.at(0)->emplace(
                    Request<LocalCPUReqDataType>{
                        {{currReq.req_origGenTime[i].front(), std::chrono::high_resolution_clock::now()}},
                        {currReq.req_e2eSLOLatency[i]},
                        {currReq.req_travelPath[i]},
                        1,
                        {
                            {currReq.upstreamReq_data[i].shape, out}
                        } //req_data
                    }
                );
                trace("{0:s} emplaced an image to CPU queue.", msvc_name);
            } else {
                msvc_OutQueue.at(0)->emplace(
                    Request<LocalGPUReqDataType>{
                        {{currReq.req_origGenTime[i].front(), std::chrono::high_resolution_clock::now()}},
                        {currReq.req_e2eSLOLatency[i]},
                        {currReq.req_travelPath[i]},
                        1,
                        {
                            currReq.upstreamReq_data[i]
                        }
                    }
                );
                trace("{0:s} emplaced an image to GPU queue.", msvc_name);
            }

            uint32_t totalOutMem = totalInMem;

            /**
             * @brief There are 7 important timestamps to be recorded:
             * 1. When the request was generated
             * 2. When the request was received by the batcher
             * 3. When the request was done preprocessing by the batcher
             * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
             * 5. When the batch inferencer was completed by the inferencer 
             * 6. When the request was received by the postprocessor
             * 7. When each request was completed by the postprocessor
             */
            msvc_batchCount++;
            // If the number of warmup batches has been passed, we start to record the latency
            if (msvc_batchCount > msvc_numWarmupBatches) {
                currReq.req_origGenTime[i].emplace_back(std::chrono::high_resolution_clock::now());
                // TODO: Add the request number
                msvc_processRecords.addRecord(currReq.req_origGenTime[i], currReq_batchSize, totalInMem, totalOutMem, 0);
            }
        }

        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));

    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
}

void BaseSoftmaxClassifier::classifyProfiling() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    std::vector<RequestData<LocalGPUReqDataType>> inferTimeReportData;

    // Current incoming equest and request to be sent out to the next
    Request<LocalGPUReqDataType> inferTimeReportReq;

    // Batch size of current request
    BatchSizeType inferTimeReport_batchSize;

    info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    // TODO: remove potentially unused variable
    NumQueuesType queueIndex;

    size_t bufferSize;
    RequestDataShapeType shape;

    float *predictedProbs, *predictedLogits;
    // TODO: remove potentially unused variables
    uint16_t *predictedClass;

    auto timeNow = std::chrono::high_resolution_clock::now();

    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            if (RELOADING) {
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                setDevice();
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);
                
                predictedLogits = new float[msvc_idealBatchSize * msvc_numClasses];
                predictedProbs = new float[msvc_idealBatchSize * msvc_numClasses];
                predictedClass = new uint16_t[msvc_idealBatchSize];
                info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }

        // Processing the next incoming request
        inferTimeReportReq = msvc_InQueue.at(0)->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(inferTimeReportReq.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        }
        
        msvc_inReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }

        inferTimeReport_batchSize = inferTimeReportReq.req_batchSize;
        trace("{0:s} popped a request of batch size {1:d}", msvc_name, inferTimeReport_batchSize);

        inferTimeReportData = inferTimeReportReq.req_data;

        bufferSize = this->msvc_modelDataType * (size_t)inferTimeReport_batchSize;
        shape = inferTimeReportData[0].shape;
        for (uint8_t j = 0; j < shape.size(); ++j) {
            bufferSize *= shape[j];
        }
        checkCudaErrorCode(cudaMemcpyAsync(
            (void *) predictedProbs,
            inferTimeReportData[0].data.cudaPtr(),
            bufferSize,
            cudaMemcpyDeviceToHost,
            postProcStream
        ), __func__);

        cudaStreamSynchronize(postProcStream);

        for (uint8_t i = 0; i < inferTimeReport_batchSize; ++i) {
            predictedClass[i] = maxIndex(predictedProbs + i * msvc_numClasses, msvc_numClasses);
            timeNow = std::chrono::high_resolution_clock::now();
            inferTimeReportReq.req_origGenTime[i].emplace_back(timeNow);
        }
        
        msvc_OutQueue[0]->emplace(
            Request<LocalCPUReqDataType>{
                inferTimeReportReq.req_origGenTime,
                {},
                inferTimeReportReq.req_travelPath,
                inferTimeReportReq.req_batchSize,
                {}
            }
        );
        

        trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));

    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
}