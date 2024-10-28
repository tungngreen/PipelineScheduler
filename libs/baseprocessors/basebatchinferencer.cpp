#include "baseprocessor.h"

using namespace spdlog;
using json = nlohmann::ordered_json;
using namespace trt;

/**
 * @brief Load the configurations from the json file
 * 
 * @param jsonConfigs 
 * @param isConstructing 
 */
void BaseBatchInferencer::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOANDING configs...", __func__);
    if (!isConstructing) { // If the function is called from the constructor, the configs are already loaded.
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }

    msvc_engineConfigs = jsonConfigs.get<TRTConfigs>();
    if (msvc_allocationMode == AllocationMode::Conservative) {
        msvc_engineConfigs.maxBatchSize = msvc_idealBatchSize;
    } else if (msvc_allocationMode == AllocationMode::Aggressive) {
        msvc_engineConfigs.maxBatchSize = msvc_maxBatchSize;
    }
    msvc_engineConfigs.deviceIndex = msvc_deviceIndex;

    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);
}

/**
 * @brief Construct a new Base Preprocessor that inherites the LocalGPUDataMicroservice given the `InType`
 * 
 * @param configs 
 */
BaseBatchInferencer::BaseBatchInferencer(const json &jsonConfigs) : Microservice(jsonConfigs){

    // Load the configurations from the json file
    loadConfigs(jsonConfigs);

    // msvc_engineInputBuffers = msvc_inferenceEngine->getInputBuffers();
    // msvc_engineOutputBuffers = msvc_inferenceEngine->getOutputBuffers();

    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name);
}

void BaseBatchInferencer::inference() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Vector of GPU Mat to copy data into and out of the TRT Engine
    std::vector<LocalGPUReqDataType> trtInBuffer, trtOutBuffer;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;   

    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> data;

    Request<LocalGPUReqDataType> outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name);

    cudaStream_t inferenceStream;

    auto timeNow = std::chrono::high_resolution_clock::now();
    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
                READY = false;

                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                /**
                 * @brief Set the current working device
                 * 
                 */
                setDevice();
                // Create a new cuda stream for inference
                checkCudaErrorCode(cudaStreamCreate(&inferenceStream), __func__);

                /**
                 * @brief Generate a new instance of the inference engine with the new configurations
                 * 
                 */
                if (msvc_inferenceEngine != nullptr) {
                    delete msvc_inferenceEngine;
                }
                msvc_inferenceEngine = new Engine(msvc_engineConfigs);
                msvc_inferenceEngine->loadNetwork();

                outReqData.clear();
                trtInBuffer.clear();
                trtOutBuffer.clear();
                
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //spdlog::info("{0:s} is being PAUSED.", msvc_name);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }
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
        msvc_overallTotalReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        // if (msvc_inReqCount > 1) {
        //     updateReqRate(currReq_genTime);
        // }
        for (auto& req_genTime : currReq.req_origGenTime) {
            req_genTime.emplace_back(currReq_recvTime); //FIFTH_TIMESTAMP
        }

        // Do batched inference with TRT
        currReq_batchSize = currReq.req_batchSize;
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        for (std::size_t i = 0; i < currReq_batchSize; ++i) {
            trtInBuffer.emplace_back(currReq.req_data[i].data);
        }
        spdlog::get("container_agent")->trace("{0:s} extracts inference data from message. Run inference!", msvc_name);
        msvc_inferenceEngine->runInference(trtInBuffer, trtOutBuffer, currReq_batchSize, inferenceStream);
        spdlog::get("container_agent")->trace("{0:s} finished INFERENCE.", msvc_name);


        // After inference, 4 buffers are filled with memory, which we need to carry to post processor.
        // We put 4 buffers into a vector along with their respective shapes for the post processor to interpret.
        for (std::size_t i = 0; i < msvc_outReqShape.at(0).size(); ++i) {
            data = {
                msvc_outReqShape.at(0).at(i),
                trtOutBuffer[i]
            };
            outReqData.emplace_back(data);
        }

        cv::cuda::GpuMat dummy;
        // Add a piece of data at the end of `req_data` to let the next msvc (e.g., cropper) knows about the shape used
        // udring inference.
        // This shape together with the shape of the original data will be used to scale the bounding boxes
        RequestData<LocalGPUReqDataType> shapeGuide = {
            currReq.req_data[0].shape,
            dummy
        };

        timeNow = std::chrono::high_resolution_clock::now();
        /**
         * @brief There are six important timestamps to be recorded:
         * 1. When the request was generated
         * 2. When the request was received by the preprocessor
         * 3. When the request was done preprocessing by the preprocessor
         * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
         * 5. When the batch inferencer was completed by the inferencer 
         * 6. When each request was completed by the postprocessor
         */
        for (auto& req_genTime : currReq.req_origGenTime) {
            req_genTime.emplace_back(timeNow); //SIXTH_TIMESTAMP
        }

        outReqData.emplace_back(shapeGuide);


        // Packing everything inside the `outReq` to be sent to and processed at the next microservice
        outReq = {
            currReq.req_origGenTime,
            currReq.req_e2eSLOLatency,
            currReq.req_travelPath,
            currReq_batchSize,
            outReqData, //req_data
            currReq.upstreamReq_data // upstreamReq_data
        };
        // // After inference, the gpumat inside `inbuffer` is no longer used and can be freed.
        // for (std::size_t i = 0; i < trtInBuffer.size(); i++) {
        //     checkCudaErrorCode(cudaFree(trtInBuffer.at(i).cudaPtr()));
        // }
        spdlog::get("container_agent")->trace("{0:s} emplaced a request for a batch size of {1:d}", msvc_name, currReq_batchSize);

        msvc_OutQueue[0]->emplace(outReq);
        outReqData.clear();
        trtInBuffer.clear();
        trtOutBuffer.clear();

        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
        spdlog::get("container_agent")->flush();
    }
    checkCudaErrorCode(cudaStreamDestroy(inferenceStream), __func__);
    msvc_logFile.close();
}

void BaseBatchInferencer::inferenceProfiling() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Vector of GPU Mat to copy data into and out of the TRT Engine
    std::vector<LocalGPUReqDataType> trtInBuffer, trtOutBuffer;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;   

    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> data;

    Request<LocalGPUReqDataType> outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name);

    auto timeNow = std::chrono::high_resolution_clock::now();

    cudaStream_t inferenceStream;
    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
                READY = false;
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                /**
                 * @brief Set the current working device
                 * 
                 */
                setDevice();
                // Create a new cuda stream for inference
                checkCudaErrorCode(cudaStreamCreate(&inferenceStream), __func__);
                


                /**
                 * @brief Generate a new instance of the inference engine with the new configurations
                 * 
                 */
                if (msvc_inferenceEngine != nullptr) {
                    delete msvc_inferenceEngine;
                }
                msvc_inferenceEngine = new Engine(msvc_engineConfigs);
                msvc_inferenceEngine->loadNetwork();

                outReqData.clear();
                trtInBuffer.clear();
                trtOutBuffer.clear();
                
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        // Current incoming equest and request to be sent out to the next
        Request<LocalGPUReqDataType> currReq = msvc_InQueue.at(0)->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(currReq.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        }
        
        msvc_overallTotalReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (msvc_overallTotalReqCount > 1) {
            updateReqRate(currReq_genTime);
        }

        // Do batched inference with TRT
        currReq_batchSize = currReq.req_batchSize;
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        for (std::size_t i = 0; i < currReq_batchSize; ++i) {
            trtInBuffer.emplace_back(currReq.req_data[i].data);
        }
        spdlog::get("container_agent")->trace("{0:s} extracts inference data from message. Run inference!", msvc_name);
        msvc_inferenceEngine->runInference(trtInBuffer, trtOutBuffer, currReq_batchSize, inferenceStream);
        spdlog::get("container_agent")->trace("{0:s} finished INFERENCE.", msvc_name);

        // After inference, 4 buffers are filled with memory, which we need to carry to post processor.
        // We put 4 buffers into a vector along with their respective shapes for the post processor to interpret.
        for (std::size_t i = 0; i < msvc_outReqShape.at(0).size(); ++i) {
            data = {
                msvc_outReqShape.at(0).at(i),
                trtOutBuffer[i]
            };
            outReqData.emplace_back(data);
        }

        cv::cuda::GpuMat dummy;
        // Add a piece of data at the end of `req_data` to let the next msvc (e.g., cropper) knows about the shape used
        // udring inference.
        // This shape together with the shape of the original data will be used to scale the bounding boxes
        RequestData<LocalGPUReqDataType> shapeGuide = {
            currReq.req_data[0].shape,
            dummy
        };
        outReqData.emplace_back(shapeGuide);

        timeNow = std::chrono::high_resolution_clock::now();

        /**
         * @brief During profiling mode, there are six important timestamps to be recorded:
         * 1. When the request was generated
         * 2. When the request was received by the preprocessor
         * 3. When the request was done preprocessing by the preprocessor
         * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
         * 5. When the batch inferencer was completed by the inferencer 
         * 6. When each request was completed by the postprocessor
         */
        for (auto& req_genTime : currReq.req_origGenTime) {
            req_genTime.emplace_back(timeNow); //FIFTH_TIMESTAMP
        }

        // Packing everything inside the `outReq` to be sent to and processed at the next microservice
        outReq = {
            currReq.req_origGenTime,
            currReq.req_e2eSLOLatency,
            currReq.req_travelPath,
            currReq_batchSize,
            outReqData, //req_data
            currReq.upstreamReq_data // upstreamReq_data
        };
        // // After inference, the gpumat inside `inbuffer` is no longer used and can be freed.
        // for (std::size_t i = 0; i < trtInBuffer.size(); i++) {
        //     checkCudaErrorCode(cudaFree(trtInBuffer.at(i).cudaPtr()));
        // }
        spdlog::get("container_agent")->info("{0:s} emplaced a request for a batch size of {1:d}", msvc_name, currReq_batchSize);

        msvc_OutQueue[0]->emplace(outReq);
        outReqData.clear();
        trtInBuffer.clear();
        trtOutBuffer.clear();

        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
    }
    checkCudaErrorCode(cudaStreamDestroy(inferenceStream), __func__);
    msvc_logFile.close();
}

RequestShapeType BaseBatchInferencer::getInputShapeVector() {
    RequestShapeType shape = {};
    std::vector<nvinfer1::Dims3> engineInDims = msvc_inferenceEngine->getInputDims();
    for (uint16_t i = 0; i < engineInDims.size(); ++i) {
        RequestDataShapeType insideShape;
        for (int32_t j = 0; j < engineInDims.at(i).nbDims; ++j) {
            insideShape.emplace_back(engineInDims.at(i).d[j]);
        }
        shape.emplace_back(insideShape);
    }
    return shape;
}

RequestShapeType BaseBatchInferencer::getOutputShapeVector() {
    RequestShapeType shape = {};
    std::vector<nvinfer1::Dims32> engineOutDims = msvc_inferenceEngine->getOutputDims();
    for (uint16_t i = 0; i < engineOutDims.size(); ++i) {
        RequestDataShapeType insideShape;
        for (int32_t j = 0; j < engineOutDims.at(i).nbDims; ++j) {
            insideShape.emplace_back(engineOutDims.at(i).d[j]);
        }
        shape.emplace_back(insideShape);
    }
    return shape;
}