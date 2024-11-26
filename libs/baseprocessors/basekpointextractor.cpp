#include "baseprocessor.h"

using namespace spdlog;


BaseKPointExtractorConfigs BaseKPointExtractor::loadConfigsFromJson(const json &config) {
    BaseKPointExtractorConfigs configs;
    return configs;
}

void BaseKPointExtractor::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) { // If the microservice is being reloaded
        BasePostprocessor::loadConfigs(jsonConfigs, isConstructing);
    }
    BaseKPointExtractorConfigs configs = loadConfigsFromJson(jsonConfigs);
}


BaseKPointExtractor::BaseKPointExtractor(const json &jsonConfigs) : BasePostprocessor(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name); 
}

inline uint16_t maxIndex(float* arr, size_t size) {
    float* max_ptr = std::max_element(arr, arr + size);
    return max_ptr - arr;
}
void BaseKPointExtractor::extractor() {

    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;
    // Path
    std::string currReq_path;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;
    // List of images carried from the previous microservice here to be cropped from.
    std::vector<RequestData<LocalGPUReqDataType>> imageList;
    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> reqData;

    std::vector<RequestData<LocalGPUReqDataType>> currReq_data;

    // List of bounding boxes cropped from one single image
    std::vector<cv::cuda::GpuMat> singleImageBBoxList;

    // Current incoming equest and request to be sent out to the next
    Request<LocalGPUReqDataType> currReq, outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestDataShapeType bboxShape;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    NumQueuesType queueIndex = 0;

    size_t bufferSize;
    RequestDataShapeType shape;


    float *keyPoints = nullptr;

    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (PAUSE_THREADS) {
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

                BatchSizeType batchSize = msvc_allocationMode == AllocationMode::Conservative ? msvc_idealBatchSize : msvc_maxBatchSize;
                keyPoints = new float[batchSize * msvc_dataShape[0][0] * msvc_dataShape[0][1] * msvc_dataShape[0][2]];

                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2();
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
        } else if (strcmp(currReq.req_travelPath[0].c_str(), "WARMUP_COMPLETED") == 0) {
            msvc_profWarmupCompleted = true;
            spdlog::get("container_agent")->info("{0:s} received the signal that the warmup is completed.", msvc_name);
            msvc_OutQueue[0]->emplace(currReq);
            continue;
        }

        // 10. The moment the batch is received at the cropper (TENTH_TIMESTAMP)
        auto timeNow = std::chrono::high_resolution_clock::now();
        for (auto& req_genTime : currReq.req_origGenTime) {
            req_genTime.emplace_back(timeNow);
        }

        currReq_batchSize = currReq.req_batchSize;
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        currReq_data = currReq.req_data;

        bufferSize = msvc_modelDataType * (size_t)currReq_batchSize;
        shape = currReq_data[0].shape;
        for (uint8_t j = 0; j < shape.size(); ++j) {
            bufferSize *= shape[j];
        }
        checkCudaErrorCode(cudaMemcpyAsync(
            (void *) keyPoints,
            currReq_data[0].data.cudaPtr(),
            bufferSize,
            cudaMemcpyDeviceToHost,
            postProcStream
        ), __func__);

        cudaStreamSynchronize(postProcStream);
        spdlog::get("container_agent")->trace("{0:s} unloaded 4 buffers to CPU {1:d}", msvc_name, currReq_batchSize);

        imageList = currReq.upstreamReq_data;

        for (BatchSizeType i = 0; i < currReq.req_batchSize; i++) {
            auto numImagesInFrame = currReq.req_concatInfo[i].numImagesAdded;
            std::vector<MemUsageType> totalInMem(numImagesInFrame, 0), totalOutMem(numImagesInFrame, 0), totalEncodedOutMem(numImagesInFrame, 0);
            msvc_overallTotalReqCount++;

            // 11. The moment the request starts being processed by the cropper, after the batch was unloaded (ELEVENTH_TIMESTAMP)
            for (uint8_t concatInd = 0; concatInd < numImagesInFrame; concatInd++) {
                uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + concatInd;
                currReq.req_origGenTime[imageIndexInBatch].emplace_back(std::chrono::high_resolution_clock::now());
            }

            for (uint8_t j = 0; j < numImagesInFrame; j++) {
                uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + j;
                totalInMem[j] = (imageList[imageIndexInBatch].data.channels() *
                                 imageList[imageIndexInBatch].data.rows * imageList[imageIndexInBatch].data.cols *
                                 CV_ELEM_SIZE1(imageList[imageIndexInBatch].data.type()));
                totalOutMem[j] = totalInMem.back();

                currReq_path = currReq.req_travelPath[imageIndexInBatch] + "|1|1";

                if (msvc_activeOutQueueIndex.at(0) == 1) { //Local CPU
                    cv::Mat out;
                    cv::cuda::Stream postProcCVStream = cv::cuda::Stream();
                    currReq.upstreamReq_data[imageIndexInBatch].data.download(out, postProcCVStream);
                    postProcCVStream.waitForCompletion();
                    if (msvc_OutQueue.at(0)->getEncoded()) {
                        out = encodeResults(out);
                        totalEncodedOutMem[j] = out.channels() * out.rows * out.cols * CV_ELEM_SIZE1(out.type());
                    }

                    currReq_path += "|" + std::to_string(totalEncodedOutMem[j]) + "|" + std::to_string(totalInMem[j]) + "]";

                    msvc_OutQueue.at(0)->emplace(
                        Request<LocalCPUReqDataType>{
                            {{currReq.req_origGenTime[imageIndexInBatch].front(), std::chrono::high_resolution_clock::now()}},
                            {currReq.req_e2eSLOLatency[imageIndexInBatch]},
                            {currReq_path},
                            1,
                            {
                                {currReq.upstreamReq_data[imageIndexInBatch].shape, out}
                            } //req_data
                        }
                    );
                    spdlog::get("container_agent")->trace("{0:s} emplaced an image to CPU queue {1:d}.", msvc_name, 0);
                } else {
                    currReq_path += "|0|" + std::to_string(totalInMem[j]) + "]";
                    msvc_OutQueue.at(0)->emplace(
                        Request<LocalGPUReqDataType>{
                            {{currReq.req_origGenTime[imageIndexInBatch].front(), std::chrono::high_resolution_clock::now()}},
                            {currReq.req_e2eSLOLatency[imageIndexInBatch]},
                            {currReq_path},
                            1,
                            {currReq.upstreamReq_data[imageIndexInBatch]}, //req_data
                        }
                    );
                    spdlog::get("container_agent")->trace("{0:s} emplaced an image to GPU queue {1:d}.", msvc_name, 0);
                }

                if (warmupCompleted()) {
                    // 12. The moment the request is done being processed by the postprocessor (TWELFTH_TIMESTAMP)
                    currReq.req_origGenTime[imageIndexInBatch].emplace_back(std::chrono::high_resolution_clock::now());
                    std::string originStream = getOriginStream(currReq.req_travelPath[imageIndexInBatch]);
                    // TODO: Add the request number
                    msvc_processRecords.addRecord(currReq.req_origGenTime[imageIndexInBatch],
                                                  currReq_batchSize,
                                                  totalInMem[j], totalOutMem[j], totalEncodedOutMem[j], 0, originStream);
                    msvc_arrivalRecords.addRecord(
                            currReq.req_origGenTime[i],
                            10,
                            getArrivalPkgSize(currReq.req_travelPath[imageIndexInBatch]),
                            totalInMem[j],
                            msvc_overallTotalReqCount,
                            originStream,
                            getSenderHost(currReq.req_travelPath[imageIndexInBatch])
                    );
                    if (timeNow > currReq.req_origGenTime[imageIndexInBatch][3]) {
                        addToLatencyEWMA(
                                std::chrono::duration_cast<TimePrecisionType>(
                                        timeNow - currReq.req_origGenTime[imageIndexInBatch][3]).count());
                    }
                }
            }
        }

        
        msvc_batchCount++;
        msvc_miniBatchCount++;

        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));

    }
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
    STOPPED = true;
}

void BaseKPointExtractor::extractorProfiling() {
}