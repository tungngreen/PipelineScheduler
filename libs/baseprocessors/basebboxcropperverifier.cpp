#include "baseprocessor.h"

using namespace spdlog;

BaseBBoxCropperVerifierConfigs BaseBBoxCropperVerifier::loadConfigsFromJson(const json &jsonConfigs) {
    BaseBBoxCropperVerifierConfigs configs;
    return configs;
}

void BaseBBoxCropperVerifier::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOANDING configs...", __func__);
    if (!isConstructing) { // If the microservice is being reloaded
        BasePostprocessor::loadConfigs(jsonConfigs, isConstructing);
    }
    BaseBBoxCropperVerifierConfigs configs = loadConfigsFromJson(jsonConfigs);
    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);
}

BaseBBoxCropperVerifier::BaseBBoxCropperVerifier(const json &jsonConfigs) : BasePostprocessor(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name); 
}

void BaseBBoxCropperVerifier::cropping() {
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
    cv::cuda::Stream postProcCVStream;

    // Height and width of the image used for inference
    int orig_h, orig_w;

    /**
     * @brief Each request to the cropping microservice of YOLOv5 contains the buffers which are results of TRT inference 
     * as well as the images from which bounding boxes will be cropped.
     * `The buffers` are a vector of 4 small raw memory (float) buffers (since using `BatchedNMSPlugin` provided by TRT), which are
     * 1/ `num_detections` (Batch, 1): number of objects detected in this frame
     * 2/ `nmsed_boxes` (Batch, TopK, 4): the boxes remaining after nms is performed.
     * 3/ `nmsed_scores` (Batch, TopK): the scores of these boxes.
     * 4/ `nmsed_classes` (Batch, TopK):
     * We need to bring these buffers to CPU in order to process them.
     */

    uint16_t maxNumDets;
    
    int32_t *num_detections = nullptr;
    float *nmsed_boxes = nullptr;
    float *nmsed_scores = nullptr;
    float *nmsed_classes = nullptr;

    std::vector<float *> ptrList;

    size_t bufferSize;

    // class of the bounding box cropped from one the images in the image list
    int16_t bboxClass;
    // The index of the queue we are going to put data on based on the value of `bboxClass`
    NumQueuesType queueIndex = 0;

    // To whole the shape of data sent from the inferencer
    RequestDataShapeType shape;

    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (PAUSE_THREADS) {
            if (RELOADING){
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

                setDevice();
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);
                postProcCVStream = cv::cuda::StreamAccessor::wrapStream(postProcStream);
                
                maxNumDets = msvc_dataShape[2][0];

                delete num_detections;
                delete nmsed_boxes;
                delete nmsed_scores;
                delete nmsed_classes;

                BatchSizeType batchSize = msvc_allocationMode == AllocationMode::Conservative ? msvc_idealBatchSize : msvc_maxBatchSize;
                num_detections = new int32_t[batchSize];
                nmsed_boxes = new float[batchSize * maxNumDets * 4];
                nmsed_scores = new float[batchSize * maxNumDets];
                nmsed_classes = new float[batchSize * maxNumDets];

                ptrList = {nmsed_boxes, nmsed_scores, nmsed_classes};

                outReqData.clear();
                singleImageBBoxList.clear();

                RELOADING = false;
                READY = true;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            //info("{0:s} is being PAUSED.", msvc_name);
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

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        // if (msvc_inReqCount > 1) {
        //     updateReqRate(currReq_genTime);
        // }
        currReq_batchSize = currReq.req_batchSize;
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        currReq_data = currReq.req_data;

        for (std::size_t i = 0; i < (currReq_data.size() - 1); ++i) {
            bufferSize = msvc_modelDataType * (size_t)currReq_batchSize;
            RequestDataShapeType shape = currReq_data[i].shape;
            for (uint8_t j = 0; j < shape.size(); ++j) {
                bufferSize *= shape[j];
            }
            if (i == 0) {
                checkCudaErrorCode(cudaMemcpyAsync(
                    (void *) num_detections,
                    currReq_data[i].data.cudaPtr(),
                    bufferSize,
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
            } else {
                checkCudaErrorCode(cudaMemcpyAsync(
                    (void *) ptrList[i - 1],
                    currReq_data[i].data.cudaPtr(),
                    bufferSize,
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
            }
        }

        checkCudaErrorCode(cudaStreamSynchronize(postProcStream), __func__);
        spdlog::get("container_agent")->trace("{0:s} unloaded 4 buffers to CPU {1:d}", msvc_name, currReq_batchSize);

        // List of images to be cropped from
        imageList = currReq.upstreamReq_data; 

        // Doing post processing for the whole batch
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {
            msvc_overallTotalReqCount++;

            // We consider this when the request was received by the postprocessor
            currReq.req_origGenTime[i].emplace_back(std::chrono::high_resolution_clock::now());

            currReq_path = currReq.req_travelPath[i] + "|1|1";

            // Otherwise, we need to do some cropping.
            orig_h = imageList[i].shape[1];
            orig_w = imageList[i].shape[2];

            // crop(imageList[i].data, orig_h, orig_w, infer_h, infer_w, numDetsInFrame, nmsed_boxes[i][0], singleImageBBoxList);
            // spdlog::get("container_agent")->trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            uint32_t totalInMem = imageList[i].data.channels() * imageList[i].data.rows * imageList[i].data.cols * CV_ELEM_SIZE1(imageList[i].data.type());
            uint32_t totalOutMem = totalInMem, totalEncodedOutMem = 0;

            if (msvc_activeOutQueueIndex.at(queueIndex) == 1) { //Local CPU
                cv::Mat out;
                imageList[i].data.download(out, postProcCVStream);
                postProcCVStream.waitForCompletion();
                if (msvc_OutQueue.at(queueIndex)->getEncoded()) {
                    out = encodeResults(out);
                    totalEncodedOutMem = out.channels() * out.rows * out.cols * CV_ELEM_SIZE1(out.type());
                }

                currReq_path += "|" + std::to_string(totalEncodedOutMem) + "|" + std::to_string(totalInMem) + "]";

                msvc_OutQueue.at(0)->emplace(
                    Request<LocalCPUReqDataType>{
                        {{currReq.req_origGenTime[i].front(), std::chrono::high_resolution_clock::now()}},
                        {currReq.req_e2eSLOLatency[i]},
                        {currReq_path},
                        1,
                        {
                            {imageList[i].shape, out}
                        } //req_data
                    }
                );
                spdlog::get("container_agent")->trace("{0:s} emplaced an image to CPU queue {2:d}.", msvc_name, bboxClass, queueIndex);
            } else {
                currReq_path += "|0|" + std::to_string(totalInMem) + "]";
                msvc_OutQueue.at(0)->emplace(
                    Request<LocalGPUReqDataType>{
                        {{currReq.req_origGenTime[i].front(), std::chrono::high_resolution_clock::now()}},
                        {currReq.req_e2eSLOLatency[i]},
                        {currReq_path},
                        1,
                        {imageList[i]}, //req_data
                    }
                );
                spdlog::get("container_agent")->trace("{0:s} emplaced an image to GPU queue {2:d}.", msvc_name, bboxClass, queueIndex);
            }


            /**
             * @brief There are 8 important timestamps to be recorded:
             * 1. When the request was generated
             * 2. When the request was received by the preprocessor
             * 3. When the request was done preprocessing by the preprocessor
             * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
             * 5. When the batch inferencer popped the batch sent from batcher
             * 6. When the batch inference was completed by the inferencer 
             * 7. When the request was received by the postprocessor
             * 8. When each request was completed by the postprocessor
             */

            // If the number of warmup batches has been passed, we start to record the latency
            if (warmupCompleted()) {
                currReq.req_origGenTime[i].emplace_back(std::chrono::high_resolution_clock::now());
                std::string originStream = getOriginStream(currReq.req_travelPath[i]);
                // TODO: Add the request number
                msvc_processRecords.addRecord(currReq.req_origGenTime[i], currReq_batchSize, totalInMem, totalOutMem, totalEncodedOutMem, 0, originStream);
                msvc_arrivalRecords.addRecord(
                        currReq.req_origGenTime[i],
                        10,
                        getArrivalPkgSize(currReq.req_travelPath[i]),
                        totalInMem,
                        msvc_overallTotalReqCount,
                        originStream,
                        getSenderHost(currReq.req_travelPath[i])
                );
                addToLatencyEWMA(
                        std::chrono::duration_cast<TimePrecisionType>(currReq_recvTime - currReq.req_origGenTime[i][3]).count());
            }

            outReqData.clear();
            singleImageBBoxList.clear();
        }
        // // Free all the output buffers of trtengine after cropping is done.
        // for (size_t i = 0; i < currReq_data.size(); i++) {
        //     checkCudaErrorCode(cudaFree(currReq_data.at(i).data.cudaPtr()));
        // }

        msvc_batchCount++;
        msvc_miniBatchCount++;

        
        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
        // Synchronize the cuda stream
    }


    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
    STOPPED = true;
}

void BaseBBoxCropperVerifier::cropProfiling() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

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
    Request<LocalGPUReqDataType> inferTimeReportReq, outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestDataShapeType bboxShape;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    /**
     * @brief Each request to the cropping microservice of YOLOv5 contains the buffers which are results of TRT inference 
     * as well as the images from which bounding boxes will be cropped.
     * `The buffers` are a vector of 4 small raw memory (float) buffers (since using `BatchedNMSPlugin` provided by TRT), which are
     * 1/ `num_detections` (Batch, 1): number of objects detected in this frame
     * 2/ `nmsed_boxes` (Batch, TopK, 4): the boxes remaining after nms is performed.
     * 3/ `nmsed_scores` (Batch, TopK): the scores of these boxes.
     * 4/ `nmsed_classes` (Batch, TopK):
     * We need to bring these buffers to CPU in order to process them.
     */

    uint16_t maxNumDets;
    
    int32_t *num_detections = nullptr;
    float *nmsed_boxes = nullptr;
    float *nmsed_scores = nullptr;
    float *nmsed_classes = nullptr;

    std::vector<float *> ptrList;

    size_t bufferSize;

    // class of the bounding box cropped from one the images in the image list
    int16_t bboxClass;
    // The index of the queue we are going to put data on based on the value of `bboxClass`
    NumQueuesType queueIndex;

    // To whole the shape of data sent from the inferencer
    RequestDataShapeType shape;

    //
    auto time_now = std::chrono::high_resolution_clock::now();

    // Containing the current timestamps to be sent back to receiver to calculate the end-to-end latency.
    std::vector<RequestData<LocalCPUReqDataType>> inferTimeReportData;

    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (PAUSE_THREADS) {
            if (RELOADING){
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

                setDevice();
                checkCudaErrorCode(cudaStreamCreate(&postProcStream), __func__);
                
                maxNumDets = msvc_dataShape[2][0];

                delete num_detections;
                delete nmsed_boxes;
                delete nmsed_scores;
                delete nmsed_classes;

                num_detections = new int32_t[msvc_idealBatchSize];
                nmsed_boxes = new float[msvc_idealBatchSize * maxNumDets * 4];
                nmsed_scores = new float[msvc_idealBatchSize * maxNumDets];
                nmsed_classes = new float[msvc_idealBatchSize * maxNumDets];

                ptrList = {nmsed_boxes, nmsed_scores, nmsed_classes};

                outReqData.clear();
                singleImageBBoxList.clear();

                RELOADING = false;
                READY = true;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
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

        msvc_overallTotalReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (msvc_overallTotalReqCount > 1) {
            updateReqRate(currReq_genTime);
        }
        currReq_batchSize = inferTimeReportReq.req_batchSize;
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}", msvc_name, currReq_batchSize);

        currReq_data = inferTimeReportReq.req_data;

        for (std::size_t i = 0; i < (currReq_data.size() - 1); ++i) {
            bufferSize = msvc_modelDataType * (size_t)currReq_batchSize;
            RequestDataShapeType shape = currReq_data[i].shape;
            for (uint8_t j = 0; j < shape.size(); ++j) {
                bufferSize *= shape[j];
            }
            if (i == 0) {
                checkCudaErrorCode(cudaMemcpyAsync(
                    (void *) num_detections,
                    currReq_data[i].data.cudaPtr(),
                    bufferSize,
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
            } else {
                checkCudaErrorCode(cudaMemcpyAsync(
                    (void *) ptrList[i - 1],
                    currReq_data[i].data.cudaPtr(),
                    bufferSize,
                    cudaMemcpyDeviceToHost,
                    postProcStream
                ), __func__);
            }
        }

        checkCudaErrorCode(cudaStreamSynchronize(postProcStream), __func__);
        spdlog::get("container_agent")->trace("{0:s} unloaded 4 buffers to CPU {1:d}", msvc_name, currReq_batchSize);

        // List of images to be cropped from
        imageList = inferTimeReportReq.upstreamReq_data; 

        // Doing post processing for the whole batch
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {

            currReq_path = inferTimeReportReq.req_travelPath[i];

            // If there is no object in frame, we don't have to do nothing.
            int numDetsInFrame = (int)num_detections[i];

            spdlog::get("container_agent")->trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            /**
             * @brief During profiling mode, there are six important timestamps to be recorded:
             * 1. When the request was generated
             * 2. When the request was received by the preprocessor
             * 3. When the request was done preprocessing by the preprocessor
             * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
             * 5. When the batch inferencer was completed by the inferencer 
             * 6. When each request was completed by the postprocessor
             */

            time_now = std::chrono::high_resolution_clock::now();
            inferTimeReportReq.req_origGenTime[i].emplace_back(time_now);

            spdlog::get("container_agent")->trace("{0:s} emplaced an image to queue {2:d}.", msvc_name, bboxClass, queueIndex);
            // // After cropping is done for this image in the batch, the image's cuda memory can be freed.
            // checkCudaErrorCode(cudaFree(imageList[i].data.cudaPtr()));
            // Clearing out data of the vector

            outReqData.clear();
            singleImageBBoxList.clear();
        }
        // // Free all the output buffers of trtengine after cropping is done.
        // for (size_t i = 0; i < currReq_data.size(); i++) {
        //     checkCudaErrorCode(cudaFree(currReq_data.at(i).data.cudaPtr()));
        // }

        msvc_OutQueue[0]->emplace(
            Request<LocalCPUReqDataType>({
                inferTimeReportReq.req_origGenTime,
                {},
                inferTimeReportReq.req_travelPath,
                inferTimeReportReq.req_batchSize,
                {}, //req_data
            })
        );
        inferTimeReportData.clear();
        
        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
    }


    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
    STOPPED = true;
}