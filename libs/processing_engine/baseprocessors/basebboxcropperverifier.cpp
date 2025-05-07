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
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;
    // Path
    std::string currReq_path;

    // List of images carried from the previous microservice here to be cropped from.
    std::vector<RequestData<LocalGPUReqDataType>> imageList;
    // Instance of data to be packed into the out req
    RequestData<LocalGPUReqDataType> reqData;
    RequestData<LocalCPUReqDataType> reqDataCPU;

    std::vector<RequestData<LocalGPUReqDataType>> currReq_data;

    // List of bounding boxes cropped from one single image
    std::vector<BoundingBox<cv::cuda::GpuMat>> singleImageBBoxList;

    // Current incoming equest
    Request<LocalGPUReqDataType> currReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestDataShapeType bboxShape;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;
    cv::cuda::Stream *postProcCVStream;

    // Height and width of the image used for inference
    // int orig_h, orig_w, infer_h, infer_w;

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

    uint16_t maxNumDets = 0;
    
    int32_t *num_detections = nullptr;
    float *nmsed_boxes = nullptr;
    float *nmsed_scores = nullptr;
    float *nmsed_classes = nullptr;

    std::vector<float *> ptrList;

    size_t bufferSize;

    // class of the bounding box cropped from one the images in the image list
    int16_t bboxClass = 0;
    // The index of the queue we are going to put data on based on the value of `bboxClass`
    std::vector<NumQueuesType> queueIndex;

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
                READY = false;
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
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
                postProcCVStream = new cv::cuda::Stream();


                msvc_inferenceShape = upstreamMicroserviceList.at(0).expectedShape;

                concatConfigsGenerator(msvc_inferenceShape, msvc_concat, 2);

                // infer_h = msvc_inferenceShape[0][1];
                // infer_w = msvc_inferenceShape[0][2];

                maxNumDets = msvc_dataShape[2][0];

                delete num_detections;
                if (nmsed_boxes) delete nmsed_boxes;
                if (nmsed_scores) delete nmsed_scores;
                if (nmsed_classes) delete nmsed_classes;

                BatchSizeType batchSize;
                if (msvc_allocationMode == AllocationMode::Conservative) {
                    batchSize = msvc_idealBatchSize;
                } else if (msvc_allocationMode == AllocationMode::Aggressive) {
                    batchSize = msvc_maxBatchSize;
                }
                num_detections = new int32_t[batchSize];
                nmsed_boxes = new float[batchSize * maxNumDets * 4];
                nmsed_scores = new float[batchSize * maxNumDets];
                nmsed_classes = new float[batchSize * maxNumDets];


                ptrList = {nmsed_boxes, nmsed_scores, nmsed_classes};

                singleImageBBoxList.clear();

                RELOADING = false;
                READY = true;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2(msvc_name);
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


        auto timeNow = std::chrono::high_resolution_clock::now();
        // 10. The moment the batch is received at the cropper (TENTH_TIMESTAMP)
        for (auto& req_genTime : currReq.req_origGenTime) {
            req_genTime.emplace_back(timeNow);
        }

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
            auto numImagesInFrame = currReq.req_concatInfo[i].numImagesAdded;
            std::vector<MemUsageType> totalInMem(numImagesInFrame, 0), totalOutMem(numImagesInFrame, 0), totalEncodedOutMem(numImagesInFrame, 0);
            msvc_overallTotalReqCount++;

            // 11. The moment the request starts being processed by the cropper, after the batch was unloaded (ELEVENTH_TIMESTAMP)
            for (uint8_t concatInd = 0; concatInd < numImagesInFrame; concatInd++) {
                uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + concatInd;
                currReq.req_origGenTime[imageIndexInBatch].emplace_back(std::chrono::high_resolution_clock::now());
            }

            // Otherwise, we need to do some cropping.
            //orig_h = imageList[i].shape[1];
            //orig_w = imageList[i].shape[2];

            // crop(imageList[i].data, orig_h, orig_w, infer_h, infer_w, numDetsInFrame, nmsed_boxes[i][0], singleImageBBoxList);
            // spdlog::get("container_agent")->trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            for (uint8_t j = 0; j < numImagesInFrame; j++) {
                uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + j;
                totalInMem[j] = (imageList[imageIndexInBatch].data.channels() *
                                 imageList[imageIndexInBatch].data.rows * imageList[imageIndexInBatch].data.cols *
                                 CV_ELEM_SIZE1(imageList[imageIndexInBatch].data.type()));
                totalOutMem[j] = totalInMem.back();

                currReq_path = currReq.req_travelPath[imageIndexInBatch] + "|1|1";


                if (msvc_activeOutQueueIndex.at(0) == 1) { //Local CPU
                    cv::Mat out;
                    imageList[imageIndexInBatch].data.download(out, *postProcCVStream);
                    postProcCVStream->waitForCompletion();
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
                                {imageList[imageIndexInBatch].shape, out}
                            } //req_data
                        }
                    );
                    spdlog::get("container_agent")->trace("{0:s} emplaced an image to CPU queue {2:d}.", msvc_name, bboxClass, 0);
                } else {
                    currReq_path += "|0|" + std::to_string(totalInMem[j]) + "]";
                    msvc_OutQueue.at(0)->emplace(
                        Request<LocalGPUReqDataType>{
                            {{currReq.req_origGenTime[imageIndexInBatch].front(), std::chrono::high_resolution_clock::now()}},
                            {currReq.req_e2eSLOLatency[imageIndexInBatch]},
                            {currReq_path},
                            1,
                            {imageList[imageIndexInBatch]}, //req_data
                        }
                    );
                    spdlog::get("container_agent")->trace("{0:s} emplaced an image to GPU queue {2:d}.", msvc_name, bboxClass, 0);
                }

                // If the number of warmup batches has been passed, we start to record the latency
                if (warmupCompleted()) {
                    // 12. The moment the request was completed by the postprocessor (TWELFTH_TIMESTAMP)
                    currReq.req_origGenTime[imageIndexInBatch].emplace_back(std::chrono::high_resolution_clock::now());
                    std::string originStream = getOriginStream(currReq.req_travelPath[imageIndexInBatch]);
                    // TODO: Add the request number
                    msvc_processRecords.addRecord(currReq.req_origGenTime[imageIndexInBatch],
                                                  currReq_batchSize,
                                                  totalInMem[j], totalOutMem[j], totalEncodedOutMem[j], 0, originStream);
                    msvc_arrivalRecords.addRecord(
                            currReq.req_origGenTime[imageIndexInBatch],
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

            // Clearing out data of the vector
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
    if (postProcCVStream) {
        delete postProcCVStream;
        postProcCVStream = nullptr; // Avoid dangling pointer
    }
    msvc_logFile.close();
    STOPPED = true;
}

void BaseBBoxCropperVerifier::cropProfiling() {
}