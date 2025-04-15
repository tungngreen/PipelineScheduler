#include "baseprocessor.h"

using namespace spdlog;


/**
 * @brief Scale the bounding box coordinates to the original aspect ratio of the image
 * 
 * @param orig_h Original height of the image
 * @param orig_w Original width of the image
 * @param infer_h Height of the image used for inference
 * @param infer_w Width of the image used for inference
 * @param bbox_coors [x1, y1, x2, y2]
 */
inline void scaleBBox(
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    const float *infer_bboxCoors,
    int * orig_bboxCoors
) {
    float ratio = std::min(1.f * infer_h / orig_h, 1.f * infer_w / orig_w);
    infer_h = (int) (ratio * orig_h);
    infer_w = (int) (ratio * orig_w);

    // TO BE IMPLEMENTED
    float coor[4];
    for (uint8_t i = 0; i < 4; ++i) {
        coor[i] = (*(infer_bboxCoors + i));
    }

    float gain = std::min(1.f * infer_h / orig_h, 1.f * infer_w / orig_w);

    float pad_w = (1.f * infer_w - orig_w * gain) / 2.f;
    float pad_h = (1.f * infer_h - orig_h * gain) / 2.f;

    coor[0] -= pad_w;
    coor[1] -= pad_h;
    coor[2] -= pad_w;
    coor[3] -= pad_h;

    // if (scale_h > scale_w) {
    //     coor[1]= coor[1] / scale_w;
    //     coor[3]= coor[3] / scale_w;
    //     coor[0]= (coor[0] - (infer_h - scale_w * orig_h) / 2) / scale_w;
    //     coor[2]= (coor[2] - (infer_h - scale_w * orig_h) / 2) / scale_w;
    // } else {
    //     coor[1]= (coor[1] - (infer_w - scale_h * orig_w) / 2) / scale_h;
    //     coor[3]= (coor[3] - (infer_w - scale_h * orig_w) / 2) / scale_h;
    //     coor[0]= coor[0] / scale_h;
    //     coor[2]= coor[2] / scale_h;
    // }

    for (uint8_t i = 0; i < 4; ++i) {
        coor[i] /= gain;
        int maxcoor = (i % 2 == 0) ? orig_w : orig_h;
        if (coor[i] >= maxcoor) {
            coor[i] = maxcoor - 1;
        }
        if (coor[i] < 0) {
            coor[i] = 0;
        }
        *(orig_bboxCoors + i) = (int)coor[i];
    }
}

/**
 * @brief Cropping multiple boxes from 1 picture
 * 
 * @param image 
 * @param infer_h 
 * @param infer_w 
 * @param numDetections 
 * @param bbox_coorList 
 * @param croppedBBoxes 
 */
inline void crop(
    const cv::cuda::GpuMat &image,
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
) {
    int orig_bboxCoors[4];
    for (uint16_t i = 0; i < numDetections; ++i) {
        scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList + i * 4, orig_bboxCoors);
        // std::cout << (int)orig_bboxCoors[0] << " " << (int)orig_bboxCoors[1] << " " << (int)orig_bboxCoors[2] << " " << (int)orig_bboxCoors[3] << std::endl;
        cv::cuda::GpuMat croppedBBox = image(cv::Range((int)orig_bboxCoors[1], (int)orig_bboxCoors[3]), cv::Range((int)orig_bboxCoors[0], (int)orig_bboxCoors[2])).clone();
        croppedBBoxes.emplace_back(croppedBBox);
    }
}

/**
 * @brief Cropping 1 box from 1 picture
 * 
 * @param image 
 * @param infer_h 
 * @param infer_w 
 * @param numDetections 
 * @param bbox_coorList 
 * @param croppedBBoxes 
 */
inline void cropOneBox(
    const cv::cuda::GpuMat &image,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    cv::cuda::GpuMat &croppedBBoxes
) {
    int orig_h, orig_w;
    orig_h = image.rows;
    orig_w = image.cols;
    int orig_bboxCoors[4];
    scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList, orig_bboxCoors);
    cv::cuda::GpuMat croppedBBox = image(cv::Range((int)orig_bboxCoors[0], (int)orig_bboxCoors[2]), cv::Range((int)orig_bboxCoors[1], (int)orig_bboxCoors[3])).clone();
    croppedBBoxes = croppedBBox;
}

void BaseBBoxCropperAugmentation::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOANDING configs...", __func__);
    if (!isConstructing) { // If this is not called from the constructor
        BasePostprocessor::loadConfigs(jsonConfigs, isConstructing);
    }
    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);
}

BaseBBoxCropperAugmentation::BaseBBoxCropperAugmentation(const json &jsonConfigs) : BasePostprocessor(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name); 
}

void BaseBBoxCropperAugmentation::cropping() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
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
    std::vector<cv::cuda::GpuMat> singleImageBBoxList;

    // Current incoming request
    Request<LocalGPUReqDataType> currReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestDataShapeType bboxShape;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name); 


    cudaStream_t postProcStream;

    // Height and width of the image used for inference
    int orig_h, orig_w, infer_h = 0, infer_w = 0;

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
    int16_t bboxClass;
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

                msvc_inferenceShape = upstreamMicroserviceList.at(0).expectedShape;

                infer_h = msvc_inferenceShape[0][1];
                infer_w = msvc_inferenceShape[0][2];
                
                maxNumDets = msvc_dataShape[2][0];

                delete num_detections;
                if (nmsed_boxes) delete nmsed_boxes;
                if (nmsed_scores) delete nmsed_scores;
                if (nmsed_classes) delete nmsed_classes;

                BatchSizeType batchSize = (msvc_allocationMode == AllocationMode::Conservative) ? msvc_idealBatchSize : msvc_maxBatchSize;
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

            // There could be multiple timestamps in the request, but the first one always represent
            // the moment this request was generated at the very beginning of the pipeline
            currReq_genTime = currReq.req_origGenTime[i][0];
            currReq_path = currReq.req_travelPath[i];

            // First we need to set the infer_h,w and the original h,w of the image.
            // infer_h,w are given in the last dimension of the request data from the inferencer
            infer_h = currReq.req_data.back().shape[1];
            infer_w = currReq.req_data.back().shape[2];
            // orig_h,w are given in the shape of the image in the image list, which is carried from the batcher
            orig_h = imageList[i].shape[1];
            orig_w = imageList[i].shape[2];

            // If there is no object in frame, we decide if we add a random image or not.
            int numDetsInFrame = (int)num_detections[i];
            if (numDetsInFrame <= 0) {

                // Generate a random box for downstream wrorkload
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, 1);

                if (dis(gen) == 0) {
                    continue;
                }

                numDetsInFrame = 1;
                singleImageBBoxList.emplace_back(
                    cv::cuda::GpuMat(64, 64, CV_8UC3)
                );
                nmsed_classes[i * maxNumDets] = 1;
            } else {
                crop(imageList[i].data, orig_h, orig_w, infer_h, infer_w, numDetsInFrame, nmsed_boxes + i * maxNumDets * 4, singleImageBBoxList);
                spdlog::get("container_agent")->info("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);
            }

            uint32_t totalInMem = imageList[i].data.channels() * imageList[i].data.rows * imageList[i].data.cols * CV_ELEM_SIZE1(imageList[i].data.type());
            uint32_t totalOutMem = 0, totalEncodedOutMem = 0;

            std::vector<std::vector<PerQueueOutRequest>> outReqList(msvc_OutQueue.size(), std::vector<PerQueueOutRequest>());

            // After cropping, we need to find the right queues to put the bounding boxes in
            float processedCounter = 0;
            for (int j = 0; j < numDetsInFrame; ++j) {
                bboxClass = (int16_t)nmsed_classes[i * maxNumDets + j];

                cv::Mat cpuBox;
                cv::Mat encodedBox;
                uint32_t boxEncodedMemSize = 0;
                // in the constructor of each microservice, we map the class number to the corresponding queue index in 
                // `classToDntreamMap`.
                for (size_t k = 0; k < this->classToDnstreamMap.size(); ++k) {
                    NumQueuesType qIndex = MAX_NUM_QUEUES;
                    if ((classToDnstreamMap.at(k).first == bboxClass) || (classToDnstreamMap.at(k).first == -1)) {
                        qIndex = this->classToDnstreamMap.at(k).second;
                        queueIndex.emplace_back(qIndex);
                    }
                    if (qIndex == MAX_NUM_QUEUES) {
                        continue;
                    }
                    if (msvc_activeOutQueueIndex.at(qIndex) != 1) { //If not supposed to be sent as CPU serialized data
                        continue;
                    }
                    if (cpuBox.empty()) {
                        // cv::Mat box(singleImageBBoxList[j].size(), CV_8UC3);
                        // checkCudaErrorCode(cudaMemcpyAsync(
                        //     box.data,
                        //     singleImageBBoxList[j].cudaPtr(),
                        //     singleImageBBoxList[j].cols * singleImageBBoxList[j].rows * singleImageBBoxList[j].channels() * CV_ELEM_SIZE1(singleImageBBoxList[j].type()),
                        //     cudaMemcpyDeviceToHost,
                        //     postProcStream
                        // ), __func__);
                        // std::cout << singleImageBBoxList[j].type() << std::endl;
                        // // Synchronize the cuda stream right away to avoid any race condition
                        // checkCudaErrorCode(cudaStreamSynchronize(postProcStream), __func__);
                        cv::cuda::Stream cvStream = cv::cuda::Stream();
                        singleImageBBoxList[j].download(cpuBox, cvStream);
                        cvStream.waitForCompletion();
                    }
                    if (msvc_OutQueue.at(qIndex)->getEncoded() && encodedBox.empty() && !cpuBox.empty()) {
                        encodedBox = encodeResults(cpuBox);
                        boxEncodedMemSize = encodedBox.cols * encodedBox.rows * encodedBox.channels() * CV_ELEM_SIZE1(encodedBox.type());
                    }
                }
                // If this class number is not needed anywhere downstream
                if (queueIndex.empty()) {
                    continue;
                }

                // if (bboxClass == 0 || bboxClass == 2) {
                //     saveGPUAsImg(singleImageBBoxList[j], "bbox_" + std::to_string(j) + ".jpg");
                // }

                // Putting the bounding box into an `outReq` to be sent out
                bboxShape = {singleImageBBoxList[j].channels(), singleImageBBoxList[j].rows, singleImageBBoxList[j].cols};

                for (auto qIndex : queueIndex) {
                    std::string path = currReq_path;
                    path += "|" + std::to_string(numDetsInFrame) + "|" + std::to_string(j);
                    if (dnstreamMicroserviceList[bboxClass].portions.size() > 0 &&
                        (processedCounter++ / numDetsInFrame)
                            > dnstreamMicroserviceList[bboxClass].portions[outReqList.at(qIndex).size()-1]) {
                        outReqList.at(qIndex).emplace_back();
                        processedCounter = 0;
                    }
                    // Put the correct type of outreq for the downstream, a sender, which expects either LocalGPU or localCPU
                    if (msvc_activeOutQueueIndex.at(qIndex) == 1) { //Local CPU
                        if (msvc_OutQueue.at(qIndex)->getEncoded()) {
                            reqDataCPU = {
                                bboxShape,
                                encodedBox.clone()
                            };
                        } else {
                            reqDataCPU = {
                                bboxShape,
                                cpuBox.clone()
                            };
                        }

                        outReqList.at(qIndex).back().cpuReq.req_origGenTime.emplace_back(RequestTimeType{currReq.req_origGenTime[i].front()});
                        outReqList.at(qIndex).back().cpuReq.req_e2eSLOLatency.emplace_back(currReq.req_e2eSLOLatency[i]);
                        outReqList.at(qIndex).back().cpuReq.req_travelPath.emplace_back(path);
                        outReqList.at(qIndex).back().cpuReq.req_data.emplace_back(reqDataCPU);
                        outReqList.at(qIndex).back().cpuReq.req_batchSize = 1;

                        spdlog::get("container_agent")->trace("{0:s} emplaced a bbox of class {1:d} to CPU queue {2:d}.", msvc_name, bboxClass, qIndex);

                    } else {
                        cv::cuda::GpuMat out(singleImageBBoxList[j].size(), singleImageBBoxList[j].type());
                        checkCudaErrorCode(cudaMemcpyAsync(
                            out.cudaPtr(),
                            singleImageBBoxList[j].cudaPtr(),
                            singleImageBBoxList[j].cols * singleImageBBoxList[j].rows * singleImageBBoxList[j].channels() * CV_ELEM_SIZE1(singleImageBBoxList[j].type()),
                            cudaMemcpyDeviceToDevice,
                            postProcStream
                        ), __func__);

                        reqData = {
                            bboxShape,
                            out
                        };
                        outReqList.at(qIndex).back().gpuReq.req_origGenTime.emplace_back(RequestTimeType{currReq.req_origGenTime[i].front()});
                        outReqList.at(qIndex).back().gpuReq.req_e2eSLOLatency.emplace_back(currReq.req_e2eSLOLatency[i]);
                        outReqList.at(qIndex).back().gpuReq.req_travelPath.emplace_back(path);
                        outReqList.at(qIndex).back().gpuReq.req_data.emplace_back(reqData);
                        outReqList.at(qIndex).back().cpuReq.req_batchSize = 1;

                        spdlog::get("container_agent")->trace("{0:s} emplaced a bbox of class {1:d} to GPU queue {2:d}.", msvc_name, bboxClass, qIndex);
                    }
                    uint32_t imageMemSize = singleImageBBoxList[j].cols * singleImageBBoxList[j].rows * singleImageBBoxList[j].channels() * CV_ELEM_SIZE1(singleImageBBoxList[j].type());
                    outReqList.at(qIndex).back().totalSize += imageMemSize;
                    outReqList.at(qIndex).back().totalEncodedSize += boxEncodedMemSize;
                    totalOutMem += imageMemSize;
                    totalEncodedOutMem += boxEncodedMemSize;
                }
                queueIndex.clear();
            }

            NumQueuesType qIndex = 0;
            for (auto &outList : outReqList) {
                for (auto &outReq : outList) {
                    if (msvc_activeOutQueueIndex.at(qIndex) == 1) { //Local CPU GPU
                        // Add the total size of bounding boxes heading to this queue
                        for (auto &path: outReq.cpuReq.req_travelPath) {
                            path += "|" + std::to_string(outReq.totalEncodedSize) + "|" +
                                    std::to_string(outReq.totalSize) + "]";
                        }
                        // Make sure the time is uniform across all the bounding boxes
                        for (auto &time: outReq.cpuReq.req_origGenTime) {
                            time.emplace_back(std::chrono::high_resolution_clock::now());
                        }
                        msvc_OutQueue.at(qIndex)->emplace(outReq.cpuReq);
                    } else { //Local GPU Queue
                        // Add the total size of bounding boxes heading to this queue
                        for (auto &path: outReq.gpuReq.req_travelPath) {
                            path += "|" + std::to_string(outReq.totalEncodedSize) + "|" +
                                    std::to_string(outReq.totalSize) + "]";
                        }
                        // Make sure the time is uniform across all the bounding boxes
                        for (auto &time: outReq.gpuReq.req_origGenTime) {
                            time.emplace_back(std::chrono::high_resolution_clock::now());
                        }
                        msvc_OutQueue.at(qIndex)->emplace(outReq.gpuReq);
                    }
                }
                qIndex++;
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
                if (currReq_recvTime > currReq.req_origGenTime[i][3]) {
                    addToLatencyEWMA(
                            std::chrono::duration_cast<TimePrecisionType>(
                                    currReq_recvTime - currReq.req_origGenTime[i][3]).count());
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
    msvc_logFile.close();
    STOPPED = true;
}


/**
 * @brief Generate random bboxes
 * 
 * @param bboxList 
 */
void BaseBBoxCropperAugmentation::generateRandomBBox(
    float *bboxList,
    const uint16_t height,
    const uint16_t width,
    const uint16_t numBboxes,
    const uint16_t seed
) {
    float x1, y1, x2, y2;

    std::mt19937 gen(seed);

    for (uint16_t j = 0; j < numBboxes; j++) {
        do {
            std::uniform_real_distribution<> x1_dis(0, width - 1);
            std::uniform_real_distribution<> y1_dis(0, height - 1);
            std::uniform_real_distribution<> width_dis(1, width / 2);
            std::uniform_real_distribution<> height_dis(1, height / 2);

            x1 = x1_dis(gen);
            y1 = y1_dis(gen);
            float width = width_dis(gen);
            float height = height_dis(gen);

            x2 = x1 + width;
            y2 = y1 + height;
        } while (x2 >= width || y2 >= height);
        *(bboxList + (j * 4) + 0) = x1;
        *(bboxList + (j * 4) + 1) = y1;
        *(bboxList + (j * 4) + 2) = x2;
        *(bboxList + (j * 4) + 3) = y2;
    }
}

void BaseBBoxCropperAugmentation::cropProfiling() {
}