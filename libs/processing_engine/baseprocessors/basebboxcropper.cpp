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
inline std::vector<std::pair<uint8_t, uint16_t>> crop(
    const std::vector<cv::cuda::GpuMat> &images,
    const std::vector<ConcatConfig> &allConcatConfigs,
    const RequestConcatInfo &reqsConcatInfo,
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    uint16_t numDetections,
    const float *bbox_coorList,
    const float *nmsed_scores,
    const float confidenceThreshold,
    std::vector<BoundingBox<cv::cuda::GpuMat>> &croppedBBoxes
) {
    std::vector<std::pair<uint8_t, uint16_t>> imageIndexList = {};
    const ConcatConfig& concatDims = allConcatConfigs[reqsConcatInfo.totalNumImages];
    int orig_bboxCoors[4];
    uint16_t numInvalidDets = 0;
    for (uint16_t i = 0; i < numDetections; ++i) {
        int infer_x1 = bbox_coorList[i * 4];
        int infer_y1 = bbox_coorList[i * 4 + 1];
        int infer_x2 = bbox_coorList[i * 4 + 2];
        int infer_y2 = bbox_coorList[i * 4 + 3];

        // std::cout << "infer " << infer_x1 << " " << infer_y1 << " " << infer_x2 << " " << infer_y2 << std::endl;

        float* adjusted = new float[4];


        int max_overlap_area = 0;
        uint8_t chosenImgIdx = 0;

        for (uint8_t imgIdx = 0; imgIdx < images.size(); ++imgIdx) {
            const ConcatDims &dims = concatDims[imgIdx];
            // Image bounding box in concatenated frame
            int img_x1 = dims.x1;
            int img_y1 = dims.y1;
            int img_x2 = dims.x1 + dims.width;
            int img_y2 = dims.y1 + dims.height;

            // Calculate the intersection rectangle between the inference bbox and the image bbox
            int intersect_x1 = std::max(infer_x1, img_x1);
            int intersect_y1 = std::max(infer_y1, img_y1);
            int intersect_x2 = std::min(infer_x2, img_x2);
            int intersect_y2 = std::min(infer_y2, img_y2);

            // Calculate the width and height of the intersection rectangle
            int intersect_width = std::max(0, intersect_x2 - intersect_x1);
            int intersect_height = std::max(0, intersect_y2 - intersect_y1);

            // Calculate the area of intersection
            int intersect_area = intersect_width * intersect_height;

            // Keep track of the image with the largest intersection
            if (intersect_area > max_overlap_area) {
                max_overlap_area = intersect_area;
                chosenImgIdx = imgIdx;
                // Update the index of the best image
                adjusted[0] = intersect_x1 - img_x1;
                adjusted[1] = intersect_y1 - img_y1;
                adjusted[2] = intersect_x2 - img_x1;
                adjusted[3] = intersect_y2 - img_y1;
            }
        }

        // std::cout << "adjusted " << adjusted[0] << " " << adjusted[1] << " " << adjusted[2] << " " << adjusted[3] << std::endl;
        scaleBBox(orig_h, orig_w, infer_h, infer_w, adjusted, orig_bboxCoors);

        delete [] adjusted;



        // Original bounding box coordinates (top-left x, y) and (bottom-right x, y)
        int x1 = orig_bboxCoors[0];
        int y1 = orig_bboxCoors[1];
        int x2 = orig_bboxCoors[2];
        int y2 = orig_bboxCoors[3];

        // std::cout << "orig " << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

        // Crop from the corresponding image
        if ((y2 - y1) <= 0 || (x2 - x1) <= 0 || nmsed_scores[i] < confidenceThreshold) {
            numInvalidDets++;
            // std::cout << "Invalid detection" << std::endl;
            continue;
        }
        // std::cout << "Valid detection" << std::endl;
        imageIndexList.emplace_back(std::make_pair(chosenImgIdx, i));
        cv::cuda::GpuMat croppedBBox = images[chosenImgIdx](
            cv::Range(y1, y2), 
            cv::Range(x1, x2)
        ).clone();

        // Store the cropped bbox in the output vector
        croppedBBoxes.emplace_back(BoundingBox<cv::cuda::GpuMat>{
            croppedBBox,
            x1, y1, x2, y2,
            0.f,
            0
        });
        // saveGPUAsImg(croppedBBox, "bbox_" + std::to_string(i) + ".jpg");
    }
    return imageIndexList;
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

void BaseBBoxCropper::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOANDING configs...", __func__);
    if (!isConstructing) { // If this is not called from the constructor
        BasePostprocessor::loadConfigs(jsonConfigs, isConstructing);
    }
    msvc_augment = jsonConfigs["msvc_augment"];
    msvc_confThreshold = jsonConfigs["msvc_confThreshold"];
    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);
}

BaseBBoxCropper::BaseBBoxCropper(const json &jsonConfigs) : BasePostprocessor(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name); 
}

void BaseBBoxCropper::cropping() {
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

    // Height and width of the image used for inference
    int orig_h, orig_w, infer_h, infer_w;

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

                concatConfigsGenerator(msvc_inferenceShape, msvc_concat, 2);

                infer_h = msvc_inferenceShape[0][1];
                infer_w = msvc_inferenceShape[0][2];
                
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
            std::vector<MemUsageType> totalInMem, totalOutMem(numImagesInFrame, 0), totalEncodedOutMem(numImagesInFrame, 0);
            msvc_overallTotalReqCount++;

            // 11. The moment the request starts being processed by the cropper, after the batch was unloaded (ELEVENTH_TIMESTAMP)
            for (uint8_t concatInd = 0; concatInd < numImagesInFrame; concatInd++) {
                uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + concatInd;
                currReq.req_origGenTime[imageIndexInBatch].emplace_back(std::chrono::high_resolution_clock::now());
            }

            std::vector<std::pair<uint8_t, uint16_t>> indexLists;

            // If there is no object in frame, we don't have to do nothing.
            int numDetsInFrame = (int)num_detections[i];
            if (numDetsInFrame <= 0) {
                if (msvc_augment) {
                    // Generate a random box for downstream wrorkload
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_int_distribution<> dis(0, 1);

                    if (dis(gen) == 0) {
                        continue;
                    }

                    for (uint8_t j = 0; j < numImagesInFrame; j++) {
                        // uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + j;
                        singleImageBBoxList.emplace_back(
                            BoundingBox<cv::cuda::GpuMat>{
                                cv::cuda::GpuMat(64, 64, CV_8UC3),
                                0, 0, 64, 64,
                                1.f,
                                1
                            }
                        );
                        nmsed_classes[i * maxNumDets] = 1;
                        indexLists.emplace_back(std::make_pair(j, 0));
                    }
                    numDetsInFrame = numImagesInFrame;
                    nmsed_classes[i * maxNumDets] = 1;
                } else {
                    continue;
                }
            } else {
                // Otherwise, we need to do some cropping.

                // First we need to set the infer_h,w and the original h,w of the image.
                // infer_h,w are given in the last dimension of the request data from the inferencer
                infer_h = currReq.req_data.back().shape[1];
                infer_w = currReq.req_data.back().shape[2];
                // orig_h,w are given in the shape of the image in the image list, which is carried from the preprocessor
                // TODO: For now, we assume that all images in the concatenated frame have the same shape
                orig_h = imageList[i].shape[1];
                orig_w = imageList[i].shape[2];

                // List of the images in the concatenated frame to be cropped from
                std::vector<cv::cuda::GpuMat> concatImageList;
                for (uint8_t concatInd = 0; concatInd < numImagesInFrame; concatInd++) {
                    uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + concatInd;
                    concatImageList.emplace_back(imageList[imageIndexInBatch].data);
                }

                // Cropping the detected bounding boxes from the original images and returns:
                // 1/ in `concatIndexList`, the list of the indices of
                //      (a) the images in the concatenated frame from which the bounding box is cropped
                //      (b) the bounding boxes in the detected list of the whole frame to retrieve the scores and classes
                // 2/ the list of the cropped bounding boxes in `singleImageBBoxList`, each object in this list contains:
                //      (a) the cropped bounding box
                //      (b) the coordinates of the bounding box in the original image
                //      (c) the score of the bounding box
                //      (d) the class of the bounding box
                indexLists = crop(concatImageList,
                                  msvc_concat.list,
                                  currReq.req_concatInfo[i],
                                  orig_h,
                                  orig_w,
                                  infer_h,
                                  infer_w,
                                  numDetsInFrame,
                                  nmsed_boxes + i * maxNumDets * 4,
                                  nmsed_scores + i * maxNumDets,
                                  msvc_confThreshold,
                                  singleImageBBoxList);
            }

            // After cropping, due to some invalid detections,
            // we need to update the number of detections in the frame
            numDetsInFrame = indexLists.size();
            spdlog::get("container_agent")->trace("{0:s} cropped {1:d} bboxes in image {2:d}", msvc_name, numDetsInFrame, i);

            std::vector<std::vector<PerQueueOutRequest>> outReqList(msvc_OutQueue.size(), std::vector<PerQueueOutRequest>());

            // calculate the total memory used for the input images
            for (BatchSizeType j = 0; j < numImagesInFrame; ++j) {
                // the index of the image in the whole batch of multiple concatenated frames
                uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + j;
                totalInMem.emplace_back(imageList[imageIndexInBatch].data.channels() * imageList[imageIndexInBatch].data.rows * 
                                        imageList[imageIndexInBatch].data.cols * CV_ELEM_SIZE1(imageList[imageIndexInBatch].data.type()));
            }

            // Number of detections in each individual image in the concatenated frame
            std::vector<uint16_t> numsDetsInImages(numImagesInFrame, 0);
            // The index of the bounding box in its corresponding image
            std::vector<uint16_t> indexInImageDetList(numDetsInFrame);
            std::unordered_map<int16_t, int> classCount;
            std::unordered_map<int16_t, int> perClassProcessedCount;
            spdlog::get("container_agent")->trace("{0:s} updating classes and scores", msvc_name);
            for (int j = 0; j < numDetsInFrame; ++j) {
                indexInImageDetList[j] = numsDetsInImages[indexLists[j].first];
                numsDetsInImages[indexLists[j].first]++;

                // update the score and class of the bounding box                
                singleImageBBoxList[j].score = nmsed_scores[i * maxNumDets + indexLists[j].second];
                singleImageBBoxList[j].classID = (int16_t)nmsed_classes[i * maxNumDets + indexLists[j].second];
                if (classCount.find(singleImageBBoxList[j].classID) == classCount.end()) {
                    classCount[singleImageBBoxList[j].classID] = 0;
                    perClassProcessedCount[singleImageBBoxList[j].classID] = 0;
                }
                classCount[singleImageBBoxList[j].classID]++;
            }

            spdlog::get("container_agent")->trace("{0:s} ready to crop {1:d} detections", msvc_name, numDetsInFrame);
            // After cropping, we need to find the right queues to put the bounding boxes in
            for (int j = 0; j < numDetsInFrame; ++j) {
                // cv::Mat test;
                // singleImageBBoxList[j].download(test);
                // cv::imwrite("bbox.jpg", test);
                auto bbox = singleImageBBoxList[j].bbox;
                bboxClass = (int16_t)singleImageBBoxList[j].classID;
                cv::Mat cpuBox;
                cv::Mat encodedBox;
                uint32_t boxEncodedMemSize = 0;

                // Find the indices of the queues that need this class number
                // And convert the bounding box to CPU if needed
                for (size_t k = 0; k < this->classToDnstreamMap.size(); ++k) {
                    NumQueuesType qIndex = MAX_NUM_QUEUES;
                    // Find the indices of the queues that need this class number
                    // in the constructor of each microservice, we map the class number to the corresponding queue index in 
                    // `classToDntreamMap`.
                    if ((classToDnstreamMap.at(k).first == bboxClass) || (classToDnstreamMap.at(k).first == -1)) {
                        qIndex = classToDnstreamMap.at(k).second;
                        queueIndex.emplace_back(qIndex);
                        //if not outReqList at qIndex is empty then insert one
                        if (outReqList.at(qIndex).empty()) {
                            outReqList.at(qIndex).emplace_back(PerQueueOutRequest());
                        }
                    }
                    if (qIndex == MAX_NUM_QUEUES) {
                        continue;
                    }

                    if (msvc_activeOutQueueIndex.at(qIndex) != 1) { //If CPU serialized data
                        continue;
                    }
                    // Because GPU->CPU is expensive, we only do it once
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
                        bbox.download(cpuBox, cvStream);
                        cvStream.waitForCompletion();
                    }
                    if (msvc_OutQueue.at(qIndex)->getEncoded() && encodedBox.empty() && !cpuBox.empty()) {
                        encodedBox = encodeResults(cpuBox);
                        boxEncodedMemSize = encodedBox.cols * encodedBox.rows * encodedBox.channels() * CV_ELEM_SIZE1(encodedBox.type());
                    }
                }
                // If this class number is not needed anywhere downstream, we don't need to do anything
                if (queueIndex.empty()) {
                    continue;
                }


                // Putting the bounding box into an `outReq` to be sent out
                bboxShape = {bbox.channels(),
                             bbox.rows,
                             bbox.cols};

                // Index of the image in the whole batch of multiple concatenated frames
                // each frame has mscc_concat.numImgs images
                uint16_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + indexLists[j].first;

                // There could be multiple timestamps in the request, but the first one always represent
                // the moment this request was generated at the very beginning of the pipeline

                // The travel path of the request is the path of the image
                // We will concat more information to it
                currReq_path = currReq.req_travelPath[imageIndexInBatch];

                // Forwards the bounding box meant for appropriate queues meant for the downstream microservices
                for (auto qIndex : queueIndex) {
                    std::string path = currReq_path;
                    // Add the number of bounding boxes in the image and the index of the bounding box in the image
                    path += "|" + std::to_string(numsDetsInImages[indexLists[j].first]) + "|" + std::to_string(indexInImageDetList[j]);
                    // // Add latency durations to the path
                    addTimestampsToPath(path, currReq.req_origGenTime[imageIndexInBatch]);
                    

                    if (dnstreamMicroserviceList[qIndex].portions.size() > 1 &&
                        ( (float) perClassProcessedCount[bboxClass]++ / (float) classCount[bboxClass])
                            > dnstreamMicroserviceList[qIndex].portions[outReqList.at(qIndex).size()-1]) {
                        outReqList.at(qIndex).emplace_back();
                        perClassProcessedCount[bboxClass] = 0;
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

                        // We only include the first timestamp in the request to the next container to make it aware of the time the request was generated
                        // at the very beginning of the pipeline, which will be used to calculate the end-to-end latency and determine things like dropping
                        outReqList.at(qIndex).back().cpuReq.req_origGenTime.emplace_back(RequestTimeType{currReq.req_origGenTime[imageIndexInBatch].front()});
                        outReqList.at(qIndex).back().cpuReq.req_e2eSLOLatency.emplace_back(currReq.req_e2eSLOLatency[imageIndexInBatch]);
                        outReqList.at(qIndex).back().cpuReq.req_travelPath.emplace_back(path);
                        outReqList.at(qIndex).back().cpuReq.req_data.emplace_back(reqDataCPU);
                        outReqList.at(qIndex).back().cpuReq.req_batchSize = 1;

                        spdlog::get("container_agent")->trace("{0:s} emplaced a bbox of class {1:d} to CPU queue {2:d}.", msvc_name, bboxClass, qIndex);
                    } else {
                        cv::cuda::GpuMat out(bbox.size(), bbox.type());
                        checkCudaErrorCode(cudaMemcpyAsync(
                            out.cudaPtr(),
                            bbox.cudaPtr(),
                            bbox.cols * bbox.rows * bbox.channels() * CV_ELEM_SIZE1(bbox.type()),
                            cudaMemcpyDeviceToDevice,
                            postProcStream
                        ), __func__);

                        outReqList.at(qIndex).back().gpuReq.req_origGenTime.emplace_back(RequestTimeType{currReq.req_origGenTime[imageIndexInBatch].front()});
                        outReqList.at(qIndex).back().gpuReq.req_e2eSLOLatency.emplace_back(currReq.req_e2eSLOLatency[imageIndexInBatch]);
                        outReqList.at(qIndex).back().gpuReq.req_travelPath.emplace_back(path);
                        outReqList.at(qIndex).back().gpuReq.req_data.emplace_back(reqData);
                        outReqList.at(qIndex).back().gpuReq.req_batchSize = 1;

                        spdlog::get("container_agent")->trace("{0:s} emplaced a bbox of class {1:d} to GPU queue {2:d}.", msvc_name, bboxClass, qIndex);
                    }
                    uint32_t imageMemSize = bbox.cols * bbox.rows * bbox.channels() * CV_ELEM_SIZE1(bbox.type());
                    outReqList.at(qIndex).back().totalSize += imageMemSize;
                    outReqList.at(qIndex).back().totalEncodedSize += boxEncodedMemSize;
                    totalOutMem[indexLists[j].first] += imageMemSize;
                    totalEncodedOutMem[indexLists[j].first] += boxEncodedMemSize;
                }
                queueIndex.clear();
            }

            NumQueuesType qIndex = 0;
            for (auto &outList : outReqList) {
                for (auto &outReq : outList) {
                    if (msvc_activeOutQueueIndex.at(qIndex) == 1) { //Local CPU GPU
                        // Make sure the time is uniform across all the bounding boxes
                        uint pathNum = 0;
                        for (auto &time: outReq.cpuReq.req_origGenTime) {
                            ClockType timeNow = std::chrono::high_resolution_clock::now();
                            time.emplace_back(timeNow);
                            outReq.cpuReq.req_travelPath[pathNum] += "|" + std::to_string(std::chrono::duration_cast<TimePrecisionType>(timeNow.time_since_epoch()).count());
                            pathNum++;
                        }
                        // Add the total size of bounding boxes heading to this queue
                        for (auto &path: outReq.cpuReq.req_travelPath) {
                            path += "|" + std::to_string(outReq.totalEncodedSize) + "|" +
                                    std::to_string(outReq.totalSize) + "]";
                            std::cout << path << std::endl;
                        }

                        msvc_OutQueue.at(qIndex)->emplace(outReq.cpuReq);
                    } else { //Local GPU Queue
                        // Make sure the time is uniform across all the bounding boxes
                        uint pathNum = 0;
                        for (auto &time: outReq.gpuReq.req_origGenTime) {
                            ClockType timeNow = std::chrono::high_resolution_clock::now();
                            time.emplace_back(timeNow);
                            outReq.gpuReq.req_travelPath[pathNum] += "|" + std::to_string(std::chrono::duration_cast<TimePrecisionType>(timeNow.time_since_epoch()).count());
                            pathNum++;
                        }
                        // Add the total size of bounding boxes heading to this queue
                        for (auto &path: outReq.gpuReq.req_travelPath) {
                            path += "|" + std::to_string(outReq.totalEncodedSize) + "|" +
                                    std::to_string(outReq.totalSize) + "]";
                        }
                        msvc_OutQueue.at(qIndex)->emplace(outReq.gpuReq);
                    }
                }
                qIndex++;
            }

            // If the number of warmup batches has been passed, we start to record the latency
            if (warmupCompleted()) {
                for (uint8_t j = 0; j < numImagesInFrame; ++j) {
                    uint8_t imageIndexInBatch = currReq.req_concatInfo[i].firstImageIndex + j;
                    // 12. When the request was completed by the postprocessor (TWELFTH_TIMESTAMP)
                    currReq.req_origGenTime[imageIndexInBatch].emplace_back(std::chrono::high_resolution_clock::now());
                    std::string originStream = getOriginStream(currReq.req_travelPath[imageIndexInBatch]);
                    // TODO: Add the request number
                    msvc_processRecords.addRecord(currReq.req_origGenTime[imageIndexInBatch], currReq_batchSize, totalInMem[j], totalOutMem[j], totalEncodedOutMem[j], 0, originStream);
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

            for (auto &outList : outReqList) {
                outList.clear();
            }
            singleImageBBoxList.clear();
        }

        msvc_batchCount++;
        msvc_miniBatchCount++;

        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamDestroy(postProcStream), __func__);
    msvc_logFile.close();
    STOPPED = true;
}


/**
 * @brief Generate random bboxes
 * 
 * @param bboxList 
 */
void BaseBBoxCropper::generateRandomBBox(
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

void BaseBBoxCropper::cropProfiling() {
}