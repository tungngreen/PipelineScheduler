#include <yolov5.h>

YoloV5Postprocessor::YoloV5Postprocessor(const BaseMicroserviceConfigs &config) : BasePostprocessor(config) {
}

void YoloV5Postprocessor::postProcessing() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Data package to be sent to and processed at the next microservice
    std::vector<RequestData<LocalGPUReqDataType>> outReqData;
    // List of images carried from the previous microservice here to be cropped from.
    std::vector<RequestData<LocalGPUReqDataType>> imageList;
    // Instance of data to be packed into `outReqData`
    RequestData<LocalGPUReqDataType> reqData;

    // List of bounding boxes cropped from one single image
    std::vector<cv::cuda::GpuMat> singleImageBBoxList;

    // Current incoming equest and request to be sent out to the next
    Request<LocalGPUReqDataType> currReq, outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;

    // Shape of cropped bounding boxes
    RequestShapeType bboxShape;

    while (true) {
        // Allowing this thread to naturally come to an end
        if (this->STOP_THREADS) {
            break;
        }
        else if (this->PAUSE_THREADS) {
            continue;
        }

        // Processing the next incoming request
        currReq = msvc_InQueue.at(0)->pop2();
        msvc_inReqCount++;

        currReq_genTime = currReq.req_origGenTime;
        // We need to check if the next request is worth processing.
        // If it's too late, then we can drop and stop processing this request.
        if (!this->checkReqEligibility(currReq_genTime)) {
            continue;
        }
        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_recvTime = std::chrono::high_resolution_clock::now();
        if (this->msvc_inReqCount > 1) {
            this->updateReqRate(currReq_genTime);
        }
        currReq_batchSize = currReq.req_batchSize;

        /**
         * @brief Each request to the postprocessing microservice of YOLOv5 contains the buffers which are results of TRT inference 
         * as well as the images from which bounding boxes will be cropped.
         * `The buffers` are a vector of 4 small raw memory (float) buffers (since using `BatchedNMSPlugin` provided by TRT), which are
         * 1/ `num_detections` (Batch, 1): number of objects detected in this frame
         * 2/ `nmsed_boxes` (Batch, TopK, 4): the boxes remaining after nms is performed.
         * 3/ `nmsed_scores` (Batch, TopK): the scores of these boxes.
         * 4/ `nmsed_classes` (Batch, TopK):
         * We need to bring these buffers to CPU in order to process them.
         */

        uint16_t maxNumDets = msvc_dataShape[2][0];

        std::vector<RequestData<LocalGPUReqDataType>> currReq_data = currReq.req_data;
        float num_detections[currReq_batchSize];
        float nmsed_boxes[currReq_batchSize][maxNumDets][4];
        float nmsed_scores[currReq_batchSize][maxNumDets];
        float nmsed_classes[currReq_batchSize][maxNumDets];
        float *numDetList = num_detections;
        float *nmsedBoxesList = &nmsed_boxes[0][0][0];
        float *nmsedScoresList = &nmsed_scores[0][0];
        float *nmsedClassesList = &nmsed_classes[0][0];

        // float numDetList[currReq_batchSize];
        // float nmsedBoxesList[currReq_batchSize][maxNumDets][4];
        // float nmsedScoresList[currReq_batchSize][maxNumDets];
        // float nmsedClassesList[currReq_batchSize][maxNumDets];
        std::vector<float *> ptrList{numDetList, nmsedBoxesList, nmsedScoresList, nmsedClassesList};
        std::vector<size_t> bufferSizeList;

        cudaStream_t postProcStream;
        checkCudaErrorCode(cudaStreamCreate(&postProcStream));
        for (std::size_t i = 0; i < currReq_data.size(); ++i) {
            size_t bufferSize = this->msvc_modelDataType * (size_t)currReq_batchSize;
            RequestShapeType shape = currReq_data[i].shape;
            for (uint8_t j = 0; j < shape.size(); ++j) {
                bufferSize *= shape[j];
            }
            bufferSizeList.emplace_back(bufferSize);
            checkCudaErrorCode(cudaMemcpyAsync(
                (void *) ptrList[i],
                currReq_data[i].data.cudaPtr(),
                bufferSize,
                cudaMemcpyDeviceToHost,
                postProcStream
            ));
        }

        // List of images to be cropped from
        imageList = currReq.upstreamReq_data; 

        // class of the bounding box cropped from one the images in the image list
        int16_t bboxClass;
        // The index of the queue we are going to put data on based on the value of `bboxClass`
        NumQueuesType queueIndex;

        // Doing post processing for the whole batch
        for (BatchSizeType i = 0; i < currReq_batchSize; ++i) {
            // Height and width of the image used for inference
            int infer_h, infer_w;

            // If there is no object in frame, we don't have to do nothing.
            int numDetsInFrame = (int)numDetList[i];
            if (numDetsInFrame <= 0) {
                continue;
            }

            // Otherwise, we need to do some cropping.

            infer_h = imageList[i].shape[1];
            infer_w = imageList[i].shape[2];
            crop(imageList[i].data, infer_h, infer_w, numDetsInFrame, nmsed_boxes[i][0], singleImageBBoxList);

            // After cropping, we need to find the right queues to put the bounding boxes in
            for (int j = 0; j < numDetsInFrame; ++i) {
                bboxClass = (int16_t)nmsed_classes[i][j];
                queueIndex = -1;
                // in the constructor of each microservice, we map the class number to the corresponding queue index in 
                // `classToDntreamMap`.
                for (size_t k = 0; k < this->classToDnstreamMap.size(); ++k) {
                    if (classToDnstreamMap.at(k).second == bboxClass) {
                        queueIndex = this->classToDnstreamMap.at(i).second; 
                        // Breaking is only appropriate if case we assume the downstream only wants to take one class
                        // TODO: More than class-of-interests for 1 queue
                        break;
                    }
                }
                // If this class number is not needed anywhere downstream
                if (queueIndex == -1) {
                    continue;
                }

                // Putting the bounding box into an `outReq` to be sent out
                bboxShape = {singleImageBBoxList[j].channels(), singleImageBBoxList[j].rows, singleImageBBoxList[j].cols};
                reqData = {
                    bboxShape,
                    singleImageBBoxList[j].clone()
                };
                outReqData.emplace_back(reqData);
                outReq = {
                    std::chrono::_V2::system_clock::now(),
                    currReq.req_e2eSLOLatency,
                    "",
                    1,
                    outReqData, //req_data
                    currReq.req_data // upstreamReq_data
                };
                msvc_OutQueue.at(queueIndex)->emplace(outReq);
            }
            // Clearing out data of the vector
            outReqData.clear();
            singleImageBBoxList.clear();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(this->msvc_interReqTime));

    }


}