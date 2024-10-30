#include "baseprocessor.h"

using namespace spdlog;

inline cv::Scalar vectorToScalar(const std::vector<float> &vec) {
    // Ensure the vector has exactly 4 elements
    if (vec.size() == 1) {
        return cv::Scalar(vec[0]);
    } else if (vec.size() == 3) {
        return cv::Scalar(vec[0], vec[1], vec[2]);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
}

/**
 * @brief normalize the input data by subtracting and dividing values from the original pixesl
 * 
 * @param input the input data
 * @param subVals values to be subtracted
 * @param divVals values to be dividing by
 * @param stream an opencv stream for asynchronous operation on cuda
 */
inline cv::cuda::GpuMat normalize(
        const cv::cuda::GpuMat &input,
        cv::cuda::Stream &stream,
        const std::vector<float> &subVals,
        const std::vector<float> &divVals,
        const float normalized_scale
) {
    spdlog::get("container_agent")->trace("Going into {0:s}", __func__);
    cv::cuda::GpuMat normalized;
    cv::Scalar subValsScalar = vectorToScalar(subVals);
    cv::Scalar divValsScalar = vectorToScalar(divVals);
    if (input.channels() == 1) {
        input.convertTo(normalized, CV_32FC1, normalized_scale, stream);
        cv::cuda::subtract(normalized, subValsScalar, normalized, cv::noArray(), -1, stream);
        cv::cuda::divide(normalized, divValsScalar, normalized, 1, -1, stream);
    } else if (input.channels() == 3) {
        input.convertTo(normalized, CV_32FC3, normalized_scale, stream);
        cv::cuda::subtract(normalized, subValsScalar, normalized, cv::noArray(), -1, stream);
        cv::cuda::divide(normalized, divValsScalar, normalized, 1, -1, stream);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }

    stream.waitForCompletion();
    spdlog::get("container_agent")->trace("Finished {0:s}", __func__);

    return normalized;
}

inline cv::cuda::GpuMat cvtHWCToCHW(
        const cv::cuda::GpuMat &input,
        cv::cuda::Stream &stream,
        uint8_t IMG_TYPE
) {

    spdlog::get("container_agent")->trace("Going into {0:s}", __func__);
    uint16_t height = input.rows;
    uint16_t width = input.cols;
    /**
     * @brief TODOs
     * This is the correct way but since we haven't figured out how to carry to image to be cropped
     * it screws things up a bit.
     * cv::cuda::GpuMat transposed(1, height * width, CV_8UC3);
     */
    // cv::cuda::GpuMat transposed(height, width, CV_8UC3);
    cv::cuda::GpuMat transposed(1, height * width, IMG_TYPE);
    std::vector<cv::cuda::GpuMat> channels;
    if (input.channels() == 1) {
        uint8_t IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE;
        channels = {
                cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[0]))
        };
    } else if (input.channels() == 3) {
        uint8_t IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE ^ 16;
        size_t channel_mem_width = height * width;

        channels = {
                cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[0])),
                cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channel_mem_width])),
                cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channel_mem_width * 2]))
        };
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
    cv::cuda::split(input, channels, stream);

    stream.waitForCompletion();

    spdlog::get("container_agent")->trace("Finished {0:s}", __func__);

    return transposed;
}

inline cv::cuda::GpuMat convertColor(
        const cv::cuda::GpuMat &input,
        uint8_t IMG_TYPE,
        uint8_t COLOR_CVT_TYPE,
        cv::cuda::Stream &stream
) {
    spdlog::get("container_agent")->trace("Going into {0:s}", __func__);
    // If the image is grayscale, then the target image type should be 0
    uint16_t TARGET_IMG_TYPE;
    if (GRAYSCALE_CONVERSION_CODES.count(COLOR_CVT_TYPE)) {
        TARGET_IMG_TYPE = 0;
    } else {
        TARGET_IMG_TYPE = IMG_TYPE;
    }

    cv::cuda::GpuMat color_cvt_image(input.rows, input.cols, TARGET_IMG_TYPE);
    cv::cuda::cvtColor(input, color_cvt_image, COLOR_CVT_TYPE, 0, stream);

    stream.waitForCompletion();
    spdlog::get("container_agent")->trace("Finished {0:s}", __func__);

    return color_cvt_image;
}

/**
 * @brief resize the input data without changing the aspect ratio and pad the empty area with a designated color
 * 
 * @param input the input data
 * @param height the expected height after processing
 * @param width  the expect width after processing
 * @param bgcolor color to pad the empty area with
 * @return cv::cuda::GpuMat 
 */
inline cv::cuda::GpuMat resizePadRightBottom(
        const cv::cuda::GpuMat &input,
        const size_t height,
        const size_t width,
        const std::vector<float> &bgcolor,
        cv::cuda::Stream &stream,
        uint8_t IMG_TYPE,
        uint8_t COLOR_CVT_TYPE,
        uint8_t RESIZE_INTERPOL_TYPE

) {
    spdlog::get("container_agent")->trace("Going into {0:s}", __func__);

    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    //Create a new GPU Mat 
    cv::cuda::GpuMat resized(unpad_h, unpad_w, input.type());
    cv::cuda::resize(input, resized, resized.size(), 0, 0, RESIZE_INTERPOL_TYPE, stream);
    cv::cuda::GpuMat out(height, width, input.type(), vectorToScalar(bgcolor));
    // Creating an opencv stream for asynchronous operation on cuda
    resized.copyTo(out(cv::Rect(0, 0, resized.cols, resized.rows)), stream);

    stream.waitForCompletion();
    spdlog::get("container_agent")->trace("Finished {0:s}", __func__);

    return out;
}

inline bool resizeIntoFrame(
        const cv::cuda::GpuMat &input,
        cv::cuda::GpuMat &frame,
        const uint16_t left,
        const uint16_t top,
        const uint16_t height,
        const uint16_t width,
        cv::cuda::Stream &stream,
        uint8_t IMG_TYPE,
        uint8_t COLOR_CVT_TYPE,
        uint8_t RESIZE_INTERPOL_TYPE
) {
    spdlog::get("container_agent")->trace("Going into {0:s}", __func__);

    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat resized(unpad_h, unpad_w, input.type());
    cv::cuda::resize(input, resized, resized.size(), 0, 0, RESIZE_INTERPOL_TYPE, stream);

    resized.copyTo(frame(cv::Rect(left, top, resized.cols, resized.rows)), stream);
    stream.waitForCompletion();
    spdlog::get("container_agent")->trace("Finished {0:s}", __func__);

    return true;
}

/**
 * @brief Get number at index from a string of comma separated numbers
 * 
 * @param str 
 * @param index 
 * @return uint64_t 
 */
inline uint64_t getNumberAtIndex(const std::string &str, int index) {
    int currentIndex = 0;
    int startPos = 0;
    uint64_t number = 0;

    if (index == -1) {
        // Handle the case where the index is -1 (last number)
        for (size_t i = 0; i < str.length(); i++) {
            if (str[i] == ',') {
                startPos = i + 1;
            }
        }
        std::string numberStr = str.substr(startPos);
        number = std::stoull(numberStr);
        return number;
    }

    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == ',') {
            if (currentIndex == index) {
                std::string numberStr = str.substr(startPos, i - startPos);
                number = std::stoull(numberStr);
                return number;
            }
            startPos = i + 1;
            currentIndex++;
        }
    }

    // Handle the last number in the string
    if (currentIndex == index) {
        std::string numberStr = str.substr(startPos);
        number = std::stoull(numberStr);
        return number;
    }

    return 0; // Return 0 if the index is out of range
}

BasePreprocessorConfigs BasePreprocessor::loadConfigsFromJson(const json &jsonConfigs) {
    BasePreprocessorConfigs configs;

    jsonConfigs.at("msvc_imgType").get_to(configs.msvc_imgType);
    jsonConfigs.at("msvc_colorCvtType").get_to(configs.msvc_colorCvtType);
    jsonConfigs.at("msvc_resizeInterpolType").get_to(configs.msvc_resizeInterpolType);
    std::string normVal;
    jsonConfigs.at("msvc_imgNormScale").get_to(normVal);
    configs.msvc_imgNormScale = fractionToFloat(normVal);
    jsonConfigs.at("msvc_subVals").get_to(configs.msvc_subVals);
    jsonConfigs.at("msvc_divVals").get_to(configs.msvc_divVals);
    return configs;
}


/**
 * @brief Load the configurations from the json file
 * 
 * @param jsonConfigs 
 * @param isConstructing
 * 
 * @note The function is called from the constructor or when the microservice is to be reloaded
 */
void BasePreprocessor::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    // Load the configs from the json file for Microservice class
    if (!isConstructing) { // If the function is not called from the constructor
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }

    BasePreprocessorConfigs configs = loadConfigsFromJson(jsonConfigs);

    msvc_concat.numImgs = jsonConfigs["msvc_concat"];

    msvc_imgType = configs.msvc_imgType;
    msvc_colorCvtType = configs.msvc_colorCvtType;
    msvc_resizeInterpolType = configs.msvc_resizeInterpolType;
    msvc_imgNormScale = configs.msvc_imgNormScale;
    msvc_subVals = configs.msvc_subVals;
    msvc_divVals = configs.msvc_divVals;

    if (msvc_BATCH_MODE == BATCH_MODE::OURS) {
        updateCycleTiming();
    }
    msvc_toReloadConfigs = false;
}

/**
 * @brief Construct a new Base Preprocessor that inherites the LocalGPUDataMicroservice given the `InType`
 * 
 * @param configs 
 */
BasePreprocessor::BasePreprocessor(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs);
    //set to maximum value
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name);
}

BasePreprocessor::BasePreprocessor(const BasePreprocessor &other) : Microservice(other) {
    std::lock(msvc_overallMutex, other.msvc_overallMutex);
    std::lock_guard<std::mutex> lock1(msvc_overallMutex, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(other.msvc_overallMutex, std::adopt_lock);

    msvc_imgType = other.msvc_imgType;
    msvc_colorCvtType = other.msvc_colorCvtType;
    msvc_resizeInterpolType = other.msvc_resizeInterpolType;
    msvc_imgNormScale = other.msvc_imgNormScale;
    msvc_subVals = other.msvc_subVals;
    msvc_divVals = other.msvc_divVals;
    msvc_toReloadConfigs = other.msvc_toReloadConfigs;
}


/**
 * @brief 
 * 
 */
void BasePreprocessor::preprocess() {
    RequestData<LocalGPUReqDataType> data;

    std::vector<Request<LocalGPUReqDataType>> outBatch;

    // Incoming request
    Request<LocalGPUReqDataType> currReq, outReq;

    Request<LocalCPUReqDataType> currCPUReq;

    auto timeNow = std::chrono::high_resolution_clock::now();

    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name);
    cv::cuda::Stream *preProcStream = nullptr;
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

                concatConfigsGenerator(msvc_outReqShape.at(0), msvc_concat, 2);

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                setDevice();
                preProcStream = new cv::cuda::Stream();
                outReq = {};

                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        currCPUReq = msvc_InQueue.at(0)->pop1();
        if (!validateRequest<LocalCPUReqDataType>(currCPUReq)) {
            continue;
        }
        currReq = uploadReq(currCPUReq);

        msvc_overallTotalReqCount++;

        if (msvc_concat.currIndex == 0) {
            // Create a new frame to hold the concatenated images
            outReq = {};
            outReq.req_origGenTime = {};
            outReq.req_e2eSLOLatency = {};
            outReq.req_travelPath = {};
            outReq.req_data = {RequestData<LocalGPUReqDataType>{}};
            outReq.req_data[0].data = cv::cuda::GpuMat(msvc_outReqShape.at(0)[0][1],
                                                       msvc_outReqShape.at(0)[0][2],
                                                       msvc_imgType);
            outReq.req_data[0].shape = RequestDataShapeType({(msvc_outReqShape.at(0))[0][0], (msvc_outReqShape.at(0))[0][1],
                                                             (msvc_outReqShape.at(0))[0][2]});
            outReq.upstreamReq_data = {};
        }

        outReq.req_origGenTime.emplace_back(currReq.req_origGenTime[0]);
        outReq.req_e2eSLOLatency.emplace_back(currReq.req_e2eSLOLatency[0]);
        outReq.req_travelPath.emplace_back(currReq.req_travelPath[0] + "[" + msvc_hostDevice + "|" + msvc_containerName + "|" +
                                           std::to_string(msvc_overallTotalReqCount));
        outReq.upstreamReq_data.emplace_back(currReq.req_data[0]);
        spdlog::get("container_agent")->trace("{0:s} popped a request. In queue size is {2:d}.",
                                              msvc_name, msvc_InQueue.at(0)->size());

        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer frame

        data.data = convertColor(currReq.req_data[0].data,
                                 msvc_imgType,
                                 msvc_colorCvtType,
                                 *preProcStream);

        bool success = resizeIntoFrame(
            data.data,
            outReq.req_data[0].data,
            msvc_concat.concatDims[msvc_concat.currIndex].x1,
            msvc_concat.concatDims[msvc_concat.currIndex].y1,
            msvc_concat.concatDims[msvc_concat.currIndex].height,
            msvc_concat.concatDims[msvc_concat.currIndex].width,
            *preProcStream,
            msvc_imgType,
            msvc_colorCvtType,
            msvc_resizeInterpolType
        );
                
        if (!success) {
            spdlog::get("container_agent")->error("{0:s} failed to resize the image.", msvc_name);
            continue;
        } else {
            spdlog::get("container_agent")->trace("{0:s} resized an image of [{1:d}, {2:d}] -> "
                                                  "[{3:d}, {4:d}] and put into frame at index {5:d}.",
                                                  msvc_name,
                                                  currReq.req_data[0].data.rows,
                                                  currReq.req_data[0].data.cols,
                                                  (this->msvc_outReqShape.at(0))[0][1],
                                                  (this->msvc_outReqShape.at(0))[0][2],
                                                  msvc_concat.currIndex);

            // Consider this the moment the request preprocessed and is waiting to be batched
            // 6. The moment the request's preprocessing is completed (SIXTH_TIMESTAMP)
            timeNow = std::chrono::high_resolution_clock::now();
            outReq.req_origGenTime.back().emplace_back(timeNow);
            msvc_concat.currIndex = (++msvc_concat.currIndex % msvc_concat.numImgs);

            // If the buffer frame is full, then send the frame to the batcher
            // TODO: Set daedline
            if (msvc_concat.currIndex == 0) {
                msvc_OutQueue[0]->emplace(outReq);
                spdlog::get("container_agent")->trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name);
            }

            // if (msvc_concat.currIndex == 0) {
            //     saveGPUAsImg(bufferData.back().data, "concatBuffer.jpg");
            // }
        }

        /**
         * @brief ONLY IN PROFILING MODE
         * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
         */
        if (checkProfileEnd(currReq.req_travelPath[0])) {
            spdlog::get("container_agent")->info("{0:s} is stopping profiling.", msvc_name);
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(
                    Request<LocalGPUReqDataType>{
                            {},
                            {},
                            {"STOP_PROFILING"},
                            0,
                            {},
                            {}
                    }
            );
            continue;
        }
    }
    msvc_logFile.close();
}

// inline void BasePreprocessor::executeBatch(BatchTimeType &genTime, RequestSLOType &slo, RequestPathType &path,
//                                   std::vector<RequestData<LocalGPUReqDataType>> &bufferData,
//                                   std::vector<RequestData<LocalGPUReqDataType>> &prevData) {
//     // if (time < oldestReqTime) {
//     //     oldestReqTime = time;
//     // }

//     // If true, copy the buffer data into the out queue
//     ClockType timeNow = std::chrono::high_resolution_clock::now();

//     // Moment of batching
//     // This is the FOURTH TIMESTAMP
//     for (auto &req_genTime: genTime) {
//         req_genTime.emplace_back(timeNow);
//     }

//     Request<LocalGPUReqDataType> outReq = {
//             genTime,
//             slo,
//             path,
//             msvc_onBufferBatchSize,
//             bufferData,
//             prevData
//     };

//     msvc_batchCount++;

//     spdlog::get("container_agent")->trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name,
//                                             msvc_onBufferBatchSize);
//     msvc_OutQueue[0]->emplace(outReq);
//     msvc_concat.currIndex = 0;
//     msvc_onBufferBatchSize = 0;
//     msvc_numsOnBufferReqs = 0;
//     genTime.clear();
//     path.clear();
//     slo.clear();
//     bufferData.clear();
//     prevData.clear();
//     oldestReqTime = std::chrono::high_resolution_clock::time_point::max();

//     // spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
//     // std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
// }

template<typename T>
bool BasePreprocessor::
validateRequest(Request<T> &req) {
    // Meaning the the timeout in pop() has been reached and no request was actually popped
    if (strcmp(req.req_travelPath[0].c_str(), "empty") == 0) {
        return false;
    } else if (strcmp(req.req_travelPath[0].c_str(), "WARMUP_COMPLETED") == 0) {
        spdlog::get("container_agent")->info("{0:s} received the signal that the warmup is completed.", msvc_name);
        return false;
    }

    // 5. The moment the request is received at the preprocessor (FIFTH_TIMESTAMP)
    req.req_origGenTime[0].emplace_back(std::chrono::high_resolution_clock::now());
    return true;
}

void BasePreprocessor::preprocessProfiling() {
    // msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    // // The time where the last request was generated.
    // ClockType lastReq_genTime;
    // // The time where the current incoming request was generated.
    // ClockType currReq_genTime;
    // // The time where the current incoming request arrives

    // // Batch reqs' gen time
    // BatchTimeType outBatch_genTime;

    // // Batch reqs' slos
    // RequestSLOType outBatch_slo;

    // // Batch reqs' paths
    // RequestPathType outReq_path;

    // // Buffer memory for each batch
    // std::vector<RequestData<LocalGPUReqDataType>> bufferData;

    // // // Data carried from upstream microservice to be processed at a downstream
    // std::vector<RequestData<LocalGPUReqDataType>> prevData;
    // RequestData<LocalGPUReqDataType> data;

    // // Incoming request
    // Request<LocalGPUReqDataType> currReq;

    // Request<LocalCPUReqDataType> currCPUReq;

    // // Request sent to a downstream microservice
    // Request<LocalGPUReqDataType> outReq;

    // // Batch size of current request
    // BatchSizeType currReq_batchSize;
    // spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name);
    // cv::cuda::Stream *preProcStream = nullptr;

    // auto timeNow = std::chrono::high_resolution_clock::now();
    // while (true) {
    //     // Allowing this thread to naturally come to an end
    //     if (STOP_THREADS) {
    //         spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
    //         break;
    //     } else if (PAUSE_THREADS) {
    //         if (RELOADING) {
    //             /**
    //              * @brief Opening a new log file
    //              * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
    //              * 
    //              */
    //             if (msvc_toReloadConfigs) {
    //                 loadConfigs(msvc_configs, true);
    //                 msvc_toReloadConfigs = false;
    //             }

    //             if (msvc_logFile.is_open()) {
    //                 msvc_logFile.close();
    //             }
    //             msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

    //             // delete preProcStream;
    //             setDevice();
    //             preProcStream = new cv::cuda::Stream();

    //             outBatch_genTime.clear();
    //             outReq_path.clear();
    //             outBatch_slo.clear();
    //             bufferData.clear();
    //             prevData.clear();

    //             RELOADING = false;
    //             READY = true;
    //             spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
    //         }
    //         //info("{0:s} is being PAUSED.", msvc_name);
    //         continue;
    //     }
    //     // Processing the next incoming request
    //     if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
    //         if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
    //             msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
    //             // spdlog::get("container_agent")->trace("{0:s} Set current active queue index to {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
    //         }
    //     }
    //     spdlog::get("container_agent")->trace("{0:s} Current active queue index {1:d}.", msvc_name,
    //                                           msvc_activeInQueueIndex.at(0));
    //     if (msvc_activeInQueueIndex.at(0) == 1) {
    //         currCPUReq = msvc_InQueue.at(0)->pop1();
    //         if (!validateRequest<LocalCPUReqDataType>(currCPUReq)) {
    //             continue;
    //         }
    //         currReq = uploadReq(currCPUReq);
    //     } else if (msvc_activeInQueueIndex.at(0) == 2) {
    //         currReq = msvc_InQueue.at(0)->pop2();
    //         if (!validateRequest<LocalGPUReqDataType>(currReq)) {
    //             continue;
    //         }
    //     }

    //     msvc_overallTotalReqCount++;

    //     // The generated time of this incoming request will be used to determine the rate with which the microservice should
    //     // check its incoming queue.
    //     currReq_genTime = currReq.req_origGenTime[0][0];
    //     if (msvc_overallTotalReqCount > 1) {
    //         updateReqRate(currReq_genTime);
    //     }

    //     currReq_batchSize = currReq.req_batchSize;

    //     spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}. In queue size is {2:d}.",
    //                                           msvc_name, currReq_batchSize, msvc_InQueue.at(0)->size());

    //     // Resize the incoming request image the padd with the grey color
    //     // The resize image will be copied into a reserved buffer


    //     prevData.emplace_back(currReq.req_data[0]);

    //     spdlog::get("container_agent")->trace("{0:s} resizing a frame of [{1:d}, {2:d}] -> [{3:d}, {4:d}]",
    //                                           msvc_name,
    //                                           currReq.req_data[0].data.rows,
    //                                           currReq.req_data[0].data.cols,
    //                                           (msvc_outReqShape.at(0))[0][1],
    //                                           (msvc_outReqShape.at(0))[0][2]
    //     );
    //     data.data = resizePadRightBottom(
    //             currReq.req_data[0].data,
    //             (msvc_outReqShape.at(0))[0][1],
    //             (msvc_outReqShape.at(0))[0][2],
    //             {128, 128, 128},
    //             *preProcStream,
    //             msvc_imgType,
    //             msvc_colorCvtType,
    //             msvc_resizeInterpolType
    //     );

    //     // data.data = cvtHWCToCHW(data.data, *preProcStream, msvc_imgType);

    //     // data.data = normalize(data.data, *preProcStream, msvc_subVals, msvc_divVals, msvc_imgNormScale);

    //     spdlog::get("container_agent")->trace("{0:s} finished resizing a frame", msvc_name);
    //     data.shape = RequestDataShapeType(
    //             {3, (msvc_outReqShape.at(0))[0][1], (msvc_outReqShape.at(0))[0][2]});
    //     bufferData.emplace_back(data);
    //     spdlog::get("container_agent")->trace("{0:s} put an image into buffer. Current batch size is {1:d} ", msvc_name,
    //                                           msvc_onBufferBatchSize);


    //     /**
    //      * @brief At the moment of batching we stick this time stamp into each request in the batch.
    //      * This lets us know how much each individual request has to wait and how much is the batched inference
    //      * time exactly.
    //      * 
    //      */
    //     outBatch_slo.emplace_back(currReq.req_e2eSLOLatency[0]);
    //     outReq_path.emplace_back(currReq.req_travelPath[0]);

    //     // Consider this the moment the request preprocessed and is waiting to be batched
    //     timeNow = std::chrono::high_resolution_clock::now();

    //     // Add the whole time vector of currReq to outReq
    //     currReq.req_origGenTime[0].emplace_back(timeNow); // THIRD_TIMESTAMP
    //     outBatch_genTime.emplace_back(currReq.req_origGenTime[0]);

    //     // Set the ideal batch size for this microservice using the signal from the receiver.
    //     // Only used during profiling time.
    //     msvc_idealBatchSize = getNumberAtIndex(currReq.req_travelPath[0], 1);

    //     msvc_onBufferBatchSize++;

    //     // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
    //     // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
    //     if (this->isTimeToBatch()) { // If true, copy the buffer data into the out queue

    //         // Moment of batching
    //         timeNow = std::chrono::high_resolution_clock::now();

    //         for (auto &req_genTime: outBatch_genTime) {
    //             req_genTime.emplace_back(timeNow); //FOURTH_TIMESTAMP
    //         }

    //         outReq = {
    //                 outBatch_genTime,
    //                 outBatch_slo,
    //                 outReq_path,
    //                 msvc_onBufferBatchSize,
    //                 bufferData,
    //                 prevData
    //         };
    //         spdlog::get("container_agent")->trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name,
    //                                               msvc_onBufferBatchSize);
    //         msvc_OutQueue[0]->emplace(outReq);
    //         msvc_onBufferBatchSize = 0;
    //         outBatch_genTime.clear();
    //         outReq_path.clear();
    //         outBatch_slo.clear();
    //         bufferData.clear();
    //         prevData.clear();
    //     }
    //     spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
    // }
    // msvc_logFile.close();
}
