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

BaseReqBatcherConfigs BaseReqBatcher::loadConfigsFromJson(const json &jsonConfigs) {
    BaseReqBatcherConfigs configs;

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

void BaseReqBatcher::updateCycleTiming() {
    // The number of cycles since the beginning of  this scheduling round, which is chosen to be the start of the first cycle
    uint64_t numCyclesSince;
    if (msvc_localDutyCycle == 0) {
        numCyclesSince = 0;
    } else {
        numCyclesSince = std::chrono::duration_cast<TimePrecisionType>(
            std::chrono::high_resolution_clock::now() - msvc_cycleStartTime).count() / msvc_localDutyCycle;
    }

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
    timeout = 100000; //microseconds
}

bool BaseReqBatcher::readModelProfile(const json &profile) {
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

/**
 * @brief Load the configurations from the json file
 * 
 * @param jsonConfigs 
 * @param isConstructing
 * 
 * @note The function is called from the constructor or when the microservice is to be reloaded
 */
void BaseReqBatcher::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    // Load the configs from the json file for Microservice class
    if (!isConstructing) { // If the function is not called from the constructor
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }

    BaseReqBatcherConfigs configs = loadConfigsFromJson(jsonConfigs);

    bool readProfile = readModelProfile(jsonConfigs["msvc_modelProfile"]);

    if (!readProfile && msvc_RUNMODE == RUNMODE::DEPLOYMENT && msvc_taskName != "dsrc" && msvc_taskName != "datasource") {
        spdlog::get("container_agent")->error("{0:s} No model profile found.", __func__);
        exit(1);
    }

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
BaseReqBatcher::BaseReqBatcher(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs);
    //set to maximum value
    oldestReqTime = std::chrono::high_resolution_clock::time_point::max();
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name);
}

BaseReqBatcher::BaseReqBatcher(const BaseReqBatcher &other) : Microservice(other) {
    std::lock(msvc_overallMutex, other.msvc_overallMutex);
    std::lock_guard<std::mutex> lock1(msvc_overallMutex, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(other.msvc_overallMutex, std::adopt_lock);

    msvc_imgType = other.msvc_imgType;
    msvc_colorCvtType = other.msvc_colorCvtType;
    msvc_resizeInterpolType = other.msvc_resizeInterpolType;
    msvc_imgNormScale = other.msvc_imgNormScale;
    msvc_subVals = other.msvc_subVals;
    msvc_divVals = other.msvc_divVals;
    msvc_batchInferProfileList = other.msvc_batchInferProfileList;
    oldestReqTime = other.oldestReqTime;
    msvc_toReloadConfigs = other.msvc_toReloadConfigs;
}


/**
 * @brief 
 * 
 */
void BaseReqBatcher::batchRequests() {
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives
    ClockType currReq_recvTime;

    // Batch reqs' gen time
    BatchTimeType outBatch_genTime;

    // out request's gen time
    RequestTimeType outReq_genTime;

    // Batch reqs' slos
    RequestSLOType outBatch_slo;

    // Batch reqs' paths
    RequestPathType outBatch_path;

    // Buffer memory for each batch
    std::vector<RequestData<LocalGPUReqDataType>> bufferData;

    // // Data carried from upstream microservice to be processed at a downstream
    std::vector<RequestData<LocalGPUReqDataType>> prevData;
    RequestData<LocalGPUReqDataType> data;

    // Incoming request
    Request<LocalGPUReqDataType> currReq;

    Request<LocalCPUReqDataType> currCPUReq;

    auto timeNow = std::chrono::high_resolution_clock::now();

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name);
    cv::cuda::Stream *preProcStream = nullptr;
    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

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

                outBatch_genTime.clear();
                outBatch_path.clear();
                outBatch_slo.clear();
                bufferData.clear();
                prevData.clear();

                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
                RELOADING = false;
                READY = true;
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        // if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
        //     if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
        //         msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
        //         spdlog::get("container_agent")->trace("{0:s} Set current active queue index to {1:d}.", msvc_name,
        //                                               msvc_activeInQueueIndex.at(0));
        //     }
        //     spdlog::get("container_agent")->trace("{0:s} Current active queue index {1:d}.", msvc_name,
        //                                           msvc_activeInQueueIndex.at(0));
        // }
        if (isTimeToBatch()) {
            executeBatch(outBatch_genTime, outBatch_slo, outBatch_path, bufferData, prevData);
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        // if (msvc_activeInQueueIndex.at(0) == 1) {
        //     currCPUReq = msvc_InQueue.at(0)->pop1(timeout);
        //     if (!validateRequest<LocalCPUReqDataType>(currCPUReq)) {
        //         continue;
        //     }
        //     currReq = uploadReq(currCPUReq);
        // } else if (msvc_activeInQueueIndex.at(0) == 2) {
        //     currReq = msvc_InQueue.at(0)->pop2(timeout);
        //     if (!validateRequest<LocalGPUReqDataType>(currReq)) {
        //         continue;
        //     }
        // }
        currCPUReq = msvc_InQueue.at(0)->pop1(timeout);
        if (!validateRequest<LocalCPUReqDataType>(currCPUReq)) {
            continue;
        }
        currReq = uploadReq(currCPUReq);
        // even if a valid request is not popped, if it's time to batch, we should batch the requests
        // as it doesn't take much time and otherwise, we are running the risk of the whole batch being late.
        // if (isTimeToBatch()) {
        //     executeBatch(outBatch_genTime, outBatch_slo, outBatch_path, bufferData, prevData);
        // }

        if (msvc_numsOnBufferReqs == 0) {
            oldestReqTime = startTime;
            // We update the oldest request time and the must batch time for this request
            // We try to account for its inference and postprocessing time
            uint64_t timeOutByLastReq = msvc_contSLO - 
                            (msvc_batchInferProfileList.at(1).p95inferLat +
                            msvc_batchInferProfileList.at(1).p95postLat) * 1.2;
            msvc_nextMustBatchTime = oldestReqTime + TimePrecisionType(timeOutByLastReq);
        }

        msvc_overallTotalReqCount++;

        // After the communication-related timestamps have been kept in the arrival record, all except the very first one (genTime) are removed.
        // The first timestamp will be carried till the end of the pipeline to determine the total latency and if the request is late, along the way.
        // outReq_genTime = {currReq_genTime, std::chrono::high_resolution_clock::now()};
        currReq.req_origGenTime[0].emplace_back(std::chrono::high_resolution_clock::now());

        currReq_batchSize = currReq.req_batchSize;

        outBatch_slo.emplace_back(currReq.req_e2eSLOLatency[0]);
        outBatch_path.emplace_back(currReq.req_travelPath[0] + "[" + msvc_hostDevice + "|" + msvc_containerName + "|" +
                                   std::to_string(msvc_overallTotalReqCount) + "|" + msvc_name + "]");
        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}. In queue size is {2:d}.",
                                              msvc_name, currReq_batchSize, msvc_InQueue.at(0)->size());

        msvc_numsOnBufferReqs++;
        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer


        prevData.emplace_back(currReq.req_data[0]);

        data.data = convertColor(
                currReq.req_data[0].data,
                msvc_imgType,
                msvc_colorCvtType,
                *preProcStream
        );

        if (msvc_concat.currIndex == 0) {
            // Create a new frame to hold the concatenated images
            msvc_concatBuffer = cv::cuda::GpuMat(
                (msvc_outReqShape.at(0))[0][1],
                (msvc_outReqShape.at(0))[0][2],
                msvc_imgType, cv::Scalar(128, 128, 128)
            );
        }

        bool success = resizeIntoFrame(
            data.data,
            msvc_concatBuffer,
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
            spdlog::get("container_agent")->trace("{0:s} resized an image of [{1:d}, {2:d}] -> [{3:d}, {4:d}] and put into frame at index {5:d}.",
                msvc_name,
                currReq.req_data[0].data.rows,
                currReq.req_data[0].data.cols,
                (this->msvc_outReqShape.at(0))[0][1],
                (this->msvc_outReqShape.at(0))[0][2],
                msvc_concat.currIndex
            );

            // Consider this the moment the request preprocessed and is waiting to be batched
            timeNow = std::chrono::high_resolution_clock::now();

            // Add the whole time vector of currReq to outReq
            // outReq_genTime.emplace_back(timeNow);
            currReq.req_origGenTime[0].emplace_back(timeNow);
            outBatch_genTime.emplace_back(currReq.req_origGenTime[0]);
            if (msvc_concat.currIndex == 0) {
                msvc_onBufferReadyBatchSize++;
                data.data = msvc_concatBuffer;
                data.shape = RequestDataShapeType({(msvc_outReqShape.at(0))[0][0], (msvc_outReqShape.at(0))[0][1],
                                                   (msvc_outReqShape.at(0))[0][2]});
                bufferData.emplace_back(data);
            }
            msvc_concat.currIndex = (++msvc_concat.currIndex % msvc_concat.numImgs);

            if (msvc_concat.currIndex == 0) {
                saveGPUAsImg(bufferData.back().data, "concatBuffer.jpg");
            }
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

inline void BaseReqBatcher::executeBatch(BatchTimeType &genTime, RequestSLOType &slo, RequestPathType &path,
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
            msvc_onBufferReadyBatchSize,
            bufferData,
            prevData
    };

    msvc_miniBatchCount++;
    msvc_batchCount++;

    spdlog::get("container_agent")->trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name,
                                            msvc_onBufferReadyBatchSize);
    msvc_OutQueue[0]->emplace(outReq);
    msvc_avgBatchSize += (msvc_onBufferReadyBatchSize - msvc_avgBatchSize) / msvc_miniBatchCount;
    msvc_concat.currIndex = 0;
    msvc_onBufferReadyBatchSize = 0;
    msvc_numsOnBufferReqs = 0;
    genTime.clear();
    path.clear();
    slo.clear();
    bufferData.clear();
    prevData.clear();
    oldestReqTime = std::chrono::high_resolution_clock::time_point::max();

    // spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
    // std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
}

template<typename T>
bool BaseReqBatcher::
validateRequest(Request<T> &req) {
    // Meaning the the timeout in pop() has been reached and no request was actually popped
    if (strcmp(req.req_travelPath[0].c_str(), "empty") == 0) {
        return false;
    } else if (strcmp(req.req_travelPath[0].c_str(), "WARMUP_COMPLETED") == 0) {
        spdlog::get("container_agent")->info("{0:s} received the signal that the warmup is completed.", msvc_name);
        return false;
    }

    req.req_origGenTime[0].emplace_back(std::chrono::high_resolution_clock::now());
    return true;
}

void BaseReqBatcher::batchRequestsProfiling() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    // The time where the last request was generated.
    ClockType lastReq_genTime;
    // The time where the current incoming request was generated.
    ClockType currReq_genTime;
    // The time where the current incoming request arrives

    // Batch reqs' gen time
    BatchTimeType outBatch_genTime;

    // Batch reqs' slos
    RequestSLOType outBatch_slo;

    // Batch reqs' paths
    RequestPathType outReq_path;

    // Buffer memory for each batch
    std::vector<RequestData<LocalGPUReqDataType>> bufferData;

    // // Data carried from upstream microservice to be processed at a downstream
    std::vector<RequestData<LocalGPUReqDataType>> prevData;
    RequestData<LocalGPUReqDataType> data;

    // Incoming request
    Request<LocalGPUReqDataType> currReq;

    Request<LocalCPUReqDataType> currCPUReq;

    // Request sent to a downstream microservice
    Request<LocalGPUReqDataType> outReq;

    // Batch size of current request
    BatchSizeType currReq_batchSize;
    spdlog::get("container_agent")->info("{0:s} STARTS.", msvc_name);
    cv::cuda::Stream *preProcStream = nullptr;

    auto timeNow = std::chrono::high_resolution_clock::now();
    while (true) {
        // Allowing this thread to naturally come to an end
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */
                if (msvc_toReloadConfigs) {
                    loadConfigs(msvc_configs, true);
                    msvc_toReloadConfigs = false;
                }

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                // delete preProcStream;
                setDevice();
                preProcStream = new cv::cuda::Stream();

                outBatch_genTime.clear();
                outReq_path.clear();
                outBatch_slo.clear();
                bufferData.clear();
                prevData.clear();

                RELOADING = false;
                READY = true;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            //info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        // Processing the next incoming request
        if (msvc_InQueue.at(0)->getActiveQueueIndex() != msvc_activeInQueueIndex.at(0)) {
            if (msvc_InQueue.at(0)->size(msvc_activeInQueueIndex.at(0)) == 0) {
                msvc_activeInQueueIndex.at(0) = msvc_InQueue.at(0)->getActiveQueueIndex();
                // spdlog::get("container_agent")->trace("{0:s} Set current active queue index to {1:d}.", msvc_name, msvc_activeInQueueIndex.at(0));
            }
        }
        spdlog::get("container_agent")->trace("{0:s} Current active queue index {1:d}.", msvc_name,
                                              msvc_activeInQueueIndex.at(0));
        if (msvc_activeInQueueIndex.at(0) == 1) {
            currCPUReq = msvc_InQueue.at(0)->pop1();
            if (!validateRequest<LocalCPUReqDataType>(currCPUReq)) {
                continue;
            }
            currReq = uploadReq(currCPUReq);
        } else if (msvc_activeInQueueIndex.at(0) == 2) {
            currReq = msvc_InQueue.at(0)->pop2();
            if (!validateRequest<LocalGPUReqDataType>(currReq)) {
                continue;
            }
        }

        msvc_overallTotalReqCount++;

        // The generated time of this incoming request will be used to determine the rate with which the microservice should
        // check its incoming queue.
        currReq_genTime = currReq.req_origGenTime[0][0];
        if (msvc_overallTotalReqCount > 1) {
            updateReqRate(currReq_genTime);
        }

        currReq_batchSize = currReq.req_batchSize;

        spdlog::get("container_agent")->trace("{0:s} popped a request of batch size {1:d}. In queue size is {2:d}.",
                                              msvc_name, currReq_batchSize, msvc_InQueue.at(0)->size());

        // Resize the incoming request image the padd with the grey color
        // The resize image will be copied into a reserved buffer


        prevData.emplace_back(currReq.req_data[0]);

        spdlog::get("container_agent")->trace("{0:s} resizing a frame of [{1:d}, {2:d}] -> [{3:d}, {4:d}]",
                                              msvc_name,
                                              currReq.req_data[0].data.rows,
                                              currReq.req_data[0].data.cols,
                                              (msvc_outReqShape.at(0))[0][1],
                                              (msvc_outReqShape.at(0))[0][2]
        );
        data.data = resizePadRightBottom(
                currReq.req_data[0].data,
                (msvc_outReqShape.at(0))[0][1],
                (msvc_outReqShape.at(0))[0][2],
                {128, 128, 128},
                *preProcStream,
                msvc_imgType,
                msvc_colorCvtType,
                msvc_resizeInterpolType
        );

        // data.data = cvtHWCToCHW(data.data, *preProcStream, msvc_imgType);

        // data.data = normalize(data.data, *preProcStream, msvc_subVals, msvc_divVals, msvc_imgNormScale);

        spdlog::get("container_agent")->trace("{0:s} finished resizing a frame", msvc_name);
        data.shape = RequestDataShapeType(
                {3, (msvc_outReqShape.at(0))[0][1], (msvc_outReqShape.at(0))[0][2]});
        bufferData.emplace_back(data);
        spdlog::get("container_agent")->trace("{0:s} put an image into buffer. Current batch size is {1:d} ", msvc_name,
                                              msvc_onBufferReadyBatchSize);


        /**
         * @brief At the moment of batching we stick this time stamp into each request in the batch.
         * This lets us know how much each individual request has to wait and how much is the batched inference
         * time exactly.
         * 
         */
        outBatch_slo.emplace_back(currReq.req_e2eSLOLatency[0]);
        outReq_path.emplace_back(currReq.req_travelPath[0]);

        // Consider this the moment the request preprocessed and is waiting to be batched
        timeNow = std::chrono::high_resolution_clock::now();

        // Add the whole time vector of currReq to outReq
        currReq.req_origGenTime[0].emplace_back(timeNow); // THIRD_TIMESTAMP
        outBatch_genTime.emplace_back(currReq.req_origGenTime[0]);

        // Set the ideal batch size for this microservice using the signal from the receiver.
        // Only used during profiling time.
        msvc_idealBatchSize = getNumberAtIndex(currReq.req_travelPath[0], 1);

        msvc_onBufferReadyBatchSize++;

        // First we need to decide if this is an appropriate time to batch the buffered data or if we can wait a little more.
        // Waiting more means there is a higher chance the earliest request in the buffer will be late eventually.
        if (this->isTimeToBatch()) { // If true, copy the buffer data into the out queue

            // Moment of batching
            timeNow = std::chrono::high_resolution_clock::now();

            for (auto &req_genTime: outBatch_genTime) {
                req_genTime.emplace_back(timeNow); //FOURTH_TIMESTAMP
            }

            outReq = {
                    outBatch_genTime,
                    outBatch_slo,
                    outReq_path,
                    msvc_onBufferReadyBatchSize,
                    bufferData,
                    prevData
            };
            spdlog::get("container_agent")->trace("{0:s} emplaced a request of batch size {1:d} ", msvc_name,
                                                  msvc_onBufferReadyBatchSize);
            msvc_OutQueue[0]->emplace(outReq);
            msvc_onBufferReadyBatchSize = 0;
            outBatch_genTime.clear();
            outReq_path.clear();
            outBatch_slo.clear();
            bufferData.clear();
            prevData.clear();
        }
        spdlog::get("container_agent")->trace("{0:s} sleeps for {1:d} millisecond", msvc_name, msvc_interReqTime);
        std::this_thread::sleep_for(std::chrono::milliseconds(msvc_interReqTime));
    }
    msvc_logFile.close();
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
inline bool BaseReqBatcher::isTimeToBatch() {
    // timeout for the pop() function
    timeout = 100000;
    if ((msvc_RUNMODE == RUNMODE::PROFILING || 
         msvc_BATCH_MODE == BATCH_MODE::FIXED) && 
        msvc_numsOnBufferReqs >= msvc_idealBatchSize * msvc_concat.numImgs) {
        return true;
    }

    // OURS BATCH MODE
    if (msvc_BATCH_MODE != BATCH_MODE::OURS) {
        return false;
    }
    //First of all, whenever the batch is full, then it's time to batch
    if (msvc_onBufferReadyBatchSize == 0) {
        return false;
    // If the batch is empty, then it doesn't really matter if it's time to batch or not
    } else if (msvc_numsOnBufferReqs >= msvc_idealBatchSize * msvc_concat.numImgs) {
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
    // Since this calculation is before the preprocessing in the batcher function, we add one preprocessing time unit
    // into the total reserved time for the requests already in batch.
    // If this preprocessing doesnt happen (as the next request doesn't come as expectd), then the batcher will just batch
    // the next time as this timer is expired
    uint64_t timeOutByLastReq = msvc_contSLO - lastReqWaitTime - 
                            (msvc_batchInferProfileList.at(msvc_onBufferReadyBatchSize).p95inferLat +
                            msvc_batchInferProfileList.at(msvc_onBufferReadyBatchSize).p95postLat) * msvc_onBufferReadyBatchSize * 1.2 -
                            msvc_batchInferProfileList.at(msvc_onBufferReadyBatchSize).p95prepLat * 1.2;
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
