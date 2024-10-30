#include "data_reader.h"

inline cv::Mat resizePadRightBottom(
        const cv::Mat &input,
        const size_t height,
        const size_t width,
        const std::vector<float> &bgcolor,
        uint8_t RESIZE_INTERPOL_TYPE
) {
    spdlog::get("container_agent")->trace("Going into {0:s}", __func__);

    float r = std::min((float) width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    // Create a new Mat
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(unpad_w, unpad_h), 0, 0, RESIZE_INTERPOL_TYPE);
    cv::Mat out = cv::Mat(height, width, input.type(), cv::Scalar(bgcolor[0], bgcolor[1], bgcolor[2]));
    // Copy resized image to output Mat
    resized.copyTo(out(cv::Rect(0, 0, resized.cols, resized.rows)));

    spdlog::get("container_agent")->trace("Finished {0:s}", __func__);

    return out;
}

DataReader::DataReader(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", __func__);
};

PerSecondArrivalRecord DataReader::getPerSecondArrivalRecord() {
    PerSecondArrivalRecord perSecondArrivalRecord;
    perSecondArrivalRecord.numRequests = target_fps;
    perSecondArrivalRecord.interArrivalMean = 1.0 / target_fps;
    perSecondArrivalRecord.interArrivalVariance = 0.0;
    return perSecondArrivalRecord;
}

void DataReader::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    if (!isConstructing) {
        Microservice::loadConfigs(jsonConfigs);
    }

    link = jsonConfigs.at("msvc_upstreamMicroservices")[0].at("nb_link")[0];
    source = cv::VideoCapture(link);
    msvc_currFrameID = 0;
    target_fps = jsonConfigs.at("msvc_idealBatchSize").get<int>();
    wait_time_ms = 1000 / target_fps;
    skipRatio = 30.f / target_fps;
    link = link.substr(link.find_last_of('/') + 1);
    msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
};

cv::Mat encodeImage(const cv::Mat &image) {
    std::vector<uchar> buf;
    cv::imencode(".jpg", image, buf, {cv::IMWRITE_JPEG_QUALITY, 95});
    RequestMemSizeType encodedMemSize = buf.size();
    cv::Mat encoded(1, encodedMemSize, CV_8UC1, buf.data());
    return encoded.clone();
}

void DataReader::Process() {
    int frameCount = 0;
    uint16_t readFrames = 0;
    while (true) {
        ClockType start = std::chrono::system_clock::now();
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                RELOADING = false;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            source.set(cv::CAP_PROP_POS_FRAMES, msvc_currFrameID);
            continue;
        }
        cv::Mat frame;
        if (!source.read(frame)) {
            if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
                spdlog::get("container_agent")->info("No more frames to read, exiting Video Processing.");
                stopThread();
                continue;
            }
            spdlog::get("container_agent")->info("Resetting Video Processing.");
            source.set(cv::CAP_PROP_POS_FRAMES, 0);
            source >> frame;
            frameCount = 0;
            readFrames = 0;
        }
        if (std::fmod(frameCount, skipRatio) < 1) {
            readFrames++;
            msvc_currFrameID = (int) source.get(cv::CAP_PROP_POS_FRAMES);
            if (msvc_dataShape[0][1] != -1 && msvc_dataShape[0][2] != -1) {
                frame = resizePadRightBottom(frame, msvc_dataShape[0][1], msvc_dataShape[0][2],
                                             {128, 128, 128}, cv::INTER_AREA);
            }
            RequestMemSizeType frameMemSize = frame.channels() * frame.rows * frame.cols * CV_ELEM_SIZE1(frame.type());
            for (auto q: msvc_OutQueue) {
                Request<LocalCPUReqDataType> req;
                if (!q->getEncoded()) {
                    // 1. The very moment request is originally generated at the beggining of the pipeline. (FIRST_TIMESTAMP)
                    // This timestamp will remain throughout the lifetime of the request
                    ClockType time = std::chrono::system_clock::now();
                    req = {{{time, time}}, {msvc_contSLO},
                           {"[" + msvc_hostDevice + "|" + link + "|" + std::to_string(readFrames) +
                            "|1|1|" + std::to_string(frameMemSize) + "|" + std::to_string(frameMemSize) + "]"}, 1,
                           {RequestData<LocalCPUReqDataType>{{frame.dims, frame.rows, frame.cols}, frame}}}; 
                    q->emplace(req);
                    continue;
                }

                cv::Mat encoded = encodeImage(frame);
                RequestMemSizeType encodedMemSize = encoded.channels() * encoded.rows * encoded.cols * CV_ELEM_SIZE1(encoded.type());
                ClockType time = std::chrono::system_clock::now();
                req = {{{time, time}}, {msvc_contSLO},
                       {"[" + msvc_hostDevice + "|" + link + "|" + std::to_string(readFrames) +
                        "|1|1|" + std::to_string(frameMemSize) + "|" + std::to_string(encodedMemSize) + "]"}, 1,
                       {RequestData<LocalCPUReqDataType>{{frame.dims, frame.rows, frame.cols},
                                                         encoded}}};
                q->emplace(req);
            }
            // remove processing time from wait time
            ClockType end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms) - elapsed_seconds);
        }
        frameCount++;
    }
};