#ifndef BASEPROCESSOR_H
#define BASEPROCESSOR_H

#include <microservice.h>
#include <opencv2/core/cuda.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thread>
#include <misc.h>
#include <trtengine.h>
#include <random>

typedef uint16_t BatchSizeType;
using namespace msvcconfigs;
using json = nlohmann::ordered_json;


inline uint64_t getNumberAtIndex(const std::string& str, int index);
inline std::string getTimeDifString(const ClockType &start, const ClockType &end) {
    auto duration = std::chrono::duration_cast<TimePrecisionType>(end - start);
    return std::to_string(duration.count());
}

inline cv::Scalar vectorToScalar(const std::vector<float>& vec);

inline cv::cuda::GpuMat convertColor(
    const cv::cuda::GpuMat &input,
    uint8_t IMG_TYPE,
    uint8_t COLOR_CVT_TYPE,
    cv::cuda::Stream &stream
);

inline cv::cuda::GpuMat resizePadRightBottom(
    const cv::cuda::GpuMat &input,
    size_t height,
    size_t width,
    const std::vector<float>& bgcolor = {128, 128, 128},
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    uint8_t IMG_TYPE = 16, //CV_8UC3
    uint8_t COLOR_CVT_TYPE = 4, //CV_BGR2RGB
    uint8_t RESIZE_INTERPOL_TYPE = 3 //INTER_AREA
);

inline bool resizeIntoFrame(
    const cv::cuda::GpuMat &input,
    cv::cuda::GpuMat &frame,
    const uint16_t left,
    const uint16_t top,
    const uint16_t height,
    const uint16_t width,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    uint8_t IMG_TYPE = 16, //CV_8UC3
    uint8_t COLOR_CVT_TYPE = 4, //CV_BGR2RGB
    uint8_t RESIZE_INTERPOL_TYPE = 3 //INTER_AREA
);

inline cv::cuda::GpuMat normalize(
    const cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    const std::vector<float>& subVals = {0.f, 0.f, 0.f},
    const std::vector<float>& divVals = {1.f, 1.f, 1.f},
    const float normalized_scale = 1.f / 255.f
);

inline cv::cuda::GpuMat cvtHWCToCHW(
    const cv::cuda::GpuMat &input,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    uint8_t IMG_TYPE = 16 //CV_8UC3
);

/**
 * @brief 
 * 
 */
struct BaseReqBatcherConfigs : BaseMicroserviceConfigs{
    uint8_t msvc_imgType = 16; //CV_8UC3
    uint8_t msvc_colorCvtType = 4; //CV_BGR2RGB
    uint8_t msvc_resizeInterpolType = 3; //INTER_AREA
    float msvc_imgNormScale = 1.f / 255.f;
    std::vector<float> msvc_subVals = {0.f, 0.f, 0.f};
    std::vector<float> msvc_divVals = {1.f, 1.f, 1.f};
};

/**
 * @brief 
 * 
 */
struct BaseBatchInferencerConfigs : BaseMicroserviceConfigs {
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine;
};

struct BaseBBoxCropperConfigs : BaseMicroserviceConfigs {
    RequestShapeType msvc_inferenceShape;
};

struct BaseBBoxCropperVerifierConfigs : BaseBBoxCropperConfigs {
};

struct BaseKPointExtractorConfigs : BaseMicroserviceConfigs {
    RequestShapeType msvc_inferenceShape;
};

struct BaseClassifierConfigs : BaseMicroserviceConfigs {
    uint16_t msvc_numClasses;
};

struct ConcatDims {
    int32_t x1, y1, width, height;
};

struct ConcatConfigs {
    uint8_t numImgs = 1;
    uint8_t currIndex = 0;

    std::vector<ConcatDims> concatDims;
};


void concatConfigsGenerator(
    const RequestShapeType &inferenceShapes,
    ConcatConfigs &concat,
    const uint8_t padding = 0
);



class BaseReqBatcher : public Microservice {
public:
    BaseReqBatcher(const json &jsonConfigs);
    ~BaseReqBatcher() = default;

    BaseReqBatcher(const BaseReqBatcher &other);

    virtual void batchRequests();
    virtual void batchRequestsProfiling();
    inline void executeBatch(BatchTimeType &genTime, RequestSLOType &slo, RequestPathType &path,
                      std::vector<RequestData<LocalGPUReqDataType>> &buffer,
                      std::vector<RequestData<LocalGPUReqDataType>> &prev);
    
    virtual void updateCycleTiming();

    void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread batcher(&BaseReqBatcher::batchRequestsProfiling, this);
            batcher.detach();
            return;
        }
        spdlog::get("container_agent")->trace("{0:s} dispatching batching thread.", __func__);
        std::thread batcher(&BaseReqBatcher::batchRequests, this);
        batcher.detach();
    }

    BaseReqBatcherConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;

    bool readModelProfile(const json &profile);

protected:
    // number of concatentated and ready to be batched requests
    BatchSizeType msvc_onBufferReadyBatchSize = 0;
    // total number of requests that have been put into a concatenated request in this batch
    BatchSizeType msvc_numsOnBufferReqs = 0;
    std::vector<cv::cuda::GpuMat> msvc_batchBuffer;
    inline bool isTimeToBatch() override;
    template <typename T>
    bool validateRequest(Request<T> &req);

    uint8_t msvc_imgType, msvc_colorCvtType, msvc_resizeInterpolType;
    float msvc_imgNormScale;
    std::vector<float> msvc_subVals, msvc_divVals;
    BatchInferProfileListType msvc_batchInferProfileList;
    ClockType oldestReqTime;
    // This is the time calculated by the ideal schedule
    // It will be calculated assuming the batch is filled with the ideal batch size
    // and the requests come in at the ideal rate
    ClockType msvc_nextIdealBatchTime;
    // This is the time calculated by the actual schedule to for oldest req in the batch
    // to be processed on time
    ClockType msvc_nextMustBatchTime;
    uint64_t timeout = 100000; //microseconds

    ConcatConfigs msvc_concat;
    cv::cuda::GpuMat msvc_concatBuffer;
};


typedef uint16_t BatchSizeType;

class BaseBatchInferencer : public Microservice {
public:
    BaseBatchInferencer(const json &jsonConfigs);
    ~BaseBatchInferencer() = default;
    virtual void inference();
    virtual void inferenceProfiling();

    RequestShapeType getInputShapeVector();
    RequestShapeType getOutputShapeVector();

    void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread inferencer(&BaseBatchInferencer::inferenceProfiling, this);
            inferencer.detach();
            return;
        }
        spdlog::get("container_agent")->trace("{0:s} dispatching inference thread.", __func__);
        std::thread inferencer(&BaseBatchInferencer::inference, this);
        inferencer.detach();
    }

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
    virtual std::string getModelName() override {
        return msvc_inferenceEngine->getEngineName();
    }
protected:
    BatchSizeType msvc_onBufferBatchSize;
    std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
    TRTConfigs msvc_engineConfigs;
    Engine* msvc_inferenceEngine = nullptr;
};

/**
 * @brief crop from input image all bounding boxes whose coordinates are provided by `bbox_coorList`
 * 
 * @param image 
 * @param infer_h the height of the image during inference, used to scale to bounding boxes to their original coordinates for cropping
 * @param infer_w the height of the image during inference, used to scale to bounding boxes to their original coordinates for cropping
 * @param bbox_coorList pointer to a 2d `float` array of bounding box coordinates of size (TopK, 4). The box format is 
 *                      [x1, y1, x2, y2] (e.g., [0, 266, 260, 447])
 * @return cv::cuda::GpuMat
 */
inline std::vector<uint8_t> crop(
    const std::vector<cv::cuda::GpuMat> &images,
    const std::vector<ConcatDims> &concatDims,
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    uint16_t numDetections,
    const float *bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
);

inline void cropOneBox(
    const cv::cuda::GpuMat &image,
    int infer_h,
    int infer_w,
    int numDetections,
    const float *bbox_coorList,
    cv::cuda::GpuMat &croppedBBoxes
);

class BasePostprocessor : public Microservice {
public:
    BasePostprocessor(const json &jsonConfigs) : Microservice(jsonConfigs) {
        loadConfigs(jsonConfigs, true);
    };
    ~BasePostprocessor() = default;

    BasePostprocessor(const BasePostprocessor &other) : Microservice(other) {
        std::lock(msvc_overallMutex, other.msvc_overallMutex);
        std::lock_guard<std::mutex> lockThis(msvc_overallMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lockOther(other.msvc_overallMutex, std::adopt_lock);

        msvc_inferenceShape = other.msvc_inferenceShape;
        msvc_concat = other.msvc_concat;
    };

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override {
        if (!isConstructing) {
            Microservice::loadConfigs(jsonConfigs, isConstructing);
        }
        msvc_processRecords.setKeepLength((uint64_t)jsonConfigs.at("cont_metricsScrapeIntervalMillisec") * 2);
        msvc_arrivalRecords.setKeepLength((uint64_t) jsonConfigs.at("cont_metricsScrapeIntervalMillisec") * 2);

        msvc_concat.numImgs = jsonConfigs["msvc_concat"];
    };
    virtual ProcessRecordType getProcessRecords()  {
        return msvc_processRecords.getRecords();
    }

    virtual void getProcessRecords(ProcessRecordType &overallRecords) {
        msvc_processRecords.getRecords(overallRecords);
    }

    virtual BatchInferRecordType getBatchInferRecords() {
        return msvc_processRecords.getBatchInferRecords();
    }

    virtual void getBatchInferRecords(BatchInferRecordType &overallRecords) {
        msvc_processRecords.getBatchInferRecords(overallRecords);
    }

    virtual ArrivalRecordType getArrivalRecords() {
        return msvc_arrivalRecords.getRecords();
    }

    virtual void getArrivalRecords(ArrivalRecordType &overallRecords) {
        msvc_arrivalRecords.getRecords(overallRecords);
    }

    virtual void addToPath(RequestPathType &path, uint64_t reqNum) {
        
    }

    /**
     * @brief Get the size of the arrival package which is recorded in the travel path
     * 
     * @param path 
     * @return RequestSizeType 
     */
    RequestMemSizeType getArrivalPkgSize(const std::string& path) {
        // Path looks like this
        // [hostDeviceName|microserviceID|inReqNumber|totalNumberOfOutputs|NumberInOutputs|outPackageSize (in byte)]
        // [edge|YOLOv5_01|05|05][server|retinaface_02|09|09]
        std::string temp = splitString(path, "[").back();
        temp = splitString(temp, "]").front();
        return std::stoul(splitString(temp, "|").back());
    }

    inline cv::Mat encodeResults(const cv::Mat &image) {
        std::vector<uchar> buf;
        cv::imencode(".jpg", image, buf, {cv::IMWRITE_JPEG_QUALITY, 80});
        RequestMemSizeType encodedMemSize = buf.size();
        cv::Mat encoded(1, encodedMemSize, CV_8UC1, buf.data());
        return encoded.clone();
    }

protected:
    ProcessReqRecords msvc_processRecords;
    // Record
    ArrivalReqRecords msvc_arrivalRecords;

    struct PerQueueOutRequest {
        bool used = false;
        uint32_t totalSize = 0;
        uint32_t totalEncodedSize = 0;
        Request<LocalCPUReqDataType> cpuReq;
        Request<LocalGPUReqDataType> gpuReq;
    };

    RequestShapeType msvc_inferenceShape;

    ConcatConfigs msvc_concat;
};

class BaseBBoxCropper : public BasePostprocessor {
public:
    BaseBBoxCropper(const json &jsonConfigs);
    ~BaseBBoxCropper() = default;

    BaseBBoxCropper(const BaseBBoxCropper &other) : BasePostprocessor(other) {};

    void cropping();

    void generateRandomBBox(
        float *bboxList,
        const uint16_t height,
        const uint16_t width,
        const uint16_t numBboxes,
        const uint16_t seed = 2024
    );

    void cropProfiling();

    void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread postprocessor(&BaseBBoxCropper::cropProfiling, this);
            postprocessor.detach();
            return;
        }
        spdlog::get("container_agent")->trace("{0:s} dispatching cropping thread.", __func__);
        std::thread postprocessor(&BaseBBoxCropper::cropping, this);
        postprocessor.detach();
    }

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
};

class BaseBBoxCropperAugmentation : public BasePostprocessor {
public:
    BaseBBoxCropperAugmentation(const json &jsonConfigs);
    ~BaseBBoxCropperAugmentation() = default;

    BaseBBoxCropperAugmentation(const BaseBBoxCropperAugmentation &other) : BasePostprocessor(other) {};

    void cropping();

    void generateRandomBBox(
            float *bboxList,
            const uint16_t height,
            const uint16_t width,
            const uint16_t numBboxes,
            const uint16_t seed = 2024
    );

    void cropProfiling();

    void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread postprocessor(&BaseBBoxCropperAugmentation::cropProfiling, this);
            postprocessor.detach();
            return;
        }
        spdlog::get("container_agent")->trace("{0:s} dispatching cropping thread.", __func__);
        std::thread postprocessor(&BaseBBoxCropperAugmentation::cropping, this);
        postprocessor.detach();
    }

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
};

class BaseBBoxCropperVerifier : public BasePostprocessor {
public:
    BaseBBoxCropperVerifier(const json& jsonConfigs);
    ~BaseBBoxCropperVerifier() = default;

    void cropping();

    void cropProfiling();

    virtual void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread postprocessor(&BaseBBoxCropperVerifier::cropProfiling, this);
            postprocessor.detach();
            return;
        }
        std::thread postprocessor(&BaseBBoxCropperVerifier::cropping, this);
        postprocessor.detach();
    }

    BaseBBoxCropperVerifierConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
};

class BaseClassifier : public BasePostprocessor {
public:
    BaseClassifier(const json &jsonConfigs);
    ~BaseClassifier() = default;

    BaseClassifier(const BaseClassifier &other) : BasePostprocessor(other) {
        std::lock(msvc_overallMutex, other.msvc_overallMutex);
        std::lock_guard<std::mutex> lockThis(msvc_overallMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lockOther(other.msvc_overallMutex, std::adopt_lock);

        msvc_numClasses = other.msvc_numClasses;
    };

    virtual void classify();

    virtual void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread classifier(&BaseClassifier::classifyProfiling, this);
            classifier.detach();
            return;
        }
        std::thread classifier(&BaseClassifier::classify, this);
        classifier.detach();
    }

    virtual void classifyProfiling();

    BaseClassifierConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;

protected:
    uint16_t msvc_numClasses;
};

class BaseSoftmaxClassifier : public BaseClassifier {
public:
    BaseSoftmaxClassifier(const json &jsonConfigs);
    ~BaseSoftmaxClassifier() = default;

    BaseSoftmaxClassifier(const BaseSoftmaxClassifier &other) : BaseClassifier(other) {};

    virtual void classify() override;
    virtual void classifyProfiling() override;
};

class BaseKPointExtractor : public BasePostprocessor {
public:
    BaseKPointExtractor(const json &jsonConfigs);
    ~BaseKPointExtractor() = default;

    BaseKPointExtractor(const BaseKPointExtractor &other) : BasePostprocessor(other) {};

    virtual void extractor();

    virtual void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread extractor(&BaseKPointExtractor::extractorProfiling, this);
            extractor.detach();
            return;
        }
        std::thread extractor(&BaseKPointExtractor::extractor, this);
        extractor.detach();
    }

    virtual void extractorProfiling();

    BaseKPointExtractorConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
};

class BaseSink : public Microservice {
public:
    BaseSink(const json &jsonConfigs);
    ~BaseSink() = default;

    virtual void sink();

    virtual void dispatchThread() override {
        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} dispatching profiling thread.", __func__);
            std::thread sinker(&BaseSink::sink, this);
            sinker.detach();
            return;
        }
        std::thread sinker(&BaseSink::sink, this);
        sinker.detach();
        return;
    }

    BaseMicroserviceConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;
};

#endif //BASEPROCESSOR_H