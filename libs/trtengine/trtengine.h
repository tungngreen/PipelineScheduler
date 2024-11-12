#ifndef TRTENGINE_H
#define TRTENGINE_H

#include <fstream>
#include <chrono>
#include <misc.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <NvOnnxParser.h>

// Utility methods
namespace Util {
    inline bool doesFileExist(const std::string& filepath) {
        std::ifstream f(filepath.c_str());
        return f.good();
    }

    std::vector<std::string> getFilesInDirectory(const std::string& dirPath);
}

using namespace nvinfer1;
using namespace Util;
using trt::TRTConfigs;

// Precision used for GPU inference
enum class Precision {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

/**
 * @brief Engine class for tensorrt model convert and inference
 * TODO: add type and create an Engine template
 * 
 */
class Engine {
public:
    // Constructor with configurations
    Engine(const TRTConfigs& configs);
    // Destructor
    ~Engine();
    // Name of the engine
    std::string m_engineName;
    // Path to the engine file
    std::string m_enginePath;
    // Store dir
    std::string m_engineStorePath;

    // device index
    int8_t m_deviceIndex;

    // Build the network
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f] (some converted models may require this).
    // If the model requires values to be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool build();
    // Load and prepare the network for inference
    bool loadNetwork();
    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    // bool runInference(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<std::vector<float>>& outputs);
    bool runInference(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs, const int32_t batchSize, cudaStream_t &inferenceStream);

    void copyToBuffer(const std::vector<cv::cuda::GpuMat>& inputs, cudaStream_t &inferenceStream);
    void copyFromBuffer(std::vector<cv::cuda::GpuMat>& outputs, const uint16_t batchSize, cudaStream_t &inferenceStream);

    // Utility method for resizing an image while maintaining the aspect ratio by adding padding to smaller dimension after scaling
    // While letterbox padding normally adds padding to top & bottom, or left & right sides, this implementation only adds padding to the right or bottom side
    // This is done so that it's easier to convert detected coordinates (ex. YOLO model) back to the original reference frame.
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor = cv::Scalar(0, 0, 0));

    [[nodiscard]] const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims ;};

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when the output batch size is 1, but there are multiple output feature vectors
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output);

    // Utility method for transforming triple nested output array into single array
    // Should be used when the output batch size is 1, and there is only a single output feature vector
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output);

    inline cv::cuda::GpuMat cvtHWCToCHW(
        const std::vector<cv::cuda::GpuMat>& batch,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
        uint8_t IMG_TYPE = CV_8UC3 
    );

    inline void normalize(
        const cv::cuda::GpuMat &transposedBatch, // NCHW
        const BatchSizeType batchSize,
        cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
        const std::array<float, 3>& subVals = {0.0f, 0.0f, 0.0f},
        const std::array<float, 3>& divVals = {1.0f, 1.0f, 1.0f},
        const float normalizedScale = 1.f
    );

    std::string getEngineName() const;
    std::vector<void *>& getInputBuffers();
    std::vector<void *>& getOutputBuffers();
private:
    // Converts the engine options into a string
    void serializeEngineOptions(const TRTConfigs& options);

    void getDeviceNames(std::vector<std::string>& deviceNames);

    // Normalization, scaling, and mean subtraction of inputs
    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    float m_normalizedScale;

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers, m_inputBuffers, m_outputBuffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
    // Dimemsions of inputs without batch size
    std::vector<nvinfer1::Dims3> m_inputDims;
    // Dimensions of outputs with batch size
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    // The Batch size with which we are going to allocate memory buffers
    int32_t m_inputBatchSize; 

    // Must keep IRuntime around for inference, see: https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const TRTConfigs m_configs;
    Logger m_logger;
    // This flag signifies whether the output goes out as cuda data or gpu data
    bool outToGPU = true;
    bool isDynamic = false;

    size_t m_maxWorkspaceSize = 1 << 30;

    MODEL_DATA_TYPE m_precision;
};

#endif