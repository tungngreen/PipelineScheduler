#include "baseprocessor.h"

/**
 * @brief normalize the input data by subtracting and dividing values from the original pixesl
 * 
 * @param input the input data
 * @param subVals values to be subtracted
 * @param divVals values to be dividing by
 * @param stream an opencv stream for asynchronous operation on cuda
 */
void normalize(
    const cv::cuda::GpuMat &input,
    cv::cuda::GpuMat &normalized,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
    const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
    const std::array<float, 3>& divVals = {1.f, 1.f, 1.f}
) {
    spdlog::trace("Going into {0:s}", __func__);
    
    cv::cuda::cvtColor(input, input, cv::COLOR_BGR2RGB);
    input.convertTo(normalized, CV_32FC3, 1.f / 255.f, stream);

    cv::cuda::subtract(input, cv::Scalar(subVals[0], subVals[1], subVals[2]), input, cv::noArray(), -1);
    cv::cuda::divide(input, cv::Scalar(divVals[0], divVals[1], divVals[2]), input, 1, -1);
    spdlog::trace("Finished {0:s}", __func__);
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
cv::cuda::GpuMat resizePadRightBottom(
    const cv::cuda::GpuMat &input,
    const size_t height,
    const size_t width,
    const cv::Scalar &bgcolor,
    bool toNormalize
) {
    spdlog::trace("Going into {0:s}", __func__);
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    //Create a new GPU Mat 
    cv::cuda::GpuMat resized(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, resized, resized.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    // Creating an opencv stream for asynchronous operation on cuda
    cv::cuda::Stream stream;
    resized.copyTo(out(cv::Rect(0, 0, resized.cols, resized.rows)), stream);

    if (toNormalize) {
        cv::cuda::GpuMat normalized;
        normalize(out, normalized, stream);
        spdlog::trace("Finished {0:s}", __func__);
        return normalized;
    }
    spdlog::trace("Finished {0:s}", __func__);
    return out;
}

/**
 * @brief Construct a new Base Preprocessor that inherites the LocalGPUDataMicroservice given the `InType`
 * 
 * @param configs 
 */
BasePreprocessor::BasePreprocessor(const BaseMicroserviceConfigs &configs) : Microservice(configs){
    this->msvc_idealBatchSize = configs.msvc_idealBatchSize;
    // for (BatchSizeType i = 0; i < msvc_idealBatchSize; ++i) {
    //     msvc_batchBuffer.emplace_back(cv::cuda::GpuMat(configs.msvc_dataShape[2], configs.msvc_dataShape[3], CV_32FC3));
    // }
}

/**
 * @brief Simplest function to decide if the requests should be batched and sent to the main processor.
 * 
 * @return true True if its time to batch
 * @return false if otherwise
 */
bool BasePreprocessor::isTimeToBatch() {
    if (msvc_onBufferBatchSize == this->msvc_idealBatchSize) {
        return true;
    }
    return false;
}

/**
 * @brief Check if the request is still worth being processed.
 * For instance, if the request is already late at the moment of checking, there is no value in processing it anymore.
 * 
 * @return true 
 * @return false 
 */
bool BasePreprocessor::checkReqEligibility(ClockType currReq_gentime) {
    return true;
}