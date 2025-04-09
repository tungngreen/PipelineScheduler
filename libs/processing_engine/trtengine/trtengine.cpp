#include <trtengine.h>

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

/**
 * @brief Construct a new Engine:: Engine object
 * 
 * @param configs 
 */
Engine::Engine(const TRTConfigs &configs) : m_configs(configs) {
    serializeEngineOptions(m_configs);
}

std::string getLastWord(const std::string& str) {
    size_t lastSpacePos = str.find_last_of(" ");

    if (lastSpacePos == std::string::npos) {
        // The string doesn't contain any spaces, so return the entire string
        return str;
    } else {
        // Return the substring from the last space to the end of the string
        return str.substr(lastSpacePos + 1);
    }
}

/**
 * @brief 
 * 
 * @param configs 
 * @param onnxModelPath 
 * @return std::string 
 */
void Engine::serializeEngineOptions(const TRTConfigs &configs) {
    const std::string& onnxModelPath = m_configs.path;
    // Generate trt model's file name from onnx's. model.onnx -> model.engine

    m_subVals = configs.subVals;
    m_divVals = configs.divVals;
    m_normalizedScale = configs.normalizeScale;
    m_precision = configs.precision;
    m_deviceIndex = configs.deviceIndex;

    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos);
    // If store path is not specified, use the path to either onnx or engine as the store path
    std::string enginePath;

    m_maxWorkspaceSize = configs.maxWorkspaceSize;

    enginePath = onnxModelPath.substr(0, onnxModelPath.find_last_of('.'));    
    if (configs.storePath.empty()) {
        m_engineStorePath = onnxModelPath.substr(0, onnxModelPath.find_last_of('/'));   
    } else {
        m_engineStorePath = configs.storePath;
    }

    // If we are converting onnx file to engine, some information about gpu should be added.
    if (m_configs.path.find(".onnx") != std::string::npos) {
    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
        std::vector<std::string> deviceNames;
        getDeviceNames(deviceNames);

        if (static_cast<size_t>(configs.deviceIndex) >= deviceNames.size()) {
            throw std::runtime_error("Error, provided device index is out of range!");
        }

        auto deviceName = deviceNames[configs.deviceIndex];
        // Remove spaces from the device name
        deviceName = getLastWord(deviceName);

        engineName+= "_" + deviceName;

        // Serialize the specified options into the filename
        if (configs.precision == MODEL_DATA_TYPE::fp16) {
            engineName += "_fp16";
        } else if (configs.precision == MODEL_DATA_TYPE::fp32){
            engineName += "_fp32";
        } else {
            engineName += "_int8";
        }
        engineName += "_" + std::to_string(configs.maxBatchSize);
        engineName += "_" + std::to_string(configs.optBatchSize);
    }



    engineName += ".engine";

    m_engineName = engineName;
    m_enginePath = m_engineStorePath + "/" + engineName;
}

/**
 * @brief Build an TRT inference engine from ONNX model file
 * 
 * @param configs configurations for the engine
 * @return true if engine is successfully generated
 * @return false if shit goes south otherwise
 */
bool Engine::build() {
    const std::string& onnxModelPath = m_configs.path;
    // Only regenerate the engine file if it has not already been generated for the specified options
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_enginePath)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating. This could take a while..." << std::endl;

    // Set the device index
    auto ret = cudaSetDevice(m_deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_configs.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch size is deprecated).
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        throw std::runtime_error("Unable to parse onnx file");
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }
    const auto onnxBatchSize = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != onnxBatchSize) {
            throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
    bool doesSupportDynamicBatch = false;
    if (onnxBatchSize == -1) {
        doesSupportDynamicBatch = true;
        std::cout << "Model supports dynamic batch size" << std::endl;
    } else {
        std::cout << "Model only supports fixed batch size of " << onnxBatchSize << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
        if (m_configs.optBatchSize != onnxBatchSize || m_configs.maxBatchSize != onnxBatchSize) {
            throw std::runtime_error("Error, model only supports a fixed batch size of " + std::to_string(onnxBatchSize) +
            ". Must set Options.optBatchSize and Options.maxBatchSize to 1");
        }
    }

    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig) {
        return false;
    }
    builderConfig->setMaxWorkspaceSize(m_maxWorkspaceSize);

    // Register a single optimization profile
    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        // [B, C, H, W]
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile`
        if (doesSupportDynamicBatch) {
            optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        } else {
            optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(m_configs.optBatchSize, inputC, inputH, inputW));
        }
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_configs.optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_configs.maxBatchSize, inputC, inputH, inputW));
    }
    builderConfig->addOptimizationProfile(optProfile);
    if (m_configs.precision == MODEL_DATA_TYPE::fp16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        builderConfig->setFlag(BuilderFlag::kFP16);
    } else if (m_configs.precision == MODEL_DATA_TYPE::int8) {
        if (numInputs > 1) {
            throw std::runtime_error("Error, this implementation currently only supports INT8 quantization for single input models");
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("Error: GPU does not support INT8 precision");
        }

        // Ensure the user has provided path to calibration data directory
        if (m_configs.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
        }

        builderConfig->setFlag((BuilderFlag::kINT8));

        const auto calibrationFileName = m_engineName + ".calibration";
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream), __func__);
    builderConfig->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *builderConfig)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineStorePath + "/" + m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineStorePath + "/" + m_engineName << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream), __func__);

    return true;
}

std::string Engine::getEngineName() const {
    return m_engineName;
}

/**
 * @brief Load the saved (or previously generated) engine file
 * 
 * @return true if the engine is successfully loaded
 * @return false if shit goes south
 */
bool Engine::loadNetwork() {
    std::ifstream file(m_configs.path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        throw std::runtime_error("File does not exist or cannot be opened: " + m_enginePath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<IRuntime> {createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_configs.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    initLibNvInferPlugins(&m_logger, "");

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbBindings());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream), __func__);

    std::int32_t batchSize = m_configs.maxBatchSize;
    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();
    uint32_t alloMemSize;
    for (uint32_t i = 0; i < m_buffers.size(); ++i) {
        const auto tensorName = m_engine->getBindingName(i);
        m_IOTensorNames.emplace_back(tensorName);
        // const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getBindingDimensions(i);
        if (m_engine->bindingIsInput(i)) {
            //
            if (tensorShape.d[0] == -1) {
                isDynamic = true;
            }
            // Allocate memory for the input
            // We want to allocate memory to fit the batch size specified in `m_configs`, but sometimes that aint really possible
            // cuz the batch size with which we generated the engine is smaller than on in `m_configs`.
            std::int32_t m_engineMaxBatchSize = m_engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
            batchSize = std::min(m_engineMaxBatchSize, batchSize);

            // Allocate enough to fit the max batch size we chose (we could end up using less later)
            alloMemSize = batchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * m_precision;
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], alloMemSize, stream), __func__);
            
            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            m_inputBatchSize = batchSize;
            m_inputBuffers.emplace_back(m_buffers[i]);
        }
    }
    for (uint32_t i = 0; i < m_buffers.size(); i++) {
        const auto tensorName = m_engine->getBindingName(i);
        m_IOTensorNames.emplace_back(tensorName);
        // const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getBindingDimensions(i);
        if (!m_engine->bindingIsInput(i)) {
            // The binding is an output
            uint32_t outputLenFloat = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * batchSize * m_precision, stream), __func__);

            m_outputBuffers.emplace_back(m_buffers[i]);
        }
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream), __func__);
    checkCudaErrorCode(cudaStreamDestroy(stream), __func__);

    return true;
}

std::vector<void *>& Engine::getInputBuffers() {
    return m_inputBuffers;
}

std::vector<void *>& Engine::getOutputBuffers() {
    return m_outputBuffers;
}

/**
 * @brief Destroy the Engine:: Engine object
 * 
 */
Engine::~Engine() {
    // Free the GPU memory
    for (auto & buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer), __func__);
    }
    m_buffers.clear();
}

/**
 * @brief Copy the preprocessed data into TRT buffer to be ready for inference. 
 * 
 * @param batch A vector of GpuMat images representing the batched data.
 * @param inferenceStream 
 */
void Engine::copyToBuffer(
    const std::vector<cv::cuda::GpuMat>& batch,
    cudaStream_t &inferenceStream
) {
    // Number of the batch predefined within the trt engine when built
    spdlog::get("container_agent")->trace("[{0:s}] going in. ", __func__);
    const auto numInputs = m_inputBuffers.size();
    // We need to copy batched data to all pre-defined batch
    for (std::size_t i = 0; i < numInputs; ++i) {
        /**
         * @brief Because the pointer to i-th input buffer is of `void *` type, which is
         * arimathically inoperable, we need a float (later change to a random type T) pointer to point
         * to the buffer.
         */
        float * inputBufferPtr;

        inputBufferPtr = (float *)(m_inputBuffers[i]);

        uint32_t singleDataSize = 1;
        // Calculating the size of each image in memory.
        for (uint8_t j = 0; j < 3; ++j) {
            singleDataSize *= m_inputDims[i].d[j];
        }
        /**
         * @brief Now, we copy all the images in the `batch` vector to the buffer
         * In the case, where the engine model has more than 1 input, the `batch` vector would look
         * like batch = {input1.1,input1.2,...,input1.M,...,inputN.1,inputN.2,...,inputN.M}, where
         * N is batch size and M is the `numInputs`.
         */
        for (std::size_t j = i; j < (batch.size() * numInputs); j += numInputs) {
            const void * dataPtr = batch.at(j).ptr<void>();
            void * bufferPtr = (void *)(inputBufferPtr);
            checkCudaErrorCode(
                cudaMemcpyAsync(
                    bufferPtr,
                    dataPtr,
                    singleDataSize * m_precision,
                    cudaMemcpyDeviceToDevice,
                    inferenceStream
                ), __func__
            );
            inputBufferPtr += singleDataSize;
        }
    }
    spdlog::get("container_agent")->trace("[{0:s}] Finished. Comming out. ", __func__);
}

/**
 * @brief After inference, we need to copy the data residing in the output buffers to 
 * 
 * @param outputs carry back the inference results in the form of GpuMat vector to the inference class
 * @param inferenceStream to ensure the operations will be done in a correct order `copyToBuffer -> inference -> copyFromBuffer` in the same stream 
 */
void Engine::copyFromBuffer(
    std::vector<cv::cuda::GpuMat>& outputs,
    const uint16_t batchSize,
    cudaStream_t &inferenceStream
) {
    spdlog::get("container_agent")->trace("[{0:s}] going in. ", __func__);
    for (std::size_t i = 0; i < m_outputBuffers.size(); ++i) {
        // After inference the 4 buffers, namely `num_detections`, `nmsed_boxes`, `nmsed_scores`, `nmsed_classes`
        // will be filled with inference results.

        // Calculating the memory for each sample in the output buffer number `i`
        uint32_t bufferMemSize = 1;
        for (int32_t j = 1; j < m_outputDims[i].nbDims; ++j) {
            bufferMemSize *= m_outputDims[i].d[j];
        }
        // Creating a GpuMat to which we would copy the memory in output buffer.
        if (i == 0) {
            cv::cuda::GpuMat batch_outputBuffer(batchSize, bufferMemSize, CV_32F);
            outputs.emplace_back(batch_outputBuffer);

            void * ptr = batch_outputBuffer.ptr<void>();
            checkCudaErrorCode(
                cudaMemcpyAsync(
                    ptr,
                    m_outputBuffers[i],
                    bufferMemSize * m_precision * batchSize,
                    cudaMemcpyDeviceToDevice,
                    inferenceStream
                ),
                __func__
            );
        } else {
            cv::cuda::GpuMat batch_outputBuffer(batchSize, bufferMemSize, CV_32F);
            outputs.emplace_back(batch_outputBuffer);

            void * ptr = batch_outputBuffer.ptr<void>();
            checkCudaErrorCode(
                cudaMemcpyAsync(
                    ptr,
                    m_outputBuffers[i],
                    bufferMemSize * m_precision * batchSize,
                    cudaMemcpyDeviceToDevice,
                    inferenceStream
                ),
                __func__
            );
        }
    }
    spdlog::get("container_agent")->trace("[{0:s}] Finished. Comming out. ", __func__);
}

inline cv::cuda::GpuMat Engine::cvtHWCToCHW(
    const std::string &callerName,
    const std::vector<cv::cuda::GpuMat>& batch,
    cv::cuda::Stream &stream,
    uint8_t IMG_TYPE
) {
    spdlog::get("container_agent")->trace("[{0:s}] going in. ", callerName + "::" + __func__);
    const BatchSizeType batchSize = batch.size();
    cv::cuda::GpuMat transposed(1, batch[0].rows * batch[0].cols * batchSize, IMG_TYPE);

    uint8_t IMG_SINGLE_CHANNEL_TYPE;
    if (batch[0].channels() == 1) {
        IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE;
        for (size_t img = 0; img < batchSize; img++) {
            std::vector<cv::cuda::GpuMat> input_channels{
                    cv::cuda::GpuMat(batch[0].rows, batch[0].cols, IMG_TYPE, &(transposed.ptr()[0 + batch[0].rows * batch[0].cols * img])),
            };
            cv::cuda::split(batch[img], input_channels);  // HWC -> CHW
        }
    } else {
        IMG_SINGLE_CHANNEL_TYPE = IMG_TYPE ^ 16;
        uint16_t height = batch[0].rows;
        uint16_t width = batch[0].cols;
        uint32_t channelMemWidth = height * width;
        for (size_t img = 0; img < batchSize; img++) {
            uint32_t offset = channelMemWidth * 3 * img;
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[0 + offset])),
                cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channelMemWidth + offset])),
                cv::cuda::GpuMat(height, width, IMG_SINGLE_CHANNEL_TYPE, &(transposed.ptr()[channelMemWidth * 2 + offset]))
            };
            cv::cuda::split(batch[img], input_channels, stream);  // HWC -> CHW
        }
    }

    stream.waitForCompletion();
    spdlog::get("container_agent")->trace("[{0:s}] Finished. Comming out. ", callerName + "::" + __func__);

    return transposed;
}

inline void Engine::normalize(
    const std::string &callerName,
    const cv::cuda::GpuMat &transposedBatch, // NCHW
    const BatchSizeType batchSize,
    cv::cuda::Stream &stream,
    const std::array<float, 3>& subVals,
    const std::array<float, 3>& divVals,
    const float normalizedScale
) {
    spdlog::get("container_agent")->trace("[{0:s}] going in. ", callerName + "::" + __func__);
    
    float * inputBufferPtr = (float *)(m_inputBuffers[0]);
    cv::cuda::GpuMat batch(1, m_inputDims.at(0).d[1] * m_inputDims.at(0).d[2] * batchSize, CV_32FC3, inputBufferPtr);
    transposedBatch.convertTo(batch, CV_32FC3, m_normalizedScale, stream);
    cv::cuda::subtract(batch, cv::Scalar(subVals[0], subVals[1], subVals[2]), batch, cv::noArray(), -1, stream);
    cv::cuda::divide(batch, cv::Scalar(divVals[0], divVals[1], divVals[2]), batch, 1, -1, stream);
    stream.waitForCompletion();

    spdlog::get("container_agent")->trace("[{0:s}] Finished. Comming out. ", callerName + "::" + __func__);
}

/**
 * @brief Inference function capable of taking varying batch size
 * 
 * @param batch 
 * @param batchSize 
 * @return true 
 * @return false 
 */
bool Engine::runInference(
    const std::vector<cv::cuda::GpuMat>& batch,
    std::vector<cv::cuda::GpuMat> &outputs,
    const int32_t batchSize,
    cudaStream_t &inferenceStream
) {
    // If we give the engine an input bigger than the previous allocated buffer, it would throw a runtime error
    if (m_inputBatchSize < batchSize) {
        std::cout << "Input's batchsize is bigger than the allocated input buffer's, which is " << m_inputBatchSize << std::endl;
        return false;
    }
    // saveGPUAsImg(batch[0], "in_reference.jpg", 255.f);

    spdlog::get("container_agent")->trace("[{0:s}] going in. ", __func__);
    // Cuda stream that will be used for inference

    // As we support dynamic batching, we need to reset the shape of the input binding everytime.
    const auto numInputs = m_inputDims.size();
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& engineInputDims = m_inputDims[i];
        nvinfer1::Dims4 inputDims = {batchSize, engineInputDims.d[0], engineInputDims.d[1], engineInputDims.d[2]};
        spdlog::get("container_agent")->trace("{0:s} has inputDims of [{1:d}, {2:d}, {3:d}, {4:d}] ", __func__, batchSize, engineInputDims.d[0], engineInputDims.d[1], engineInputDims.d[2]);
        m_context->setBindingDimensions(i, inputDims);
        // const void *dataPointer = batch.ptr<void>();
        // const int32_t inputMemSize = batchSize * engineInputDims.d[0] * engineInputDims.d[1] * engineInputDims.d[2] * sizeof(float);
        // checkCudaErrorCode(
        //     cudaMemcpyAsync(
        //         m_buffers[i],
        //         dataPointer,
        //         inputMemSize,
        //         cudaMemcpyDeviceToDevice,
        //         inferenceStream
        //     )
        // );
    }

    
    // There could be more than one inputs to the inference, and to do inference we need to make sure all the input data
    // is copied to the allocated buffers

    cv::cuda::Stream cvInferenceStream;
    cv::cuda::GpuMat transposedBatch = cvtHWCToCHW("inferencer", batch, cvInferenceStream, CV_8UC3);
    normalize("inferencer", transposedBatch, batchSize, cvInferenceStream, m_subVals, m_divVals, m_normalizedScale);

    // checkCudaErrorCode(cudaMemcpyAsync
    //     (
    //         m_inputBuffers[0],
    //         normalized.ptr<void>(),
    //         normalized.rows * normalized.cols * normalized.channels() * sizeof(float),
    //         cudaMemcpyDeviceToDevice,
    //         inferenceStream
    //     ), __func__
    // );

    // copyToBuffer(batch, inferenceStream);

    // Run Inference
    bool inferenceStatus = m_context->enqueueV2(m_buffers.data(), inferenceStream, nullptr);

    // Copy inference results from `m_outputBuffers` to `outputs`
    copyFromBuffer(outputs, batchSize, inferenceStream);
    
    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceStream), __func__);

    spdlog::get("container_agent")->trace("[{0:s}] Finished. Comming out. ", __func__);
    return inferenceStatus;


}

/**
 * @brief 
 * 
 * @param configs
 * @param onnxModelPath 
 * @return std::string 
 */

void Engine::getDeviceNames(std::vector<std::string>& deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width, const cv::Scalar &bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    //Create a new GPU Mat 
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output) {
    if (input.size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}