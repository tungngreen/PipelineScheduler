#include "container_agent.h"
#include <baseprocessor.h>

using namespace spdlog;


// class YoloV5Preprocessor : public BaseReqBatcher {
// public:
//     YoloV5Preprocessor(const BaseMicroserviceConfigs &config);
//     ~YoloV5Preprocessor() = default;
// protected:
//     friend class YoloV5Agent;
//     void batchRequests();
//     // 
//     // bool isTimeToBatch() override;
//     //
//     // bool checkReqEligibility(uint64_t currReq_genTime) override;
//     //
//     // void updateReqRate(ClockTypeTemp lastInterReqDuration) override;
// };

// class YoloV5Inference : public BaseBatchInferencer {
// public:
//     YoloV5Inference(const BaseMicroserviceConfigs &config, const TRTConfigs &engineConfigs);
//     ~YoloV5Inference() = default;
// protected:
//     friend class YoloV5Agent;
//     void inference();
//     std::vector<void *> msvc_engineInputBuffers, msvc_engineOutputBuffers;
//     TRTConfigs msvc_engineConfigs;
//     Engine* msvc_inferenceEngine;
// };

// class YoloV5Postprocessor : public BaseBBoxCropper {
// public:
//     friend class YoloV5Agent;
//     YoloV5Postprocessor(const BaseMicroserviceConfigs &config);
//     ~YoloV5Postprocessor() = default;
// protected:
//     void postProcessing();
// };

class YoloV5Agent : public ContainerAgent {
public:
    YoloV5Agent(
        const json &configs
    );
};