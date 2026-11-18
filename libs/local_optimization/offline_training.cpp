#include "bcedge.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, algorithm, "bcedge", "model you want to train");
ABSL_FLAG(MsvcSLOType, slo, 0, "base slo for bcedge training in ms");
ABSL_FLAG(uint16_t, epochs, 0, "number of epochs to train");
ABSL_FLAG(uint16_t, steps, 0, "number of steps per epoch");

torch::Dtype getTorchDtype(const std::string& type_str) {
    auto it = DTYPE_MAP.find(type_str);
    if (it != DTYPE_MAP.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Unknown Torch Dtype: " + type_str);
    }
}

std::vector<int> getBaseResolution(ModelType m) {
    switch (m) {
        case Yolov5n:
            return {3, 640, 640};
        case PlateDet:
            return {3, 224, 224};
        case CarBrand:
            return {3, 224, 224};
        case Retinaface:
            return {3, 288, 320};
        case RetinaMtface:
            return {3, 576, 640};
        case Movenet:
            return {3, 192, 192};
        case Age:
            return {3, 224, 224};
        case Gender:
            return {3, 224, 224};
        case Arcface:
            return {3, 112, 112};
        case Emotionnet:
            return {1, 64, 64};
        default:
            throw std::invalid_argument("Unknown Model Type: " + std::to_string(m));
    }
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    std::string algorithm = absl::GetFlag(FLAGS_algorithm);
    MsvcSLOType slo = absl::GetFlag(FLAGS_slo) * 1000;
    uint16_t epochs = absl::GetFlag(FLAGS_epochs);
    uint16_t steps = absl::GetFlag(FLAGS_steps);

    nlohmann::json metricsCfgs = nlohmann::json::parse(std::ifstream("../jsons/metricsserver.json"));
    MetricsServerConfigs metricsServerConfigs;
    metricsServerConfigs.from_json(metricsCfgs);
    metricsServerConfigs.schema = "pf15_ppp";
    metricsServerConfigs.user = "controller";
    metricsServerConfigs.password = "agent";
    std::unique_ptr<pqxx::connection> metricsServerConn = connectToMetricsServer(metricsServerConfigs, "controller");

    std::vector<spdlog::sink_ptr> loggerSinks = {};
    std::shared_ptr<spdlog::logger> logger;
    setupLogger(
            "../logs",
            "controller",
            0,
            0,
            loggerSinks,
            logger
    );

    std::map<std::string, PerDeviceModelProfileType> profiles = {};
    std::vector<std::string> tasks = {"traffic", "people"};
    std::vector<std::string> deviceTypes = {"serv", "agx", "nx", "orn", "onp"};
    std::map<std::string, double> deviceMemLimits = {{"serv", 3500}, {"agx", 3000000}, {"nx", 2000000}, {"orn", 2000000}, {"onp", 3000}};
    std::map<std::string, std::vector<std::pair<std::string, ModelType>>> modelNames;
    modelNames["serv"] = {{"yolov5n_dynamic_nms_3090_fp32_16_1.engine", Yolov5n}, {"platedet_dyn_nms_3090_fp32_64_1.engine", PlateDet}, {"carbrand_dynamic_3090_fp32_64_1.engine", CarBrand}, {"retina1face_3090_fp32_64_1.engine", Retinaface}, {"retina_multiface_640_3090_fp32_16_1.engine", RetinaMtface}, {"pose_3090_fp32_64_1.engine", Movenet}, {"age_dynamic_3090_fp32_64_1.engine", Age}, {"gender_dynamic_3090_fp32_64_1.engine", Gender}, {"arcface_dynamic_3090_fp32_64_1.engine", Arcface}, {"emotion_dyn_3090_fp32_64_1.engine", Emotionnet}};
    modelNames["agx"] = {{"yolov5n_dynamic_nms_agx_fp32_16_1.engine", Yolov5n}, {"platedet_dyn_nms_agx_fp32_16_1.engine", PlateDet}, {"carbrand_dynamic_agx_fp32_16_1.engine", CarBrand}, {"retina1face_agx_fp32_16_1.engine", Retinaface}, {"pose_agx_fp32_16_1.engine", Movenet}, {"age_dynamic_agx_fp32_16_1.engine", Age}, {"gender_dynamic_agx_fp32_16_1.engine", Gender}, {"arcface_dynamic_agx_fp32_16_1.engine", Arcface}};
    modelNames["nx"] = {{"yolov5n_dynamic_nms_nx_fp32_8_1.engine", Yolov5n}, {"platedet_dyn_nms_nx_fp32_8_1.engine", PlateDet}, {"carbrand_dynamic_nx_fp32_8_1.engine", CarBrand}, {"retina1face_nx_fp32_8_1.engine", Retinaface}, {"pose_nx_fp32_8_1.engine", Movenet}, {"age_dynamic_nx_fp32_8_1.engine", Age}, {"gender_dynamic_nx_fp32_8_1.engine", Gender}, {"arcface_dynamic_nx_fp32_8_1.engine", Arcface}};
    modelNames["orn"] = {{"yolov5n_dynamic_nms_orn_fp32_8_1.engine", Yolov5n}, {"platedet_dyn_nms_orn_fp32_8_1.engine", PlateDet}, {"carbrand_dynamic_orn_fp32_8_1.engine", CarBrand}, {"retina1face_orn_fp32_8_1.engine", Retinaface}, {"pose_orn_fp32_8_1.engine", Movenet}, {"age_dynamic_orn_fp32_8_1.engine", Age}, {"gender_dynamic_orn_fp32_8_1.engine", Gender}, {"arcface_dynamic_orn_fp32_8_1.engine", Arcface}, {"emotion_dyn_orn_fp32_8_1.engine", Emotionnet}};
    modelNames["onp"] = {{"yolov5n_dynamic_nms_1080_fp32_16_1.engine", Yolov5n}, {"retina_multiface_640_1080_fp32_16_1.engine", RetinaMtface}, {"pose_1080_fp32_64_1.engine", Movenet}};
    std::string deviceName = "";
    for (auto &task : tasks) {
        for (auto &deviceType : deviceTypes) {
            for (auto &modelName : modelNames[deviceType]) {
                ModelProfile profile = queryModelProfile(
                        *metricsServerConn,
                        "pf15",
                        "ppp",
                        task,
                        "",
                        deviceName,
                        deviceType,
                        modelName.first,
                        15
                );
                profiles[deviceType][modelName.first] = profile;
            }
        }
    }

    if (algorithm == "bcedge") {
        for (auto &deviceType : deviceTypes) {
            BCEdgeAgent *bcedge = new BCEdgeAgent(deviceType, deviceMemLimits[deviceType], torch::kF32, steps);
            for (int i = 0; i < epochs; i++) {
                for (int j = 0; j < steps; j++) {
                    MsvcSLOType modifiedSLO = slo + ((rand() % 10 - 5));
                    auto model = modelNames[deviceType][rand() % modelNames[deviceType].size()];
                    bcedge->setState(model.second, getBaseResolution(model.second), modifiedSLO);
                    auto [batch_size, scaling, memory] = bcedge->runStep();
                    batch_size = std::min((BatchSizeType) batch_size, profiles[deviceType][model.first].maxBatchSize);
                    double avg_latency = (double) (profiles[deviceType][model.first].batchInfer[batch_size].p95prepLat +
                            profiles[deviceType][model.first].batchInfer[batch_size].p95inferLat * batch_size +
                            profiles[deviceType][model.first].batchInfer[batch_size].p95postLat +
                            ((rand() % 500) - 250.0) + 0.1);
                    bcedge->rewardCallback((double) TIME_PRECISION_TO_SEC / avg_latency, avg_latency / TIME_PRECISION_TO_SEC, modifiedSLO,
                                           profiles[deviceType][model.first].batchInfer[batch_size].gpuMemUsage * scaling);
                }
            }
            delete bcedge;
        }

    } else if (algorithm == "cheis") { // only used for testing the training
        auto *cheis = new CHEISAgent(algorithm, 7, 2, 16, 2, nullptr, {}, 1, torch::kF32, steps,
                                        0, epochs + 2, .95, .99, .75, .15);
        for (int i = 0; i < epochs; i++) {
            int last_resolution = 0, last_batch_size = 0, last_threading = 0;
            for (int j = 0; j < steps; j++) {
                double arrival = 30.0;
                auto device = deviceTypes[rand() % deviceTypes.size()];
                auto model = modelNames[device][0];
                cheis->setState(last_resolution, last_batch_size, last_threading, arrival, 0.0, 0.0, 0.0, slo,
                               profiles[device][model.first].batchInfer[last_batch_size].gpuMemUsage);
                std::tie(last_resolution, last_batch_size, last_threading) = cheis->runStep();
                double avg_latency = (double) (profiles[device][model.first].batchInfer[last_batch_size].p95prepLat +
                                               profiles[device][model.first].batchInfer[last_batch_size].p95inferLat * last_batch_size +
                                               profiles[device][model.first].batchInfer[last_batch_size].p95postLat +
                                               ((rand() % 500) - 250.0) + 0.1);
                cheis->rewardCallback(std::max(arrival, (double) TIME_PRECISION_TO_SEC / avg_latency), 0.0, avg_latency / TIME_PRECISION_TO_SEC, last_batch_size / arrival);
            }
        }

    } else if (algorithm == "edgevision") {
        // TBD
        std::cerr << "EdgeVision training is not implemented yet." << std::endl;
        return 1;
    } else {
        std::cerr << "Invalid algorithm: " << algorithm << std::endl;
        return 1;
    }

    return 0;
}