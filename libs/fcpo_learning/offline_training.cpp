#include "bcedge.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, algorithm, "bcedge", "model you want to train");
ABSL_FLAG(MsvcSLOType, slo, 0, "base slo for bcedge training in ms");
ABSL_FLAG(uint16_t, epochs, 0, "number of epochs to train");
ABSL_FLAG(uint16_t, steps, 0, "number of steps per epoch");
ABSL_FLAG(std::string, fcpo_json, "{}", "json configuration for fcpo training");

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
    nlohmann::json rl_conf = nlohmann::json::parse(absl::GetFlag(FLAGS_fcpo_json));

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
    std::map<std::string, std::vector<std::pair<std::string, ModelType>>> modelNames;
    modelNames["serv"] = {{"yolov5n_dynamic_nms_3090_fp32_16_1.engine", Yolov5n}, {"platedet_dyn_nms_3090_fp32_64_1.engine", PlateDet}, {"carbrand_dynamic_3090_fp32_64_1.engine", CarBrand}, {"retina1face_3090_fp32_64_1.engine", Retinaface}, {"retina_multiface_640_3090_fp32_16_1.engine", RetinaMtface}, {"pose_3090_fp32_64_1.engine", Movenet}, {"age_dynamic_3090_fp32_64_1.engine", Age}, {"gender_dynamic_3090_fp32_64_1.engine", Gender}, {"arcface_dynamic_3090_fp32_64_1.engine", Arcface}, {"emotion_dyn_3090_fp32_64_1.engine", Emotionnet}};
    modelNames["agx"] = {{"yolov5n_dynamic_nms_agx_fp32_16_1.engine", Yolov5n}, {"platedet_dyn_nms_agx_fp32_16_1.engine", PlateDet}, {"carbrand_dynamic_agx_fp32_16_1.engine", CarBrand}, {"retina1face_agx_fp32_16_1.engine", Retinaface}, {"pose_agx_fp32_16_1.engine", Movenet}, {"age_dynamic_agx_fp32_16_1.engine", Age}, {"gender_dynamic_agx_fp32_16_1.engine", Gender}, {"arcface_dynamic_agx_fp32_16_1.engine", Arcface}};
    modelNames["nx"] = {{"yolov5n_dynamic_nms_nx_fp32_8_1.engine", Yolov5n}, {"platedet_dyn_nms_nx_fp32_8_1.engine", PlateDet}, {"carbrand_dynamic_nx_fp32_8_1.engine", CarBrand}, {"retina1face_nx_fp32_8_1.engine", Retinaface}, {"pose_nx_fp32_8_1.engine", Movenet}, {"age_dynamic_nx_fp32_8_1.engine", Age}, {"gender_dynamic_nx_fp32_8_1.engine", Gender}, {"arcface_dynamic_nx_fp32_8_1.engine", Arcface}};
    modelNames["orn"] = {{"yolov5n_dynamic_nms_orn_fp32_8_1.engine", Yolov5n}, {"platedet_dyn_nms_orn_fp32_8_1.engine", PlateDet}, {"carbrand_dynamic_orn_fp32_8_1.engine", CarBrand}, {"retina1face_orn_fp32_8_1.engine", Retinaface}, {"pose_orn_fp32_8_1.engine", Movenet}, {"age_dynamic_orn_fp32_8_1.engine", Age}, {"gender_dynamic_orn_fp32_8_1.engine", Gender}, {"arcface_dynamic_orn_fp32_8_1.engine", Arcface}, {"emotion_dyn_orn_fp32_8_1.engine", Emotionnet}};
    modelNames["onp"] = {{"yolov5n_dynamic_nms_1080_fp32_16_1.engine", Yolov5n}, {"retina_multiface_640_1080_fp32_16_1.engine", RetinaMtface}, {"pose_1080_fp32_64_1.engine", Movenet}};
    for (auto &task : tasks) {
        for (auto &deviceType : deviceTypes) {
            for (auto &modelName : modelNames[deviceType]) {
                ModelProfile profile = queryModelProfile(
                        *metricsServerConn,
                        "pf15",
                        "ppp",
                        task,
                        "",
                        "",
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
            BCEdgeAgent *bcedge = new BCEdgeAgent(deviceType, 4000, torch::kF32, steps);
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
                    bcedge->rewardCallback((double) TIME_PRECISION_TO_SEC / avg_latency, avg_latency, modifiedSLO,
                                           profiles[deviceType][model.first].batchInfer[batch_size].gpuMemUsage * scaling);
                }
            }
            delete bcedge;
        }
    } else if (algorithm == "fcpo") {
        FCPOAgent *fcpo = new FCPOAgent(algorithm, rl_conf["state_size"], rl_conf["resolution_size"],
                                        rl_conf["batch_size"], rl_conf["threads_size"], nullptr, nullptr,
                                        getTorchDtype(rl_conf["precision"]), rl_conf["update_steps"],
                                        0, epochs + 1, rl_conf["lambda"],
                                        rl_conf["gamma"], rl_conf["clip_epsilon"], rl_conf["penalty_weight"]);
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < steps; j++) {
                //fcpo->setState();
                fcpo->runStep();
                //fcpo->rewardCallback();
            }
        }
    } else {
        std::cerr << "Invalid algorithm: " << algorithm << std::endl;
        return 1;
    }

    return 0;
}