#include "container_agent.h"

ABSL_FLAG(std::optional<std::string>, json, std::nullopt, "configurations for microservices as json");
ABSL_FLAG(std::optional<std::string>, json_path, std::nullopt, "json for configuration inside a file");
ABSL_FLAG(std::optional<std::string>, trt_json, std::nullopt, "optional json for TRTConfiguration");
ABSL_FLAG(std::optional<std::string>, trt_json_path, std::nullopt, "json for TRTConfiguration");
ABSL_FLAG(uint16_t, port, 0, "control port for the service");
ABSL_FLAG(uint16_t, port_offset, 0, "port offset for control communication");
ABSL_FLAG(int16_t, device, 0, "Index of GPU device");
ABSL_FLAG(uint16_t, verbose, 2, "verbose level 0:trace, 1:debug, 2:info, 3:warn, 4:error, 5:critical, 6:off");
ABSL_FLAG(uint16_t, logging_mode, 0, "0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, log_dir, "../logs", "Log path for the container");
ABSL_FLAG(uint16_t, profiling_mode, 0,
          "flag to make the model running in profiling mode 0:deployment, 1:profiling, 2:empty_profiling");


torch::Dtype getTorchDtype(const std::string& type_str) {
    auto it = DTYPE_MAP.find(type_str);
    if (it != DTYPE_MAP.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Unknown Torch Dtype: " + type_str);
    }
}

void addProfileConfigs(json &msvcConfigs, const json &profileConfigs) {
    msvcConfigs["profile_inputRandomizeScheme"] = profileConfigs.at("profile_inputRandomizeScheme");
    msvcConfigs["profile_stepMode"] = profileConfigs.at("profile_stepMode");
    msvcConfigs["profile_step"] = profileConfigs.at("profile_step");
    msvcConfigs["profile_numProfileReqs"] = profileConfigs.at("profile_numProfileReqs");
    msvcConfigs["msvc_idealBatchSize"] = profileConfigs.at("profile_minBatch");
    msvcConfigs["profile_numWarmUpBatches"] = profileConfigs.at("profile_numWarmUpBatches");
    msvcConfigs["profile_maxBatch"] = profileConfigs.at("profile_maxBatch");
    msvcConfigs["profile_minBatch"] = profileConfigs.at("profile_minBatch");
}

/**
 * @brief 
 * 
 * @param containerConfigs 
 * @param profilingConfigs 
 * @return json 
 */
void manageJsonConfigs(json &configs) {
    json *containerConfigs = &configs["container"];
    json *profilingConfigs = &configs["profiling"];
    std::string name = containerConfigs->at("cont_name");

    BatchSizeType minBatch = profilingConfigs->at("profile_minBatch");
    std::string templateModelPath = profilingConfigs->at("profile_templateModelPath");

    /**
     * @brief     If this is profiling, set configurations to the first batch size that should be profiled
     * This includes
     * 1. Setting its name based on the template model path
     * 2. Setting the batch size to the smallest profile batch size
     * 
     */
    uint16_t runmode = containerConfigs->at("cont_RUNMODE");
    std::string logPath = containerConfigs->at("cont_logPath").get<std::string>();
    if (runmode == 2) {
        name = removeSubstring(templateModelPath, ".engine");
        name = replaceSubstring(name, "[batch]", std::to_string(minBatch));
        name = splitString(name, "/").back();
        logPath = "../model_profiles";
    }

    logPath += "/" + containerConfigs->at("cont_experimentName").get<std::string>();
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    logPath += "/" + containerConfigs->at("cont_systemName").get<std::string>();
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    logPath += "/" + containerConfigs->at("cont_pipeName").get<std::string>() + "_" + name;
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );
    containerConfigs->at("cont_logPath") = logPath;

    std::ifstream metricsServerCfgsFile = std::ifstream(containerConfigs->at("cont_metricServerConfigs"));
    json metricsServerConfigs = json::parse(metricsServerCfgsFile);

    (*containerConfigs)["cont_metricsServerConfigs"] = metricsServerConfigs;
    if (containerConfigs->at("cont_taskName") == "dsrc") {
        (*containerConfigs)["cont_maxBatchSize"] = 30; // we only support 30 fps for one source
    } else {
        (*containerConfigs)["cont_inferModelName"] = splitString(containerConfigs->at("cont_pipeline")[3]["path"], "/").back();
        containerConfigs->at("cont_inferModelName") = splitString(containerConfigs->at("cont_inferModelName"), ".").front();
        // The maximum batch size supported by the model (for TensorRT)
        std::vector<std::string> modelOptions = splitString(containerConfigs->at("cont_inferModelName"), "_");
        BatchSizeType maxModelBatchSize = std::stoull(modelOptions[modelOptions.size() - 2]);
        if (static_cast<RUNMODE>(runmode) == RUNMODE::PROFILING) {
            (*containerConfigs)["cont_maxBatchSize"] = std::min((BatchSizeType)profilingConfigs->at("profile_maxBatch"), maxModelBatchSize);
        } else if (static_cast<RUNMODE>(runmode) == RUNMODE::DEPLOYMENT) {
            (*containerConfigs)["cont_maxBatchSize"] = maxModelBatchSize;
        }
        containerConfigs->at("cont_pipeline")[4]["msvc_concat"] = containerConfigs->at("cont_pipeline")[1]["msvc_concat"];
    }

    for (uint16_t i = 0; i < containerConfigs->at("cont_pipeline").size(); i++) {
        containerConfigs->at("cont_pipeline")[i]["msvc_contStartTime"] = containerConfigs->at("cont_startTime");
        containerConfigs->at("cont_pipeline")[i]["msvc_contEndTime"] = containerConfigs->at("cont_endTime");
        containerConfigs->at("cont_pipeline")[i]["msvc_localDutyCycle"] = containerConfigs->at("cont_localDutyCycle");
        containerConfigs->at("cont_pipeline")[i]["msvc_cycleStartTime"] = containerConfigs->at("cont_cycleStartTime");
        containerConfigs->at("cont_pipeline")[i]["msvc_batchMode"] = containerConfigs->at("cont_batchMode");
        containerConfigs->at("cont_pipeline")[i]["msvc_batchTimeout"] = containerConfigs->at("cont_batchTimeout");
        containerConfigs->at("cont_pipeline")[i]["msvc_dropMode"] = containerConfigs->at("cont_dropMode");
        containerConfigs->at("cont_pipeline")[i]["msvc_timeBudgetLeft"] = containerConfigs->at("cont_timeBudgetLeft");
        containerConfigs->at("cont_pipeline")[i]["msvc_pipelineSLO"] = containerConfigs->at("cont_pipelineSLO");
        containerConfigs->at("cont_pipeline")[i]["msvc_experimentName"] = containerConfigs->at("cont_experimentName");
        containerConfigs->at("cont_pipeline")[i]["msvc_systemName"] = containerConfigs->at("cont_systemName");
        containerConfigs->at("cont_pipeline")[i]["msvc_contName"] = name;
        containerConfigs->at("cont_pipeline")[i]["msvc_pipelineName"] = containerConfigs->at("cont_pipeName");
        containerConfigs->at("cont_pipeline")[i]["msvc_taskName"] = containerConfigs->at("cont_taskName");
        containerConfigs->at("cont_pipeline")[i]["msvc_hostDevice"] = containerConfigs->at("cont_hostDevice");
        containerConfigs->at("cont_pipeline")[i]["msvc_deviceIndex"] = containerConfigs->at("cont_device");
        containerConfigs->at("cont_pipeline")[i]["msvc_containerLogPath"] = logPath;
        containerConfigs->at("cont_pipeline")[i]["msvc_RUNMODE"] = runmode;
        containerConfigs->at(
                "cont_pipeline")[i]["cont_metricsScrapeIntervalMillisec"] = metricsServerConfigs["metricsServer_metricsReportIntervalMillisec"];
        containerConfigs->at("cont_pipeline")[i]["msvc_numWarmUpBatches"] = containerConfigs->at("cont_numWarmUpBatches");
        containerConfigs->at("cont_pipeline")[i]["msvc_maxBatchSize"] = containerConfigs->at("cont_maxBatchSize");
        if (containerConfigs->at("cont_taskName") != "dsrc") {
            containerConfigs->at("cont_pipeline")[i]["msvc_allocationMode"] = containerConfigs->at("cont_allocationMode");
        }

        /**
         * @brief     If this is profiling, set configurations to the first batch size that should be profiled
         * This includes
         * 1. Setting its profile dir whose name is based on the template model path
         * 2. Setting the batch size to the smallest profile batch size
         * 
         */
        if (runmode == 1 && containerConfigs->at("cont_taskName") != "dsrc" && containerConfigs->at("cont_taskName") != "datasource") {
            addProfileConfigs(containerConfigs->at("cont_pipeline")[i], *profilingConfigs);
            
        } else if (runmode == 2) {
            containerConfigs->at("cont_pipeline")[i].at("msvc_idealBatchSize") = minBatch;
            if (i == 0) {
                addProfileConfigs(containerConfigs->at("cont_pipeline")[i], *profilingConfigs);
            } else if (i == 3) {
                // Set the path to the engine
                containerConfigs->at("cont_pipeline")[i].at("path") = replaceSubstring(templateModelPath, "[batch]",
                                                                                      std::to_string(minBatch));
            }
        }

        if (i == 2) {
            containerConfigs->at("cont_pipeline")[i]["msvc_modelProfile"] = containerConfigs->at("cont_modelProfile");
        }
    };

    std::cout << configs.dump(4) << std::endl;
}

json loadRunArgs(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    int8_t device = (int8_t) absl::GetFlag(FLAGS_device);
    uint16_t logLevel = absl::GetFlag(FLAGS_verbose);
    uint16_t loggingMode = absl::GetFlag(FLAGS_logging_mode);
    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    uint16_t profiling_mode = absl::GetFlag(FLAGS_profiling_mode);

    RUNMODE runmode = static_cast<RUNMODE>(profiling_mode);

    json configs = msvcconfigs::loadJson();

    // TODO: Add most of the configurations to the json file instead of the command line
    configs.at("container")["cont_device"] = device;
    configs.at("container")["cont_logLevel"] = logLevel;
    configs.at("container")["cont_logPath"] = logPath;
    configs.at("container")["cont_RUNMODE"] = runmode;
    configs.at("container")["cont_loggingMode"] = loggingMode;
    configs.at("container")["cont_port"] = absl::GetFlag(FLAGS_port);

    if (configs.at("container")["cont_taskName"] != "dsrc") {
        checkCudaErrorCode(cudaSetDevice(device), __func__);
    }

    manageJsonConfigs(configs);

    return configs;
};

std::vector<BaseMicroserviceConfigs> msvcconfigs::LoadFromJson() {
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from command line.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            spdlog::trace("{0:s} finished parsing Microservice Configs from command line.", __func__);
            return json::parse(file).get<std::vector<BaseMicroserviceConfigs>>();
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to parse Microservice Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            spdlog::trace("{0:s} finished parsing Microservice Configs from file.", __func__);
            return json::parse(absl::GetFlag(FLAGS_json).value()).get<std::vector<BaseMicroserviceConfigs>>();
        }
    }
}

json msvcconfigs::loadJson() {
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to load Json Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            auto json_file = json::parse(file);
            file = std::ifstream("../jsons/container_lib.json");
            auto containerLibs = json::parse(file);
            std::string d = json_file["container"]["cont_taskName"].get<std::string>() + "_" +
                            json_file["container"]["cont_hostDeviceType"].get<std::string>();
            // json_file["container"]["cont_pipeline"][3]["path"] = containerLibs[d]["modelPath"];
            // json_file["profiling"]["profile_templateModelPath"] = containerLibs[d]["modelPath"];
            spdlog::trace("{0:s} finished loading Json Configs from file.", __func__);
            return json_file;
        } else {
            spdlog::error("No Configurations found. Please provide configuration either as json or file.");
            exit(1);
        }
    } else {
        spdlog::trace("{0:s} attempts to load Json Configs from commandline.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            spdlog::error("Two Configurations found. Please provide configuration either as json or file.");
            exit(1);
        } else {
            auto json_file = json::parse(absl::GetFlag(FLAGS_json).value());
            spdlog::trace("{0:s} finished loading Json Configs from command line.", __func__);
            return json_file;
        }
    }
}

bool ContainerAgent::readModelProfile(const json &profile) {
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
        cont_batchInferProfileList[batch].p95prepLat = profile[i + 1].get<BatchSizeType>();
        cont_batchInferProfileList[batch].p95inferLat = profile[i + 2].get<BatchSizeType>();
        cont_batchInferProfileList[batch].p95postLat = profile[i + 3].get<BatchSizeType>();

        i += NUM_NUMBERS_PER_BATCH;
    } while (true);
    return true;
}

ContainerAgent::ContainerAgent(const json& configs) {

    json containerConfigs = configs["container"];
    //std::cout << containerConfigs.dump(4) << std::endl;

    cont_experimentName = containerConfigs["cont_experimentName"].get<std::string>();
    cont_name = containerConfigs["cont_name"].get<std::string>();
    cont_pipeName = containerConfigs["cont_pipeName"].get<std::string>();
    cont_pipeSLO = containerConfigs["cont_pipelineSLO"].get<int>();
    cont_modelSLO = cont_pipeSLO - containerConfigs["cont_timeBudgetLeft"].get<int>();
    cont_taskName = containerConfigs["cont_taskName"].get<std::string>();
    cont_hostDevice = containerConfigs["cont_hostDevice"].get<std::string>();
    cont_hostDeviceType = containerConfigs["cont_hostDeviceType"].get<std::string>();
    cont_systemName = containerConfigs["cont_systemName"].get<std::string>();

    cont_deviceIndex = containerConfigs["cont_device"];
    cont_RUNMODE = containerConfigs["cont_RUNMODE"];
    cont_logDir = containerConfigs["cont_logPath"].get<std::string>();

    std::filesystem::create_directory(
        std::filesystem::path(cont_logDir)
    );

    setupLogger(
        cont_logDir,
        cont_name,
        containerConfigs["cont_loggingMode"],
        containerConfigs["cont_logLevel"],
        cont_loggerSinks,
        cont_logger
    );

    if (cont_taskName != "dsrc" && cont_taskName != "datasource") {
        cont_inferModel = abbreviate(containerConfigs["cont_inferModelName"].get<std::string>());
        cont_metricsServerConfigs.from_json(containerConfigs["cont_metricsServerConfigs"]);
        cont_metricsServerConfigs.schema = abbreviate(cont_experimentName + "_" + cont_systemName);
        cont_metricsServerConfigs.user = "container_agent";
        cont_metricsServerConfigs.password = "agent";

        cont_metricsServerConn = connectToMetricsServer(cont_metricsServerConfigs, cont_name);

        cont_logger->info("{0:s} connected to metrics server.", cont_name);

        // Create arrival table
        std::string sql_statement;

        sql_statement = absl::StrFormat("CREATE SCHEMA IF NOT EXISTS %s;", cont_metricsServerConfigs.schema);
        pushSQL(*cont_metricsServerConn, sql_statement);

        std::string cont_experimentNameAbbr = abbreviate(cont_experimentName);
        std::string cont_pipeNameAbbr = abbreviate(cont_pipeName);
        std::string cont_taskNameAbbr = abbreviate(cont_taskName);
        std::string cont_hostDeviceAbbr = abbreviate(cont_hostDevice);
        std::string cont_hostDeviceTypeAbbr = abbreviate(cont_hostDeviceType);

        if (cont_RUNMODE == RUNMODE::DEPLOYMENT) {
            cont_arrivalTableName = cont_metricsServerConfigs.schema + "." + cont_experimentNameAbbr + "_" +  cont_pipeNameAbbr + "_" + cont_taskNameAbbr + "_arr";
            cont_processTableName = cont_metricsServerConfigs.schema + "." + cont_experimentNameAbbr + "_" +  cont_pipeNameAbbr + "__" + cont_inferModel + "__" + cont_hostDeviceTypeAbbr + "_proc";
            cont_batchInferTableName = cont_metricsServerConfigs.schema + "." + cont_experimentNameAbbr + "_" +  cont_pipeNameAbbr + "__" + cont_inferModel + "__" + cont_hostDeviceTypeAbbr + "_batch";
            cont_hwMetricsTableName = cont_metricsServerConfigs.schema + "." + cont_experimentNameAbbr + "_" +  cont_pipeNameAbbr + "__" + cont_inferModel + "__" + cont_hostDeviceTypeAbbr + "_hw";
            cont_networkTableName = cont_metricsServerConfigs.schema + "." + cont_experimentNameAbbr + "_" + cont_hostDeviceAbbr + "_netw";
        } else if (cont_RUNMODE == RUNMODE::PROFILING) {
            cont_arrivalTableName = cont_experimentNameAbbr + "_" + cont_taskNameAbbr +  "_arr";
            cont_processTableName = cont_experimentNameAbbr + "__" + cont_inferModel + "__" + cont_hostDeviceTypeAbbr + "_proc";
            cont_batchInferTableName = cont_experimentNameAbbr + "__" + cont_inferModel + "__" + cont_hostDeviceTypeAbbr + "_batch";
            cont_hwMetricsTableName =
                    cont_experimentNameAbbr + "__" + cont_inferModel + "__" + cont_hostDeviceTypeAbbr + "_hw";
            cont_networkTableName = cont_experimentNameAbbr + "_" + cont_hostDeviceTypeAbbr + "_netw";
            cont_metricsServerConfigs.schema = "public";

            std::string question = absl::StrFormat("Do you want to remove old profile entries of %s?", cont_inferModel);

            if (!confirmIntention(question, "yes")) {
                spdlog::get("container_agent")->info("Profile entries of {0:s} will NOT BE REMOVED.", cont_inferModel);
            } else {
                spdlog::get("container_agent")->info("Profile entries of {0:s} will BE REMOVED.", cont_inferModel);

                if (tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, cont_arrivalTableName)) {
                    sql_statement = "DELETE FROM " + cont_arrivalTableName + " WHERE model_name = '" + cont_inferModel + "'";
                    pushSQL(*cont_metricsServerConn, sql_statement);
                }

                sql_statement = "DROP TABLE IF EXISTS " + cont_processTableName + ";";
                pushSQL(*cont_metricsServerConn, sql_statement);

                sql_statement = "DROP TABLE IF EXISTS " + cont_batchInferTableName + ";";
                pushSQL(*cont_metricsServerConn, sql_statement);

                sql_statement = "DROP TABLE IF EXISTS " + cont_hwMetricsTableName + ";";
                pushSQL(*cont_metricsServerConn, sql_statement);

                sql_statement = "DROP TABLE IF EXISTS " + cont_batchInferTableName + ";";
                pushSQL(*cont_metricsServerConn, sql_statement);
            }
        }

        /**
         * @brief Table for network metrics, which will be used to estimate network latency
         * This will almost always be created by the device agent
         *
         */
        if (!tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, cont_networkTableName)) {
            sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_networkTableName + " ("
                                                                                    "timestamps BIGINT NOT NULL, "
                                                                                    "sender_host TEXT NOT NULL, "
                                                                                    "p95_transfer_duration_us BIGINT NOT NULL, "
                                                                                    "p95_total_package_size_b INTEGER NOT NULL)";

            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "SELECT create_hypertable('" + cont_networkTableName + "', 'timestamps', if_not_exists => TRUE);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_networkTableName + " (timestamps);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_networkTableName + " (sender_host);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "GRANT ALL PRIVILEGES ON " + cont_networkTableName + " TO " + "controller, device_agent" + ";";
            pushSQL(*cont_metricsServerConn, sql_statement);
        }

        /**
         * @brief Table for summarized arrival records
         *
         */
        if (!tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, cont_arrivalTableName)) {
            sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_arrivalTableName + " ("
                                                                                    "timestamps BIGINT NOT NULL, ";
            for (auto &period : cont_metricsServerConfigs.queryArrivalPeriodMillisec) {
                sql_statement += "arrival_rate_" + std::to_string(period/1000) + "s FLOAT, ";
                sql_statement += "coeff_var_" + std::to_string(period/1000) + "s FLOAT, ";
            }
            sql_statement += "stream TEXT NOT NULL, "
                             "model_name TEXT NOT NULL, "
                             "sender_host TEXT NOT NULL, "
                             "receiver_host TEXT NOT NULL, "
                             "p95_out_queueing_duration_us BIGINT NOT NULL, "
                             "p95_transfer_duration_us BIGINT NOT NULL, "
                             "p95_queueing_duration_us BIGINT NOT NULL, "
                             "p95_total_package_size_b INTEGER NOT NULL, "
                             "late_requests INTEGER NOT NULL, "
                             "queue_drops INTEGER NOT NULL)";

            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "SELECT create_hypertable('" + cont_arrivalTableName + "', 'timestamps', if_not_exists => TRUE);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_arrivalTableName + " (timestamps);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_arrivalTableName + " (stream);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_arrivalTableName + " (sender_host);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_arrivalTableName + " (receiver_host);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_arrivalTableName + " (model_name);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "GRANT ALL PRIVILEGES ON " + cont_arrivalTableName + " TO " + "controller, device_agent" + ";";
            pushSQL(*cont_metricsServerConn, sql_statement);
        }

        /**
         * @brief Table for summarized process records
         *
         */
        if (!tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, cont_processTableName)) {
            sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_processTableName + " ("
                                                                                    "timestamps BIGINT NOT NULL, "
                                                                                    "stream TEXT NOT NULL, "
                                                                                    "infer_batch_size INT2 NOT NULL,";
            for (auto &period : cont_metricsServerConfigs.queryArrivalPeriodMillisec) {
                sql_statement += "thrput_" + std::to_string(period/1000) + "s FLOAT, ";
            }
            sql_statement +=  "p95_prep_duration_us INTEGER NOT NULL, "
                              "p95_batch_duration_us INTEGER NOT NULL, "
                              "p95_infer_duration_us INTEGER NOT NULL, "
                              "p95_post_duration_us INTEGER NOT NULL, "
                              "p95_input_size_b INTEGER NOT NULL, "
                              "p95_output_size_b INTEGER NOT NULL, "
                              "p95_encoded_size_b INTEGER NOT NULL)";

            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "SELECT create_hypertable('" + cont_processTableName + "', 'timestamps', if_not_exists => TRUE);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_processTableName + " (timestamps);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_processTableName + " (stream);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_processTableName + " (infer_batch_size);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "GRANT ALL PRIVILEGES ON " + cont_processTableName + " TO " + "controller, device_agent" + ";";
            pushSQL(*cont_metricsServerConn, sql_statement);

        }

        /**
         * @brief Table for summarized batch infer records
         *
         */
        if (!tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, cont_batchInferTableName)) {
            sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_batchInferTableName + " ("
                                                                                    "timestamps BIGINT NOT NULL, "
                                                                                    "stream TEXT NOT NULL, ";
            sql_statement += "infer_batch_size INT2 NOT NULL, "
                             "p95_infer_duration_us INTEGER NOT NULL)";

            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "SELECT create_hypertable('" + cont_batchInferTableName + "', 'timestamps', if_not_exists => TRUE);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_batchInferTableName + " (timestamps);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_batchInferTableName + " (stream);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "CREATE INDEX ON " + cont_batchInferTableName + " (infer_batch_size);";
            pushSQL(*cont_metricsServerConn, sql_statement);

            sql_statement = "GRANT ALL PRIVILEGES ON " + cont_batchInferTableName + " TO " + "controller, device_agent" + ";";
            pushSQL(*cont_metricsServerConn, sql_statement);
        }

        if (cont_RUNMODE == RUNMODE::PROFILING) {
            if (!tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, cont_hwMetricsTableName)) {
                sql_statement = "CREATE TABLE IF NOT EXISTS " + cont_hwMetricsTableName + " ("
                                                                                          "   timestamps BIGINT NOT NULL,"
                                                                                          "   batch_size INT2 NOT NULL,"
                                                                                          "   cpu_usage INT2 NOT NULL," // percentage (1-100)
                                                                                          "   mem_usage INT NOT NULL," // Megabytes
                                                                                          "   rss_mem_usage INT NOT NULL," // Megabytes
                                                                                          "   gpu_usage INT2 NOT NULL," // percentage (1-100)
                                                                                          "   gpu_mem_usage INT NOT NULL," // Megabytes
                                                                                          "   PRIMARY KEY (timestamps)"
                                                                                          ");";
                pushSQL(*cont_metricsServerConn, sql_statement);

                sql_statement = "SELECT create_hypertable('" + cont_hwMetricsTableName +
                                "', 'timestamps', if_not_exists => TRUE);";
                pushSQL(*cont_metricsServerConn, sql_statement);

                sql_statement = "CREATE INDEX ON " + cont_hwMetricsTableName + " (timestamps);";
                pushSQL(*cont_metricsServerConn, sql_statement);

                sql_statement += "CREATE INDEX ON " + cont_hwMetricsTableName + " (batch_size);";
                pushSQL(*cont_metricsServerConn, sql_statement);

                sql_statement = "GRANT ALL PRIVILEGES ON " + cont_hwMetricsTableName + " TO " + "controller, device_agent" + ";";
                pushSQL(*cont_metricsServerConn, sql_statement);
            }
        }
        spdlog::get("container_agent")->info("{0:s} created arrival table and process table.", cont_name);
    }

    if (cont_systemName == "fcpo" || cont_systemName == "bce") {
        cont_localOptimizationIntervalMillisec = 1000;
    } else if (cont_systemName == "edvi") {
        cont_localOptimizationIntervalMillisec = 200;
    } else {
        cont_localOptimizationIntervalMillisec = cont_metricsServerConfigs.metricsReportIntervalMillisec;
    }

    handlers = {
        {MSG_TYPE[CONTAINER_STOP], std::bind(&ContainerAgent::stopExecution, this, std::placeholders::_1)},
        {MSG_TYPE[UPDATE_SENDER], std::bind(&ContainerAgent::updateSender, this, std::placeholders::_1)},
        {MSG_TYPE[BATCH_SIZE_UPDATE], std::bind(&ContainerAgent::updateBatchSize, this, std::placeholders::_1)},
        {MSG_TYPE[RESOLUTION_UPDATE], std::bind(&ContainerAgent::updateResolution, this, std::placeholders::_1)},
        {MSG_TYPE[TIME_KEEPING_UPDATE], std::bind(&ContainerAgent::updateTimeKeeping, this, std::placeholders::_1)},
        {MSG_TYPE[SYNC_DATASOURCES], std::bind(&ContainerAgent::transferFrameID, this, std::placeholders::_1)},
        {MSG_TYPE[TRANSFER_FRAME_ID], std::bind(&ContainerAgent::setFrameID, this, std::placeholders::_1)}

    };
    messaging_ctx = context_t(1);
    std::string server_address = absl::StrFormat("tcp://localhost:%d", IN_DEVICE_RECEIVE_PORT + absl::GetFlag(FLAGS_port_offset));
    sending_socket = socket_t(messaging_ctx, ZMQ_REQ);
    sending_socket.connect(server_address);
    server_address = absl::StrFormat("tcp://localhost:%d", IN_DEVICE_MESSAGE_QUEUE_PORT + absl::GetFlag(FLAGS_port_offset));
    device_message_queue = socket_t(messaging_ctx, ZMQ_SUB);
    device_message_queue.setsockopt(ZMQ_SUBSCRIBE, cont_name + "|");
    device_message_queue.connect(server_address);

    run = true;
    reportHwMetrics = false;
    profiler = nullptr;
    readModelProfile(containerConfigs["cont_modelProfile"]);
    initiateMicroservices(configs);

    hasDataReader = cont_msvcsGroups["receiver"].msvcList[0]->msvc_type == MicroserviceType::DataReader;
    isDataSource = hasDataReader && (cont_msvcsGroups["inference"].msvcList.size() == 0);
    if (hasDataReader && !isDataSource) for (auto &reader : cont_msvcsGroups["receiver"].msvcList) {
            reader->msvc_dataShape = {{-1, -1, -1}};
    }
    if (cont_systemName == "fcpo" && !isDataSource) {
        nlohmann::json rl_conf = configs["fcpo"];
        cont_fcpo_agent = new FCPOAgent(cont_name, rl_conf["state_size"], rl_conf["timeout_size"],
                                        rl_conf["batch_size"], rl_conf["threads_size"], &sending_socket,
                                        cont_batchInferProfileList,
                                        cont_msvcsGroups["batcher"].msvcList[0]->msvc_idealBatchSize,
                                        getTorchDtype(rl_conf["precision"]),
                                        rl_conf["update_steps"], rl_conf["update_step_incs"],
                                        rl_conf["federated_steps"], rl_conf["lambda"], rl_conf["gamma"],
                                        rl_conf["clip_epsilon"], rl_conf["penalty_weight"], rl_conf["theta"],
                                        rl_conf["sigma"] ,rl_conf["phi"], rl_conf["rho"] , rl_conf["seed"]);
        handlers.emplace(MSG_TYPE[RETURN_FL], std::bind(&FCPOAgent::federatedUpdateCallback, cont_fcpo_agent, std::placeholders::_1));
    }

    std::thread receiver(&ContainerAgent::HandleControlMessages, this);
    receiver.detach();
}

void ContainerAgent::initiateMicroservices(const json &configs) {
    std::vector<Microservice *> msvcsList;
    json pipeConfigs = configs["container"]["cont_pipeline"];
    uint8_t numSenders = 0;
    for (auto &pipeConfig: pipeConfigs) {
        std::string groupName = pipeConfig.at("msvc_name");
        if (groupName == "data_reader") {
            groupName = "receiver";
        }
        uint8_t numInstances = pipeConfig.at("msvc_numInstances");
        for (uint8_t i = 0; i < numInstances; i++) {
            MicroserviceType msvc_type = pipeConfig.at("msvc_type");
            std::vector<ThreadSafeFixSizedDoubleQueue *> inQueueList;
            if (msvc_type == MicroserviceType::DataReader) {
                std::vector<std::string> sources = pipeConfig["msvc_upstreamMicroservices"][0]["nb_link"];
                numInstances = sources.size();
                json runConfig = pipeConfig;
                runConfig["msvc_upstreamMicroservices"][0]["nb_link"] = {sources[i]};
                msvcsList.push_back(new DataReader(runConfig));
            } else if (msvc_type == MicroserviceType::Receiver) {
                msvcsList.push_back(new Receiver(pipeConfig));
            } else if (msvc_type >= MicroserviceType::Preprocessor &&
                       msvc_type < MicroserviceType::Batcher) {

                msvcsList.push_back(new BasePreprocessor(pipeConfig));
                msvcsList.back()->SetInQueue(cont_msvcsGroups["receiver"].outQueue);
            } else if (msvc_type >= MicroserviceType::Batcher &&
                       msvc_type < MicroserviceType::TRTInferencer) {
                msvcsList.push_back(new BaseBatcher(pipeConfig));
                msvcsList.back()->SetInQueue(cont_msvcsGroups["preprocessor"].outQueue);
            } else if (msvc_type >= MicroserviceType::TRTInferencer &&
                       msvc_type < MicroserviceType::Postprocessor) {
                msvcsList.push_back(new BaseBatchInferencer(pipeConfig));
                msvcsList.back()->SetInQueue(cont_msvcsGroups["batcher"].outQueue);
            } else if (msvc_type >= MicroserviceType::Postprocessor &&
                       msvc_type < MicroserviceType::Sender) {

                switch (msvc_type) {
                    case MicroserviceType::PostprocessorBBoxCropper:
                        msvcsList.push_back(new BaseBBoxCropper(pipeConfig));
                        break;
                    case MicroserviceType::PostProcessorClassifer:
                        msvcsList.push_back(new BaseClassifier(pipeConfig));
                        break;
                    case MicroserviceType::PostProcessorBBoxCropperVerifier:
                        msvcsList.push_back(new BaseBBoxCropperVerifier(pipeConfig));
                        break;
                    case MicroserviceType::PostProcessorKPointExtractor:
                        msvcsList.push_back(new BaseKPointExtractor(pipeConfig));
                        break;
                    case MicroserviceType::PostProcessorSMClassifier:
                        msvcsList.push_back(new BaseSoftmaxClassifier(pipeConfig));
                        break;
                    default:
                        spdlog::get("container_agent")->error("Unknown postprocessor type: {0:d}", msvc_type);
                        throw std::runtime_error("Unknown postprocessor type");
                        break;
                }
                msvcsList.back()->SetInQueue(cont_msvcsGroups["inference"].outQueue);
            } else if (msvc_type >= MicroserviceType::Sender) {
                if (pipeConfig.at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::localGPU) {
                    msvcsList.push_back(new GPUSender(pipeConfig));
                } else if (pipeConfig.at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::sharedMemory) {
                    msvcsList.push_back(new LocalCPUSender(pipeConfig));
                } else if (pipeConfig.at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::serialized ||
                           pipeConfig.at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::encodedCPU) {
                    msvcsList.push_back(new RemoteCPUSender(pipeConfig));
                } else {
                    throw std::runtime_error("Unknown communication method" + std::to_string((int)pipeConfig.at("msvc_dnstreamMicroservices")[0].at("nb_commMethod")));
                }
                if (pipeConfigs.size() == 2) { // If this is a data source container
                    msvcsList.back()->SetInQueue({cont_msvcsGroups["receiver"].outQueue[numSenders]});
                } else {
                    msvcsList.back()->SetInQueue({cont_msvcsGroups["postprocessor"].outQueue[numSenders]});
                }
                numSenders++;
            } else {
                spdlog::get("container_agent")->error("Unknown microservice type: {0:d}", msvc_type);
                throw std::runtime_error("Unknown microservice type");
            }
            msvcsList.back()->msvc_name += "_" + std::to_string(i);
            cont_msvcsGroups[groupName].msvcList.push_back(msvcsList.back());
            if (i == 0) {
                cont_msvcsGroups[groupName].outQueue = msvcsList.back()->GetOutQueue();
            } else {
                msvcsList.back()->msvc_OutQueue = cont_msvcsGroups[groupName].outQueue;
            }
        }
    }

    // this->addMicroservice(msvcsList);
}

bool ContainerAgent::addPreprocessor(uint8_t totalNumInstances) {
    std::lock_guard<std::mutex> lock(cont_pipeStructureMutex);
    uint8_t numCurrentInstances = cont_msvcsGroups["preprocessor"].msvcList.size();
    uint8_t numNewInstances = totalNumInstances - numCurrentInstances;
    if (numNewInstances < 1) {
        spdlog::get("container_agent")->info("{0:s} The current number of preprocessors ({1:d}) is equal or larger"
                                             " than the requested number ({2:d}).", __func__,
                                             cont_msvcsGroups["preprocessor"].msvcList.size(), totalNumInstances);
        return false;
    }
    Microservice *msvc;
    std::vector<Microservice *> newMsvcList;
    for (uint8_t i = 0; i < numNewInstances; i++) {
        if (cont_msvcsGroups["preprocessor"].msvcList[0]->msvc_type == MicroserviceType::Preprocessor) {
            BasePreprocessor *preprocessor = (BasePreprocessor*) cont_msvcsGroups["preprocessor"].msvcList[0];
            msvc = new BasePreprocessor(*preprocessor);
        // Add more types of preprocessors here
        } else {
            spdlog::get("container_agent")->error("{0:s} Unknown preprocessor type: {1:d}", __func__, cont_msvcsGroups["preprocessor"].msvcList[0]->msvc_type);
            throw std::runtime_error("Unknown preprocessor type");
        }
        std::string msvc_name = msvc->msvc_name;
        msvc_name = msvc_name.substr(0, msvc_name.find_last_of("_")) + "_" + std::to_string(numCurrentInstances + i);
        msvc->msvc_name = msvc_name;
        cont_msvcsGroups["preprocessor"].msvcList.push_back(msvc);
        msvc->SetInQueue(cont_msvcsGroups["receiver"].outQueue);
        for (auto &inferencer: cont_msvcsGroups["inference"].msvcList) {
            inferencer->msvc_InQueue.push_back(msvc->GetOutQueue()[0]);
        }
        newMsvcList.push_back(msvc);
        msvc->pauseThread();
        msvc->dispatchThread();
    }
    bool ready = false;
    while (!ready) {
        ready = true;
        for (auto &svc: newMsvcList) {
            if (!svc->checkReady()) {
                ready = false;
                break;
            } else {
                svc->unpauseThread();
            }
        }
    }
    spdlog::get("container_agent")->info("{0:s} Added {1:d} preprocessors.", __func__, numNewInstances);
    return true;
}

bool ContainerAgent::removePreprocessor(uint8_t numLeftInstances) {
    std::lock_guard<std::mutex> lock(cont_pipeStructureMutex);
    uint8_t numCurrentInstances = cont_msvcsGroups["preprocessor"].msvcList.size();
    uint8_t numRemoveInstances = numCurrentInstances - numLeftInstances;
    if (numRemoveInstances < 1) {
        numRemoveInstances = 1;
        spdlog::get("container_agent")->info("{0:s} The requested number of preprocessors ({1:d}) is equal or larger"
                                             " than the current ({2:d}). "
                                             "Need at least 1. ", __func__,
                                             cont_msvcsGroups["preprocessor"].msvcList.size(), numLeftInstances);
        return false;
    }
    for (uint8_t i = 0; i < numRemoveInstances; i++) {
        Microservice *msvc = cont_msvcsGroups["preprocessor"].msvcList.back();
        msvc->stopThread();
        delete msvc;
        cont_msvcsGroups["preprocessor"].msvcList.pop_back();
    }
    spdlog::get("container_agent")->info("{0:s} Removed {1:d} preprocessors.", __func__, numRemoveInstances);
    return true;
}

bool ContainerAgent::addPostprocessor(uint8_t totalNumInstances) {
    std::lock_guard<std::mutex> lock(cont_pipeStructureMutex);
    uint8_t numCurrentInstances = cont_msvcsGroups["postprocessor"].msvcList.size();
    uint8_t numNewInstances = totalNumInstances - numCurrentInstances;
    if (numNewInstances < 1) {
        spdlog::get("container_agent")->info("{0:s} The current number of postprocessors ({1:d}) is equal or larger"
                                             " than the requested number ({2:d}).", __func__,
                                             cont_msvcsGroups["postprocessor"].msvcList.size(), totalNumInstances);
        return false;
    }
    Microservice * msvc;
    std::vector<Microservice *> newMsvcList;
    for (uint8_t i = 0; i < numNewInstances; i++) {
        if (cont_msvcsGroups["postprocessor"].msvcList[0]->msvc_type == MicroserviceType::PostprocessorBBoxCropper) {
            BaseBBoxCropper *postprocessor = (BaseBBoxCropper*) cont_msvcsGroups["postprocessor"].msvcList[0];
            msvc = new BaseBBoxCropper(*postprocessor);
        } else if (cont_msvcsGroups["postprocessor"].msvcList[0]->msvc_type == MicroserviceType::PostProcessorClassifer) {
            BaseClassifier *postprocessor = (BaseClassifier*) cont_msvcsGroups["postprocessor"].msvcList[0];
            msvc = new BaseClassifier(*postprocessor);
        } else if (cont_msvcsGroups["postprocessor"].msvcList[0]->msvc_type == MicroserviceType::PostProcessorBBoxCropperVerifier) {
            BaseBBoxCropperVerifier *postprocessor = (BaseBBoxCropperVerifier*) cont_msvcsGroups["postprocessor"].msvcList[0];
            msvc = new BaseBBoxCropperVerifier(*postprocessor);
        } else if (cont_msvcsGroups["postprocessor"].msvcList[0]->msvc_type == MicroserviceType::PostProcessorKPointExtractor) {
            BaseKPointExtractor *postprocessor = (BaseKPointExtractor*) cont_msvcsGroups["postprocessor"].msvcList[0];
            msvc = new BaseKPointExtractor(*postprocessor);
        } else if (cont_msvcsGroups["postprocessor"].msvcList[0]->msvc_type == MicroserviceType::PostProcessorSMClassifier) {
            BaseSoftmaxClassifier *postprocessor = (BaseSoftmaxClassifier*) cont_msvcsGroups["postprocessor"].msvcList[0];
            msvc = new BaseSoftmaxClassifier(*postprocessor);
        // Add more types of postprocessors here
        } else {
            spdlog::get("container_agent")->error("{0:s} Unknown postprocessor type: {1:d}", __func__, cont_msvcsGroups["postprocessor"].msvcList[0]->msvc_type);
            throw std::runtime_error("Unknown postprocessor type");
        }
        std::string msvc_name = msvc->msvc_name;
        msvc_name = msvc_name.substr(0, msvc_name.find_last_of("_")) + "_" + std::to_string(numCurrentInstances + i);
        msvc->msvc_name = msvc_name;
        cont_msvcsGroups["postprocessor"].msvcList.push_back(msvc);
        msvc->SetInQueue(cont_msvcsGroups["inference"].outQueue);
        newMsvcList.push_back(msvc);
        msvc->pauseThread();
        msvc->dispatchThread();
    }
    bool ready = false;
    while (!ready) {
        ready = true;
        for (auto &svc: newMsvcList) {
            if (!svc->checkReady()) {
                ready = false;
                break;
            } else {
                svc->unpauseThread();
            }
        }
    }
    spdlog::get("container_agent")->info("{0:s} Added {1:d} postprocessors.", __func__, numNewInstances);
    return true;
}

bool ContainerAgent::removePostprocessor(uint8_t numLeftInstances) {
    std::lock_guard<std::mutex> lock(cont_pipeStructureMutex);
    uint8_t numCurrentInstances = cont_msvcsGroups["postprocessor"].msvcList.size();
    uint8_t numRemoveInstances = numCurrentInstances - numLeftInstances;
    if (numRemoveInstances < 1) {
        numRemoveInstances = 1;
        spdlog::get("container_agent")->info("{0:s} The requested number of postprocessors ({1:d}) is equal or larger"
                                             " than the current number ({2:d}). "
                                             "We need at least 1. ", __func__,
                                             cont_msvcsGroups["postprocessor"].msvcList.size(), numLeftInstances);
        return false;
    }
    for (uint8_t i = 0; i < numRemoveInstances; i++) {
        Microservice *msvc = cont_msvcsGroups["postprocessor"].msvcList.back();
        msvc->stopThread();
        delete msvc;
        cont_msvcsGroups["postprocessor"].msvcList.pop_back();
    }
    spdlog::get("container_agent")->info("{0:s} Removed {1:d} postprocessors.", __func__, numRemoveInstances);
    return true;
}

void ContainerAgent::reportStart() {
    ProcessData request;
    request.set_msvc_name(cont_name);
    std::string msg = sendMessageToDevice(MSG_TYPE[MSVC_START_REPORT], request.SerializeAsString());
    ProcessData reply;
    if (!reply.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse reply from device agent.");
        pid = 0;
    } else {
        pid = reply.pid();
    }

    spdlog::get("container_agent")->info("Container Agent started with pid: {0:d}", pid);
    if (cont_taskName != "dsrc" && cont_taskName != "sink" && cont_RUNMODE == RUNMODE::PROFILING) {
        profiler = new Profiler({pid}, "profile");
        reportHwMetrics = true;
    }
}


void ContainerAgent::runService(const json &pipeConfigs, const json &configs) {
    if (configs["container"]["cont_RUNMODE"] == RUNMODE::EMPTY_PROFILING) {
        //profiling(pipeConfigs, configs["profiling"]);
    } else {
        this->dispatchMicroservices();

        this->waitReady();
        this->START();

        // if (cont_taskName.find("dsrc") == std::string::npos) {
        //     addPreprocessor(2);
        //     addPostprocessor(2);
        // }

        collectRuntimeMetrics();
    }
    sleep(1);
    exit(0);
}

/**
 * @brief Get the Rates (request rates, throughputs) in differnent periods
 * 
 * @param timestamps a vector of timestamps, sorted in an ascending order (naturally as time goes, duh!)
 * @param periodMillisec a vector of periods in milliseconds, sorted in an ascending order
 * @return std::vector<float> 
 */
std::vector<float> getThrptsInPeriods(const std::vector<ClockType> &timestamps, const std::vector<uint64_t> &periodMillisec) {
    // Get the current time
    ClockType now = timestamps.back();

    // Vector to store the counts for each period
    std::vector<uint64_t> counts(periodMillisec.size(), 0);
    std::vector<float> rates(periodMillisec.size(), 0);

    uint8_t periodIndex = 0;
    // Iterate through each period
    for (int i = timestamps.size() - 1; i >= 0; i--) {
        if (timestamps[i] > now) { // TODO: This is a hack to avoid the case where the timestamp is in the future because of local Timing Updates of the device. This needs a better solution in the future
            continue;
        }
        // Calculate the lower bound time point for the current period
        uint64_t timeDif = std::chrono::duration_cast<TimePrecisionType>(now - timestamps[i]).count();

        while (timeDif > periodMillisec[periodIndex] * 1000) {
            periodIndex++;
            counts[periodIndex] += counts[periodIndex - 1];
        }
        counts[periodIndex]++;
    }

    while (periodIndex < periodMillisec.size() - 1) {
        periodIndex++;
        counts[periodIndex] = counts[periodIndex - 1];
    }

    for (unsigned int i = 0; i < counts.size(); i++) {
        rates[i] = counts[i] * 1000.f / periodMillisec[i] + 1;
    }

    return rates;
}


void ContainerAgent::collectRuntimeMetrics() {
    unsigned int tmp_lateCount, queueDrops = 0, miniBatchCount, aggQueueSize;
    double aggExecutedBatchSize, latencyEWMA;
    ArrivalRecordType arrivalRecords;
    ProcessRecordType processRecords;
    BatchInferRecordType batchInferRecords;
    std::string sql;

    // If we are not running in profiling mode, container_agent should not collect hardware metrics
    if (cont_RUNMODE != RUNMODE::PROFILING) {
        // Set the next hardware metrics scrape time to the life beyond
        cont_metricsServerConfigs.nextHwMetricsScrapeTime = std::chrono::high_resolution_clock::time_point::max();
    }

    auto timeNow = std::chrono::system_clock::now();
    if (timeNow > cont_metricsServerConfigs.nextMetricsReportTime) {
        cont_metricsServerConfigs.nextMetricsReportTime = timeNow + std::chrono::milliseconds(
                cont_metricsServerConfigs.metricsReportIntervalMillisec);
    }

    if (timeNow > cont_metricsServerConfigs.nextHwMetricsScrapeTime) {
        cont_metricsServerConfigs.nextHwMetricsScrapeTime = timeNow + std::chrono::milliseconds(
                cont_metricsServerConfigs.hwMetricsScrapeIntervalMillisec);
    }

    if (timeNow > cont_nextOptimizationMetricsTime) {
        cont_nextOptimizationMetricsTime = timeNow + std::chrono::milliseconds(cont_localOptimizationIntervalMillisec);
    }

    /**
     * @brief If the container is a data source container, it will wait for the data receiver to stop before exiting
     *
     */
    if (isDataSource) {
        while (run) {
            if (cont_msvcsGroups["receiver"].msvcList[0]->STOP_THREADS) {
                run = false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // Maximum number of seconds to keep the arrival records, usually 60
    uint16_t maxNumSeconds = cont_metricsServerConfigs.queryArrivalPeriodMillisec.back() / 1000;
    // Initiate a fixed-size vector to store the arrival records for each second
    RunningArrivalRecord perSecondArrivalRecords(maxNumSeconds);
    while (run) {
        bool hwMetricsScraped = false;
        auto metricsStopwatch = Stopwatch();
        metricsStopwatch.start();
        auto startTime = metricsStopwatch.getStartTime();
        uint64_t scrapeLatencyMillisec = 0;
        uint64_t timeDiff;

        if (reportHwMetrics) {
            if (timePointCastMillisecond(startTime) >= timePointCastMillisecond(cont_metricsServerConfigs.nextHwMetricsScrapeTime) && pid > 0) {
                Profiler::sysStats stats = profiler->reportAtRuntime(getpid(), pid);
                cont_hwMetrics = {stats.cpuUsage, stats.processMemoryUsage, stats.rssMemory, stats.gpuUtilization,
                                             stats.gpuMemoryUsage};
                metricsStopwatch.stop();
                scrapeLatencyMillisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
                hwMetricsScraped = true;
                cont_metricsServerConfigs.nextHwMetricsScrapeTime = std::chrono::high_resolution_clock::now() +
                    std::chrono::milliseconds(cont_metricsServerConfigs.hwMetricsScrapeIntervalMillisec - scrapeLatencyMillisec);
                spdlog::get("container_agent")->trace("{0:s} SCRAPE hardware metrics. Latency {1:d}ms.",
                                                     cont_name, scrapeLatencyMillisec);
                metricsStopwatch.start();
            }
        }

        if (timePointCastMillisecond(startTime) >= timePointCastMillisecond(cont_metricsServerConfigs.nextArrivalRateScrapeTime)) {
            spdlog::get("container_agent")->trace("{0:s} SCRAPE per second arrival rate.", cont_name);
            PerSecondArrivalRecord perSecondArrivalRecord;
            for (auto &receiver: cont_msvcsGroups["receiver"].msvcList) {
                perSecondArrivalRecord = perSecondArrivalRecord + receiver->getPerSecondArrivalRecord();
            }
            perSecondArrivalRecords.addRecord(perSecondArrivalRecord);
            // secondIndex = (secondIndex + 1) % maxNumSeconds;
            metricsStopwatch.stop();
            auto localScrapeLatencyMilisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
            scrapeLatencyMillisec += localScrapeLatencyMilisec;

            cont_metricsServerConfigs.nextArrivalRateScrapeTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(1000 - localScrapeLatencyMilisec);
            metricsStopwatch.start();
        }

        if (timePointCastMillisecond(startTime) >= timePointCastMillisecond(cont_nextOptimizationMetricsTime)) {
            if (cont_systemName == "fcpo") {
                tmp_lateCount = 0;
                for (auto &recv: cont_msvcsGroups["receiver"].msvcList) tmp_lateCount += recv->GetDroppedReqCount();
                cont_late_drops += tmp_lateCount;
                cont_request_arrival_rate = perSecondArrivalRecords.getAvgArrivalRate() - tmp_lateCount;

                aggExecutedBatchSize = 0.1;
                for (auto &bat: cont_msvcsGroups["batcher"].msvcList) aggExecutedBatchSize += bat->GetAggExecutedBatchSize();
                miniBatchCount = 0;
                latencyEWMA = 0.0;
                for (auto &post: cont_msvcsGroups["postprocessor"].msvcList) {
                    miniBatchCount += post->GetMiniBatchCount();
                    latencyEWMA += post->getLatencyEWMA();
                }
                cont_ewma_latency = latencyEWMA / cont_msvcsGroups["postprocessor"].msvcList.size();
                double batch_size = cont_msvcsGroups["batcher"].msvcList[0]->msvc_idealBatchSize;

                if (cont_request_arrival_rate == 0 || std::isnan(cont_request_arrival_rate)) {
                    cont_fcpo_agent->rewardCallback(0.0, 0.0,
                                                    (double) batch_size / 10.0,
                                                    (double) cont_batchInferProfileList[batch_size].gpuMemUsage / 1000.0);
                    cont_request_arrival_rate = 0;
                } else {
                    cont_request_arrival_rate = std::max(0.1, cont_request_arrival_rate); // prevent negative values in the case of many drops or no requests
                    cont_fcpo_agent->rewardCallback((double) aggExecutedBatchSize / cont_request_arrival_rate,
                                                    (double) cont_ewma_latency / TIME_PRECISION_TO_SEC,
                                                    batch_size / cont_request_arrival_rate,
                                                    (double) cont_batchInferProfileList[batch_size].gpuMemUsage / 1000.0);
                }
                spdlog::get("container_agent")->info(
                        "RL Decision Input: {0:d} miniBatches, {1:f} request rate, {2:f} latency, {3:f} aggExecutedBatchSize",
                        miniBatchCount, cont_request_arrival_rate, (double) cont_ewma_latency / TIME_PRECISION_TO_SEC, aggExecutedBatchSize);
                cont_fcpo_agent->setState(cont_msvcsGroups["preprocessor"].msvcList[0]->msvc_concat.numImgs,
                                          batch_size,
                                          cont_threadingAction,
                                          cont_request_arrival_rate / 250.0,
                                          cont_msvcsGroups["receiver"].msvcList[0]->msvc_OutQueue[0]->size(),
                                          cont_msvcsGroups["preprocessor"].msvcList[0]->msvc_OutQueue[0]->size(),
                                          cont_msvcsGroups["inference"].msvcList[0]->msvc_OutQueue[0]->size(),
                                          cont_modelSLO,
                                          (double) cont_batchInferProfileList[batch_size].gpuMemUsage);
                auto [targetTO, newBS, scaling] = cont_fcpo_agent->runStep();
                spdlog::get("container_agent")->info(
                        "RL Decision Output: Timout: {0:d}, Batch Size: {1:d}, Scaling: {2:d}", targetTO, newBS, scaling);
                applyBatchingTimeout(targetTO);
                applyBatchSize(newBS);
                applyMultiThreading(scaling);

                cont_nextOptimizationMetricsTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(cont_localOptimizationIntervalMillisec);
            } else if (cont_systemName == "bce") {
                ClientContext context;
                indevicemessages::BCEdgeData request;
                indevicemessages::BCEdgeConfig reply;
                request.set_msvc_name(cont_name);
                request.set_slo(cont_msvcsGroups["inference"].msvcList[0]->msvc_contSLO);
                aggExecutedBatchSize = 0.1;
                for (auto &bat: cont_msvcsGroups["batcher"].msvcList) aggExecutedBatchSize += bat->GetAggExecutedBatchSize();
                miniBatchCount = 0;
                latencyEWMA = 0.0;
                for (auto &post: cont_msvcsGroups["postprocessor"].msvcList) {
                    miniBatchCount += post->GetMiniBatchCount();
                    latencyEWMA += post->getLatencyEWMA();
                }
                request.set_throughput((double) aggExecutedBatchSize / perSecondArrivalRecords.getAvgArrivalRate());
                cont_ewma_latency = latencyEWMA / cont_msvcsGroups["postprocessor"].msvcList.size();
                request.set_latency(cont_ewma_latency / TIME_PRECISION_TO_SEC);
                std::string msg = sendMessageToDevice(MSG_TYPE[BCEDGE_UPDATE], request.SerializeAsString());
                if (!reply.ParseFromString(msg)) {
                    spdlog::get("container_agent")->error("Failed to parse BCEdge reply: {0:s}", msg);
                } else {
                    applyBatchSize(std::min((BatchSizeType) reply.batch_size(), cont_msvcsGroups["batcher"].msvcList[0]->msvc_maxBatchSize));
                }
                cont_nextOptimizationMetricsTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(cont_localOptimizationIntervalMillisec);
            } else {
                cont_request_arrival_rate = perSecondArrivalRecords.getAvgArrivalRate() - tmp_lateCount;
                if (std::isnan(cont_request_arrival_rate)) {
                    cont_request_arrival_rate = 0;
                }

                aggQueueSize = 0;
                for (auto group: cont_msvcsGroups) {
                    for (auto &msvc: group.second.msvcList) {
                        aggQueueSize += msvc->GetOutQueueSize();
                    }
                }
                cont_queue_size = aggQueueSize;

                latencyEWMA = 0;
                for (auto &post: cont_msvcsGroups["postprocessor"].msvcList) {
                    miniBatchCount += post->GetMiniBatchCount();
                    latencyEWMA += post->getLatencyEWMA();
                }
                cont_ewma_latency = latencyEWMA / cont_msvcsGroups["postprocessor"].msvcList.size();


                tmp_lateCount = 0;
                for (auto &recv: cont_msvcsGroups["receiver"].msvcList) tmp_lateCount += recv->GetDroppedReqCount();
                cont_late_drops += tmp_lateCount;

                aggExecutedBatchSize = 0.1;
                for (auto &bat: cont_msvcsGroups["batcher"].msvcList) aggExecutedBatchSize += bat->GetAggExecutedBatchSize();
                cont_throughput = aggExecutedBatchSize;

                ContainerMetrics request;
                request.set_name(cont_name);
                request.set_arrival_rate(cont_request_arrival_rate);
                request.set_queue_size(cont_queue_size);
                request.set_avg_latency(cont_ewma_latency);
                request.set_drops(cont_late_drops);
                request.set_throughput(cont_throughput);
                sendMessageToDevice(MSG_TYPE[CONTEXT_METRICS], request.SerializeAsString());
                cont_nextOptimizationMetricsTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(cont_localOptimizationIntervalMillisec);
            }
        }

        startTime = std::chrono::high_resolution_clock::now();
        if (timePointCastMillisecond(startTime) >=
                timePointCastMillisecond(cont_metricsServerConfigs.nextMetricsReportTime)) {
            Stopwatch pushMetricsStopWatch;
            pushMetricsStopWatch.start();
            queueDrops += cont_msvcsGroups["receiver"].msvcList[0]->GetQueueDrops();
            queueDrops += cont_msvcsGroups["preprocessor"].msvcList[0]->GetQueueDrops();
            queueDrops += cont_msvcsGroups["inference"].msvcList[0]->GetQueueDrops();
            queueDrops += cont_msvcsGroups["postprocessor"].msvcList[0]->GetQueueDrops();

            spdlog::get("container_agent")->info("{0:s} had {1:d} late requests of {2:f} total requests. ({3:d} queue drops)", cont_name, cont_late_drops, perSecondArrivalRecords.getAvgArrivalRate(), queueDrops);

            std::string modelName = cont_msvcsGroups["inference"].msvcList[0]->getModelName();
            if (cont_RUNMODE == RUNMODE::PROFILING) {
                if (reportHwMetrics && cont_hwMetrics.metricsAvailable) {
                    sql = "INSERT INTO " + cont_hwMetricsTableName +
                        " (timestamps, batch_size, cpu_usage, mem_usage, rss_mem_usage, gpu_usage, gpu_mem_usage) VALUES ";
                    sql += "(" + timePointToEpochString(std::chrono::high_resolution_clock::now()) + ", ";
                    sql += std::to_string(cont_msvcsGroups["preprocessor"].msvcList[0]->msvc_idealBatchSize) + ", ";
                    sql += std::to_string(cont_hwMetrics.cpuUsage) + ", ";
                    sql += std::to_string(cont_hwMetrics.memUsage) + ", ";
                    sql += std::to_string(cont_hwMetrics.rssMemUsage) + ", ";
                    sql += std::to_string(cont_hwMetrics.gpuUsage) + ", ";
                    sql += std::to_string(cont_hwMetrics.gpuMemUsage) + ")";
                    sql += ";";
                    pushSQL(*cont_metricsServerConn, sql.c_str());
                    cont_hwMetrics.clear();
                    spdlog::get("container_agent")->trace("{0:s} pushed hardware metrics to the database.", cont_name);
                }
                bool allStopped = true;
                for (auto &receiver: cont_msvcsGroups["receiver"].msvcList) {
                    if (!receiver->STOP_THREADS) {
                        allStopped = false;
                        break;
                    }
                }
                if (allStopped) {
                    run = false;
                    continue;
                }
            }

            arrivalRecords.clear();
            processRecords.clear();
            batchInferRecords.clear();
            for (auto &postproc: cont_msvcsGroups["postprocessor"].msvcList) {
                BasePostprocessor* postprocPointer = (BasePostprocessor*) postproc;
                postprocPointer->getArrivalRecords(arrivalRecords);
                postprocPointer->getProcessRecords(processRecords);
                postprocPointer->getBatchInferRecords(batchInferRecords);
            }

            updateArrivalRecords(arrivalRecords, perSecondArrivalRecords, cont_late_drops, queueDrops);
            updateProcessRecords(processRecords, batchInferRecords);
            pushMetricsStopWatch.stop();
            auto pushMetricsLatencyMillisec = (uint64_t) std::ceil(pushMetricsStopWatch.elapsed_microseconds() / 1000.f);
            spdlog::get("container_agent")->trace("{0:s} pushed ALL METRICS to the database. Latency {1:d}ms. Next push in {2:d}ms",
                                                 cont_name,
                                                 pushMetricsLatencyMillisec,
                                                 cont_metricsServerConfigs.metricsReportIntervalMillisec - pushMetricsLatencyMillisec);

            queueDrops = 0;
            cont_late_drops = 0;
            cont_metricsServerConfigs.nextMetricsReportTime += std::chrono::milliseconds(
                    cont_metricsServerConfigs.metricsReportIntervalMillisec - pushMetricsLatencyMillisec);
        }
        metricsStopwatch.stop();
        auto reportLatencyMillisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
        ClockType nextTime;
        nextTime = std::min(cont_metricsServerConfigs.nextMetricsReportTime,
                            cont_metricsServerConfigs.nextArrivalRateScrapeTime);
        if (reportHwMetrics && hwMetricsScraped) {
            nextTime = std::min(nextTime, cont_metricsServerConfigs.nextHwMetricsScrapeTime);
        }
        nextTime = std::min(nextTime, cont_nextOptimizationMetricsTime);
        if (hasDataReader && cont_msvcsGroups["receiver"].msvcList[0]->STOP_THREADS) {
            run = false;
        }
        timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextTime - std::chrono::high_resolution_clock::now()).count();
        std::chrono::milliseconds sleepPeriod(timeDiff - (reportLatencyMillisec) + 2);
        spdlog::get("container_agent")->trace("{0:s} Container Agent's Metric Reporter sleeps for {1:d} milliseconds.", cont_name, sleepPeriod.count());
        std::this_thread::sleep_for(sleepPeriod);
    }

    stopAllMicroservices();
}

void ContainerAgent::applyFramePacking(int resolutionConfig) {
    for (auto preproc : cont_msvcsGroups["preprocessor"].msvcList) {
        if (preproc->msvc_concat.numImgs != resolutionConfig){
            preproc->flushBuffers();
            preproc->msvc_concat.numImgs = resolutionConfig;
        }
    }
}

void ContainerAgent::applyBatchSize(int batchSize) {
    for (auto batcher : cont_msvcsGroups["batcher"].msvcList) {
        batcher->msvc_idealBatchSize = batchSize;
    }
    for (auto infer : cont_msvcsGroups["inference"].msvcList) {
        infer->msvc_idealBatchSize = batchSize;
    }
};

void ContainerAgent::applyBatchingTimeout(int timeoutChoice) {
    for (auto batcher : cont_msvcsGroups["batcher"].msvcList) {
        batcher->msvc_batchWaitLimit = std::pow(BATCH_WAIT_BASE_MICROSEC, timeoutChoice); // in microseconds
    }
};

void ContainerAgent::applyMultiThreading(int multiThreadingConfig) {
    if (cont_threadingAction != static_cast<threadingAction>(multiThreadingConfig)) {
        cont_threadingAction = static_cast<threadingAction>(multiThreadingConfig);
        return; // ensure that we always have two same decisions in a row before applying the change
    }
    int current_pre_count = cont_msvcsGroups["preprocessor"].msvcList.size();
    int current_post_count = cont_msvcsGroups["postprocessor"].msvcList.size();
    switch (multiThreadingConfig) {
        case NoMultiThreads:
            if (current_pre_count > 1) { removePreprocessor(1); }
            if (current_post_count > 1) { removePostprocessor(1); }
            break;
        case MultiPreprocess:
            if (current_pre_count == 1) { addPreprocessor(2); }
            if (current_post_count > 1) { removePostprocessor(1); }
            break;
        case MultiPostprocess:
            if (current_pre_count > 1) { removePreprocessor(1); }
            if (current_post_count == 1) { addPostprocessor(2); }
            break;
        case BothMultiThreads:
            if (current_pre_count == 1) { addPreprocessor(2); }
            if (current_post_count == 1) { addPostprocessor(2); }
            break;
        default:
            spdlog::get("container_agent")->error("{0:s} Unknown multi-threading configuration: {1:d}", __func__, multiThreadingConfig);
            break;
    }
};

void ContainerAgent::updateArrivalRecords(ArrivalRecordType arrivalRecords, RunningArrivalRecord &perSecondArrivalRecords, unsigned int lateCount, unsigned int queueDrops) {
    std::string sql;
    // Keys value here is std::pair<std::string, std::string> for stream and sender_host
    NetworkRecordType networkRecords;
    perSecondArrivalRecords.aggregateArrivalRecord(cont_metricsServerConfigs.queryArrivalPeriodMillisec);
    std::vector<float> requestRates = perSecondArrivalRecords.getArrivalRatesInPeriods();
    std::vector<float> coeffVars = perSecondArrivalRecords.getCoeffVarsInPeriods();
    for (auto &[keys, records]: arrivalRecords) {
        uint32_t numEntries = records.arrivalTime.size();
        if (numEntries == 0) continue;
        std::string stream = keys.first;
        std::string senderHostAbbr = abbreviate(keys.second);
        std::vector<uint8_t> percentiles = {95};
        std::map<uint8_t, PercentilesArrivalRecord> percentilesRecord = records.findPercentileAll(percentiles);

        if (percentilesRecord[95].transferDuration > LLONG_MAX) { // 2^63-1 is the maximum value for BIGINT
            spdlog::get("container_agent")->warn("{0:s} Transfer duration is too high: {1:d}us", cont_name, percentilesRecord[95].transferDuration);
            continue;
        }

        sql = absl::StrFormat("INSERT INTO %s (timestamps, stream, model_name, sender_host, receiver_host, ", cont_arrivalTableName);
        for (auto &period : cont_metricsServerConfigs.queryArrivalPeriodMillisec) {
            sql += "arrival_rate_" + std::to_string(period/1000) + "s, ";
            sql += "coeff_var_" + std::to_string(period/1000) + "s, ";
        }
        sql += absl::StrFormat("p95_out_queueing_duration_us, p95_transfer_duration_us, p95_queueing_duration_us, p95_total_package_size_b, late_requests, queue_drops) "
                               "VALUES ('%s', '%s', '%s', '%s', '%s'",
                               timePointToEpochString(std::chrono::system_clock::now()),
                               stream,
                               cont_inferModel,
                               senderHostAbbr,
                               abbreviate(cont_hostDevice));
        for (unsigned int i = 0; i < requestRates.size(); i++) {
            sql += ", " + std::to_string(std::isnan(requestRates[i]) ? 0 : requestRates[i]);
            sql += ", " + std::to_string(std::isnan(coeffVars[i]) ? 0 : coeffVars[i]);
        }
        sql += absl::StrFormat(", %ld, %ld, %ld, %d, %d, %d);",
                               percentilesRecord[95].outQueueingDuration,
                               percentilesRecord[95].transferDuration,
                               percentilesRecord[95].queueingDuration,
                               percentilesRecord[95].totalPkgSize,
                               lateCount,queueDrops);

        pushSQL(*cont_metricsServerConn, sql.c_str());

        if (networkRecords.find(senderHostAbbr) == networkRecords.end()) {
            networkRecords[senderHostAbbr] = {
                    percentilesRecord[95].totalPkgSize,
                    percentilesRecord[95].transferDuration
            };
        } else {
            networkRecords[senderHostAbbr] = {
                    std::max(percentilesRecord[95].totalPkgSize, networkRecords[senderHostAbbr].totalPkgSize),
                    std::max(percentilesRecord[95].transferDuration, networkRecords[senderHostAbbr].transferDuration)
            };
        }
    }
    for (auto &[senderHost, record]: networkRecords) {
        std::string senderHostAbbr = abbreviate(senderHost);
        sql = absl::StrFormat("INSERT INTO %s (timestamps, sender_host, p95_transfer_duration_us, p95_total_package_size_b) "
                              "VALUES ('%s', '%s', %ld, %d);",
                              cont_networkTableName,
                              timePointToEpochString(std::chrono::system_clock::now()),
                              senderHostAbbr,
                              record.transferDuration,
                              record.totalPkgSize);
        pushSQL(*cont_metricsServerConn, sql.c_str());
        spdlog::get("container_agent")->trace("{0:s} pushed NETWORK METRICS to the database.", cont_name);
    }

    spdlog::get("container_agent")->trace("{0:s} pushed arrival metrics to the database.", cont_name);
}

void ContainerAgent::updateProcessRecords(ProcessRecordType processRecords, BatchInferRecordType batchInferRecords) {
    for (auto& [key, records] : processRecords) {
        std::string reqOriginStream = key.first;
        BatchSizeType inferBatchSize = key.second;
        uint32_t numEntries = records.postEndTime.size();
        // Check if there are any records
        if (numEntries < 20) {
            continue;
        }

        // Construct the SQL statement
        std::string sql = absl::StrFormat("INSERT INTO %s (timestamps, stream, infer_batch_size", cont_processTableName);

        for (auto& period : cont_metricsServerConfigs.queryArrivalPeriodMillisec) {
            sql += ", thrput_" + std::to_string(period / 1000) + "s";
        }

        sql += ", p95_prep_duration_us, p95_batch_duration_us, p95_infer_duration_us, p95_post_duration_us, p95_input_size_b, p95_output_size_b, p95_encoded_size_b) VALUES (";
        sql += timePointToEpochString(std::chrono::high_resolution_clock::now()) + ", '" + reqOriginStream + "'," + std::to_string(inferBatchSize);

        // Calculate the throughput rates for the configured periods
        std::vector<float> throughputRates = getThrptsInPeriods(records.postEndTime, cont_metricsServerConfigs.queryArrivalPeriodMillisec);
        for (const auto& rate : throughputRates) {
            sql += ", " + std::to_string(rate);
        }

        std::map<uint8_t, PercentilesProcessRecord> percentilesRecord = records.findPercentileAll({95});

        // Add the 95th percentile values from the summarized records
        sql += ", " + std::to_string(percentilesRecord[95].prepDuration);
        sql += ", " + std::to_string(percentilesRecord[95].batchDuration);
        sql += ", " + std::to_string(percentilesRecord[95].inferDuration);
        sql += ", " + std::to_string(percentilesRecord[95].postDuration);
        sql += ", " + std::to_string(percentilesRecord[95].inputSize);
        sql += ", " + std::to_string(percentilesRecord[95].outputSize);
        sql += ", " + std::to_string(percentilesRecord[95].encodedOutputSize);
        sql += ")";

        // Push the SQL statement
        pushSQL(*cont_metricsServerConn, sql.c_str());
    }
    processRecords.clear();
    spdlog::get("container_agent")->trace("{0:s} pushed PROCESS METRICS to the database.", cont_name);

    for (auto& [keys, records] : batchInferRecords) {
        uint32_t numEntries = records.inferDuration.size();
        // Check if there are any records
        if (numEntries == 0) { continue; }

        std::string reqOriginStream = keys.first;
        BatchSizeType inferBatchSize = keys.second;

        std::map<uint8_t, PercentilesBatchInferRecord> percentilesRecord = records.findPercentileAll({95});

        // Construct the SQL statement
        std::string sql = absl::StrFormat("INSERT INTO %s (timestamps, stream, infer_batch_size, p95_infer_duration_us) "
                              "VALUES (%s, '%s', %d, %ld)",
                              cont_batchInferTableName,
                              timePointToEpochString(std::chrono::high_resolution_clock::now()),
                              reqOriginStream,
                              inferBatchSize,
                              percentilesRecord[95].inferDuration);

        // Push the SQL statement
        pushSQL(*cont_metricsServerConn, sql.c_str());
    }
    batchInferRecords.clear();
    spdlog::get("container_agent")->trace("{0:s} pushed BATCH INFER METRICS to the database", cont_name);
}

void ContainerAgent::HandleControlMessages() {
    while (run) {
        message_t message;
        if (device_message_queue.recv(message, recv_flags::none)) {
            std::string raw = message.to_string();
            std::istringstream iss(raw);
            std::string topic, type;
            iss >> topic;
            iss >> type;
            iss.get(); // skip the space after the topic
            std::string payload((std::istreambuf_iterator<char>(iss)),
                                std::istreambuf_iterator<char>());
            if (handlers.count(topic)) {
                handlers[topic](payload);
            } else {
                spdlog::get("container_agent")->error("Received unknown device topic: {}", topic);
            }
        } else {
            spdlog::get("container_agent")->error("Received unsupported message in device communication!");
        }
    }
}

std::string ContainerAgent::sendMessageToDevice(const std::string &type, const std::string &content) {
    std::string msg = absl::StrFormat("%s %s", type, content);
    message_t zmq_msg(msg.size()), response;
    memcpy(zmq_msg.data(), msg.data(), msg.size());
    if (!sending_socket.send(zmq_msg, send_flags::dontwait)) {
        spdlog::get("container_agent")->error("Failed to send message to device: {0:s}", msg);
        return "";
    }
    if (!sending_socket.recv(response, recv_flags::none)) {
        spdlog::get("container_agent")->error("Failed to receive response from device for message: {0:s}", msg);
        return "";
    }
    return response.to_string();
}

void ContainerAgent::stopExecution(const std::string &msg) {
    run = false;
}

void ContainerAgent::updateSender(const std::string &msg) {
    Connection request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse updateSender request: {0:s}", msg);
        return;
    }
    if (request.offloading_duration() != 0) {
        //get current time
        auto now = std::chrono::high_resolution_clock::now();
        auto start = ClockType(TimePrecisionType(request.timestamp()));
        if (now < start) {
            std::this_thread::sleep_for(start - now);
        } else if (now > start + std::chrono::seconds(request.offloading_duration())) {
            spdlog::get("container_agent")->error("Received Offloading Request for {0:s} too late", request.name());
            for (auto &group : cont_msvcsGroups) {
                for (auto msvc : group.second.msvcList) {
                    msvc->unpauseThread();
                }
            }
            return;
        }
    }
    // pause processing except senders to clear out the queues
    for (auto group : cont_msvcsGroups) {
        for (auto msvc : group.second.msvcList) {
            if (msvc->dnstreamMicroserviceList[0].name == request.name()) {
                continue;
            }
            msvc->pauseThread();
        }
    }
    json *config;
    std::string link = absl::StrFormat("%s:%d", request.ip(), request.port());
    auto senders = &cont_msvcsGroups["sender"].msvcList;
    for (auto sender: *senders) {
        if (sender->dnstreamMicroserviceList[0].name == request.name()) {
            config = &sender->msvc_configs;
            std::vector<msvcconfigs::NeighborMicroservice *> postprocessor_dnstreams = {};
            for (auto *postprocessor : cont_msvcsGroups["postprocessor"].msvcList) {
                for (auto &dnstream : postprocessor->dnstreamMicroserviceList) {
                    if (sender->msvc_name.find(dnstream.name) != std::string::npos) {
                        postprocessor_dnstreams.push_back(&dnstream);
                    }
                }
            }
            auto nb_links = config->at("msvc_dnstreamMicroservices")[0]["nb_link"];
            if (request.mode() == AdjustUpstreamMode::Overwrite) {
                if (request.old_link() == "") {
                    config->at("msvc_dnstreamMicroservices")[0]["nb_link"] = {link};
                    config->at("msvc_dnstreamMicroservices")[0]["nb_portions"] = {};
                    for (auto postprocessor : postprocessor_dnstreams) {
                        postprocessor->portions.clear();
                    }
                    spdlog::get("container_agent")->trace("Overwrote all links in {0:s} to {1:s}", sender->msvc_name, link);
                }
                auto it = std::find(nb_links.begin(), nb_links.end(), request.old_link());
                if (it == nb_links.end()) {
                    spdlog::get("container_agent")->error("Link {0:s} not found in {1:s}", request.old_link(), sender->msvc_name);
                    for (auto &group : cont_msvcsGroups) {
                        for (auto msvc : group.second.msvcList) {
                            msvc->unpauseThread();
                        }
                    }
                    return;
                }
                int index = std::distance(nb_links.begin(), it);
                config->at("msvc_dnstreamMicroservices")[0]["nb_link"][index] = link;
                if (index != 0) {
                    config->at("msvc_dnstreamMicroservices")[0]["nb_portions"][index-1] = request.data_portion();
                    for (auto postprocessor : postprocessor_dnstreams) {
                        float portion_diff = postprocessor->portions[index] - request.data_portion();
                        postprocessor->portions[0] += portion_diff;
                        postprocessor->portions[index] = request.data_portion();
                    }
                }
                spdlog::get("container_agent")->trace("Overwrote link {0:s} over {1:s}", link, request.old_link());
            } else if (request.mode() == AdjustUpstreamMode::Add) {
                    if (std::find(nb_links.begin(),nb_links.end(), link) == nb_links.end()) {
                        config->at("msvc_dnstreamMicroservices")[0]["nb_link"].push_back(link);
                        config->at("msvc_dnstreamMicroservices")[0]["nb_portions"].push_back(request.data_portion());
                        for (auto postprocessor : postprocessor_dnstreams) {
                            if (postprocessor->portions.empty()) {
                                postprocessor->portions.push_back(1.0f - request.data_portion());
                            } else {
                                postprocessor->portions[0] -= request.data_portion();
                            }
                            postprocessor->portions.push_back(request.data_portion());
                        }
                        spdlog::get("container_agent")->trace("Added link {0:s} to {1:s}", link, sender->msvc_name);
                    } else {
                        spdlog::get("container_agent")->error("Link {0:s} already exists in {1:s}", link, sender->msvc_name);
                        for (auto &group : cont_msvcsGroups) {
                            for (auto msvc : group.second.msvcList) {
                                msvc->unpauseThread();
                            }
                        }
                        return;
                    }
            } else {
                auto it = std::find(nb_links.begin(), nb_links.end(), link);
                if (it == nb_links.end()) {
                    spdlog::get("container_agent")->error("Link {0:s} not found in {1:s}", link, sender->msvc_name);
                    for (auto &group : cont_msvcsGroups) {
                        for (auto msvc : group.second.msvcList) {
                            msvc->unpauseThread();
                        }
                    }
                    return;
                }
                int index = std::distance(nb_links.begin(), it);
                if (request.mode() == AdjustUpstreamMode::Remove) {
                    nb_links.erase(std::remove(nb_links.begin(), nb_links.end(), link), nb_links.end());
                    config->at("msvc_dnstreamMicroservices")[0]["nb_link"] = nb_links;
                    if (index != 0) {
                        config->at("msvc_dnstreamMicroservices")[0]["nb_portions"].erase(
                                config->at("msvc_dnstreamMicroservices")[0]["nb_portions"].begin() + index - 1);
                        for (auto postprocessor: postprocessor_dnstreams) {
                            postprocessor->portions[0] += postprocessor->portions[index];
                            postprocessor->portions.erase(postprocessor->portions.begin() + index - 1);
                        }
                    }
                    spdlog::get("container_agent")->trace("Removed link {0:s} from {1:s}", link, sender->msvc_name);
                } else if (request.mode() == AdjustUpstreamMode::Modify) {
                    sender->dnstreamMicroserviceList[0].portions[index - 1] = request.data_portion();
                    for (auto postprocessor : postprocessor_dnstreams) {
                        float portion_diff = postprocessor->portions[index] - request.data_portion();
                        postprocessor->portions[0] += portion_diff;
                        postprocessor->portions[index] = request.data_portion();
                    }
                    spdlog::get("container_agent")->trace("Modified link {0:s} for {1:s} to portion {2:.2f}", link, sender->msvc_name, request.data_portion());
                    for (auto &group : cont_msvcsGroups) {
                        for (auto msvc : group.second.msvcList) {
                            msvc->unpauseThread();
                        }
                    }
                    return;
                }
            }
            static_cast<Sender*>(sender)->reloadDnstreams();
            static_cast<Sender*>(sender)->dispatchThread();
            for (auto &group : cont_msvcsGroups) {
                for (auto msvc : group.second.msvcList) {
                    msvc->unpauseThread();
                }
            }
            return;
        }
    }
    spdlog::get("container_agent")->error("Could not find sender to {0:s} in current configuration.", request.name());
    for (auto &group : cont_msvcsGroups) {
        for (auto msvc : group.second.msvcList) {
            msvc->unpauseThread();
        }
    }

        // TODO: Add the following again into the logic when enabling GPU Communication
//        if (request.ip() == "localhost") {
//            // change postprocessing to keep the data on gpu
//
//            // start new GPU sender
//            msvcs->push_back(new GPUSender(config));
//        } else {
        // change postprocessing to offload data from gpu
        // start new serialized sender
//        senders->push_back(new_sender);
//        }
        //start the new sender
//        senders->back()->dispatchThread();
}

void ContainerAgent::updateBatchSize(const std::string &msg) {
    indevicemessages::Int32 request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse updateBatchSize request: {0:s}", msg);
        return;
    }
    for (auto msvc : getAllMicroservices()) {
        // The batch size of the data reader (aka FPS) should not be updated by `UpdateBatchSize`
        if (msvc->msvc_type == msvcconfigs::MicroserviceType::DataReader) {
            continue;
        }
        msvc->msvc_idealBatchSize = request.value();
    }
}

void ContainerAgent::updateResolution(const std::string &msg) {
    Dimensions request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse updateResolution request: {0:s}", msg);
        return;
    }
    std::vector<int> resolution = {};
    resolution.push_back(request.channels());
    resolution.push_back(request.height());
    resolution.push_back(request.width());
    if (cont_msvcsGroups["receiver"].msvcList[0]->msvc_type == msvcconfigs::MicroserviceType::DataReader){
        cont_msvcsGroups["receiver"].msvcList[0]->msvc_dataShape = {resolution};
    } else {
        for (auto &preprocessor : cont_msvcsGroups["preprocessor"].msvcList) {
            preprocessor->msvc_dataShape = {resolution};
        }
    }
}

void ContainerAgent::updateTimeKeeping(const std::string &msg) {
    TimeKeeping request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse updateTimeKeeping request: {0:s}", msg);
        return;
    }
    for (auto &preprocessor : cont_msvcsGroups["preprocessor"].msvcList) {
        preprocessor->msvc_pipelineSLO = request.slo();
        preprocessor->msvc_timeBudgetLeft = request.time_budget();
        preprocessor->msvc_contStartTime = request.start_time();
        preprocessor->msvc_contEndTime = request.end_time();
        preprocessor->msvc_localDutyCycle = request.local_duty_cycle();
        preprocessor->msvc_cycleStartTime = ClockType(TimePrecisionType(request.cycle_start_time()));
        preprocessor->updateCycleTiming();
    }
}

void ContainerAgent::transferFrameID(const std::string &msg) {
    indevicemessages::Int32 request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse transferFrameID request: {0:s}", msg);
        return;
    }
    std::string url = absl::StrFormat("tcp://localhost:%d/", request.value());
    socket_t message_queue_pub = socket_t(messaging_ctx, ZMQ_PUB);
    message_queue_pub.connect(url);

    cont_msvcsGroups["receiver"].msvcList[0]->pauseThread();

    request.set_value(cont_msvcsGroups["receiver"].msvcList[0]->msvc_currFrameID);
    std::string req_msg = absl::StrFormat("%s| %s %s", request.name(), MSG_TYPE[TRANSFER_FRAME_ID], request.SerializeAsString());
    message_t zmq_msg(req_msg.size());
    memcpy(zmq_msg.data(), req_msg.data(), req_msg.size());
    if (message_queue_pub.send(zmq_msg, send_flags::dontwait)) {
        run = false;
        stopAllMicroservices();
    } else {
        spdlog::get("container_agent")->error("Failed to send transferFrameID message: {0:s}", req_msg);
    }
}

void ContainerAgent::setFrameID(const std::string &msg) {
    indevicemessages::Int32 request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse setFrameID request: {0:s}", msg);
        return;
    }
    for (auto &receiver : cont_msvcsGroups["receiver"].msvcList) {
        receiver->msvc_currFrameID = request.value() - 1;
        receiver->setReady();
    }
}

bool ContainerAgent::checkPause(std::vector<Microservice *> msvcs) {
    for (auto msvc: msvcs) {
        if (!msvc->checkPause()) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Wait for all the microservices to be paused
 * 
 */
void ContainerAgent::waitPause() {
    bool paused;
    while (true) {
        paused = true;
        spdlog::get("container_agent")->trace("{0:s} waiting for all microservices to be paused.", __func__);
        paused = checkPause(cont_msvcsGroups["receiver"].msvcList) && checkPause(cont_msvcsGroups["preprocessor"].msvcList) &&
                 checkPause(cont_msvcsGroups["inference"].msvcList) && checkPause(cont_msvcsGroups["postprocessor"].msvcList) &&
                 checkPause(cont_msvcsGroups["sender"].msvcList);
        if (paused) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

}

bool ContainerAgent::checkReady(std::vector<Microservice *> msvcs) {
    for (auto msvc: msvcs) {
        if (!msvc->checkReady()) {
            return false;
        }
    }
    return true;
}

// }

/**
 * @brief Wait for all the microservices to be ready
 * 
 */
void ContainerAgent::waitReady() {
    reportStart();
    bool ready = false;
    while (!ready) {
        ready = true;

        spdlog::get("container_agent")->info("{0:s} waiting for all microservices to be ready.", __func__);
        ready = checkReady(cont_msvcsGroups["receiver"].msvcList) && checkReady(cont_msvcsGroups["preprocessor"].msvcList) &&
                checkReady(cont_msvcsGroups["inference"].msvcList) && checkReady(cont_msvcsGroups["postprocessor"].msvcList) &&
                checkReady(cont_msvcsGroups["sender"].msvcList);
        if (ready) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

bool ContainerAgent::stopAllMicroservices() {
    std::lock_guard<std::mutex> lock(cont_pipeStructureMutex);
    for (auto group : cont_msvcsGroups) {
        for (auto msvc : group.second.msvcList) {
            msvc->stopThread();
        }
    }

    return true;
}

std::vector<Microservice *> ContainerAgent::getAllMicroservices() {
    std::vector<Microservice *> allMsvcs;
    std::lock_guard<std::mutex> lock(cont_pipeStructureMutex);
    for (auto group : cont_msvcsGroups) {
        for (auto msvc : group.second.msvcList) {
            allMsvcs.push_back(msvc);
        }
    }
    return allMsvcs;
}