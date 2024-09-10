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

const int INDEVICE_CONTROL_PORT = 60003;

std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::milliseconds> timePointCastMillisecond(
    std::chrono::system_clock::time_point tp) {
    return std::chrono::time_point_cast<std::chrono::milliseconds>(tp);
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

json loadRunArgs(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    int8_t device = (int8_t) absl::GetFlag(FLAGS_device);
    uint16_t logLevel = absl::GetFlag(FLAGS_verbose);
    uint16_t loggingMode = absl::GetFlag(FLAGS_logging_mode);
    std::string logPath = absl::GetFlag(FLAGS_log_dir);
    uint16_t profiling_mode = absl::GetFlag(FLAGS_profiling_mode);

    RUNMODE runmode = static_cast<RUNMODE>(profiling_mode);

    std::tuple<json, json> configs = msvcconfigs::loadJson();
    json containerConfigs = std::get<0>(configs);
    std::string name = containerConfigs["cont_name"];
    json profilingConfigs = std::get<1>(configs);

    BatchSizeType minBatch = profilingConfigs.at("profile_minBatch");
    std::string templateModelPath = profilingConfigs.at("profile_templateModelPath");

    /**
     * @brief     If this is profiling, set configurations to the first batch size that should be profiled
     * This includes
     * 1. Setting its name based on the template model path
     * 2. Setting the batch size to the smallest profile batch size
     * 
     */
    if (profiling_mode == 2) {

        name = removeSubstring(templateModelPath, ".engine");
        name = replaceSubstring(name, "[batch]", std::to_string(minBatch));
        name = splitString(name, "/").back();
        logPath = "../model_profiles";
    }

    logPath += "/" + containerConfigs["cont_experimentName"].get<std::string>();
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    logPath += "/" + containerConfigs["cont_systemName"].get<std::string>();
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    logPath += "/" + containerConfigs["cont_pipeName"].get<std::string>() + "_" + name;
    std::filesystem::create_directory(
        std::filesystem::path(logPath)
    );

    containerConfigs["cont_device"] = device;
    containerConfigs["cont_logLevel"] = logLevel;
    containerConfigs["cont_loggingMode"] = loggingMode;
    containerConfigs["cont_logPath"] = logPath;
    containerConfigs["cont_RUNMODE"] = runmode;
    containerConfigs["cont_port"] = absl::GetFlag(FLAGS_port);

    std::ifstream metricsServerCfgsFile = std::ifstream(containerConfigs.at("cont_metricServerConfigs"));
    json metricsServerConfigs = json::parse(metricsServerCfgsFile);

    containerConfigs["cont_metricsServerConfigs"] = metricsServerConfigs;
    if (containerConfigs["cont_taskName"] != "dsrc") {
        containerConfigs["cont_inferModelName"] = splitString(containerConfigs.at("cont_pipeline")[2]["path"], "/").back();
        containerConfigs["cont_inferModelName"] = splitString(containerConfigs["cont_inferModelName"], ".").front();
        // The maximum batch size supported by the model (for TensorRT)
        std::vector<std::string> modelOptions = splitString(containerConfigs["cont_inferModelName"], "_");
        BatchSizeType maxModelBatchSize = std::stoull(modelOptions[modelOptions.size() - 2]);
        if (static_cast<RUNMODE>(runmode) == RUNMODE::PROFILING) {
            containerConfigs["cont_maxBatchSize"] = std::min((BatchSizeType)profilingConfigs["profile_maxBatch"], maxModelBatchSize);
        } else if (static_cast<RUNMODE>(runmode) == RUNMODE::DEPLOYMENT) {
            containerConfigs["cont_maxBatchSize"] = maxModelBatchSize;
        }
    }

    for (uint16_t i = 0; i < containerConfigs["cont_pipeline"].size(); i++) {
        containerConfigs.at("cont_pipeline")[i]["msvc_contSLO"] = containerConfigs["cont_SLO"];
        containerConfigs.at("cont_pipeline")[i]["msvc_contStartTime"] = containerConfigs["cont_startTime"];
        containerConfigs.at("cont_pipeline")[i]["msvc_contEndTime"] = containerConfigs["cont_endTime"];
        containerConfigs.at("cont_pipeline")[i]["msvc_localDutyCycle"] = containerConfigs["cont_localDutyCycle"];
        containerConfigs.at("cont_pipeline")[i]["msvc_cycleStartTime"] = containerConfigs["cont_cycleStartTime"];
        containerConfigs.at("cont_pipeline")[i]["msvc_batchMode"] = containerConfigs["cont_batchMode"];
        containerConfigs.at("cont_pipeline")[i]["msvc_dropMode"] = containerConfigs["cont_dropMode"];
        containerConfigs.at("cont_pipeline")[i]["msvc_timeBudgetLeft"] = containerConfigs["cont_timeBudgetLeft"];
        containerConfigs.at("cont_pipeline")[i]["msvc_pipelineSLO"] = containerConfigs["cont_pipelineSLO"];
        containerConfigs.at("cont_pipeline")[i]["msvc_experimentName"] = containerConfigs["cont_experimentName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_systemName"] = containerConfigs["cont_systemName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_contName"] = name;
        containerConfigs.at("cont_pipeline")[i]["msvc_pipelineName"] = containerConfigs["cont_pipeName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_taskName"] = containerConfigs["cont_taskName"];
        containerConfigs.at("cont_pipeline")[i]["msvc_hostDevice"] = containerConfigs["cont_hostDevice"];
        containerConfigs.at("cont_pipeline")[i]["msvc_deviceIndex"] = device;
        containerConfigs.at("cont_pipeline")[i]["msvc_containerLogPath"] = containerConfigs["cont_logPath"].get<std::string>() + "/" + name;
        containerConfigs.at("cont_pipeline")[i]["msvc_RUNMODE"] = runmode;
        containerConfigs.at(
                "cont_pipeline")[i]["cont_metricsScrapeIntervalMillisec"] = metricsServerConfigs["metricsServer_metricsReportIntervalMillisec"];
        containerConfigs.at("cont_pipeline")[i]["msvc_numWarmUpBatches"] = containerConfigs.at("cont_numWarmUpBatches");
        if (containerConfigs["cont_taskName"] != "dsrc") {
            containerConfigs.at("cont_pipeline")[i]["msvc_maxBatchSize"] = containerConfigs.at("cont_maxBatchSize");
            containerConfigs.at("cont_pipeline")[i]["msvc_allocationMode"] = containerConfigs.at("cont_allocationMode");
        }

        /**
         * @brief     If this is profiling, set configurations to the first batch size that should be profiled
         * This includes
         * 1. Setting its profile dir whose name is based on the template model path
         * 2. Setting the batch size to the smallest profile batch size
         * 
         */
        if (profiling_mode == 1 && containerConfigs["cont_taskName"] != "dsrc" && containerConfigs["cont_taskName"] != "datasource") {
            addProfileConfigs(containerConfigs.at("cont_pipeline")[i], profilingConfigs);
            
        } else if (profiling_mode == 2) {
            containerConfigs.at("cont_pipeline")[i].at("msvc_idealBatchSize") = minBatch;
            if (i == 0) {
                addProfileConfigs(containerConfigs.at("cont_pipeline")[i], profilingConfigs);
            } else if (i == 2) {
                // Set the path to the engine
                containerConfigs.at("cont_pipeline")[i].at("path") = replaceSubstring(templateModelPath, "[batch]",
                                                                                      std::to_string(minBatch));
            }
        }

        if (i == 1) {
            containerConfigs.at("cont_pipeline")[i]["msvc_modelProfile"] = containerConfigs.at("cont_modelProfile");
        }
    }

    json finalConfigs;
    finalConfigs["container"] = containerConfigs;
    finalConfigs["profiling"] = profilingConfigs;

    if (containerConfigs["cont_taskName"] != "dsrc") {
        checkCudaErrorCode(cudaSetDevice(device), __func__);
    }

    return finalConfigs;
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

std::tuple<json, json> msvcconfigs::loadJson() {
    json containerConfigs, profilingConfigs;
    if (!absl::GetFlag(FLAGS_json).has_value()) {
        spdlog::trace("{0:s} attempts to load Json Configs from file.", __func__);
        if (absl::GetFlag(FLAGS_json_path).has_value()) {
            std::ifstream file(absl::GetFlag(FLAGS_json_path).value());
            auto json_file = json::parse(file);
            containerConfigs = json_file.at("container");
            try {
                profilingConfigs = json_file.at("profiling");
            } catch (json::out_of_range &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            } catch (json::parse_error &e) {
                spdlog::error("{0:s} Error parsing json file.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from file.", __func__);
            return std::make_tuple(containerConfigs, profilingConfigs);
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
            containerConfigs = json_file.at("container");
            try {
                profilingConfigs = json_file.at("profiling");
            } catch (json::out_of_range &e) {
                spdlog::trace("{0:s} No profiling configurations found.", __func__);
            }
            spdlog::trace("{0:s} finished loading Json Configs from command line.", __func__);
            return std::make_tuple(containerConfigs, profilingConfigs);
        }
    }
}

void ContainerAgent::profiling(const json &pipeConfigs, const json &profileConfigs) {

    json pipelineConfigs = pipeConfigs;

    BatchSizeType minBatch = profileConfigs.at("profile_minBatch");
    BatchSizeType maxBatch = profileConfigs.at("profile_maxBatch");
    uint8_t stepMode = profileConfigs.at("profile_stepMode");
    uint8_t step = profileConfigs.at("profile_step");
    std::string templateModelPath = profileConfigs.at("profile_templateModelPath");

    std::thread metricsThread(&ContainerAgent::collectRuntimeMetrics, this);
    metricsThread.detach();

    this->dispatchMicroservices();

    for (BatchSizeType batch = minBatch; batch <= maxBatch;) {
        spdlog::get("container_agent")->trace("{0:s} model with a max batch of {1:d}.", __func__, batch);
        if (batch != minBatch) {
            std::string profileDirPath, cont_name;

            cont_name = removeSubstring(templateModelPath, ".engine");
            cont_name = replaceSubstring(cont_name, "[batch]", std::to_string(batch));
            cont_name = splitString(cont_name, "/").back();

            profileDirPath = cont_logDir + "/" + cont_name;
            std::filesystem::create_directory(
                    std::filesystem::path(profileDirPath)
            );

            // Making sure all the microservices are paused before reloading and reallocating resources
            // this is essential to avoiding runtime memory errors
            for (uint8_t i = 0; i < cont_msvcsList.size(); i++) {
                cont_msvcsList[i]->pauseThread();
            }
            waitPause();

            // Reload the configurations and dynamic allocation based on the new configurations
            for (uint8_t i = 0; i < cont_msvcsList.size(); i++) {
                pipelineConfigs[i].at("msvc_idealBatchSize") = batch;
                pipelineConfigs[i].at("msvc_containerLogPath") = profileDirPath;
                pipelineConfigs[i].at("msvc_deviceIndex") = cont_deviceIndex;
                pipelineConfigs[i].at("msvc_contName") = cont_name;
                // Set the path to the engine
                if (i == 2) {
                    pipelineConfigs[i].at("path") = replaceSubstring(templateModelPath, "[batch]",
                                                                     std::to_string(batch));
                }
                cont_msvcsList[i]->loadConfigs(pipelineConfigs[i], false);
                cont_msvcsList[i]->setRELOAD();
            }

        }

        this->waitReady();
        this->PROFILING_START(batch);

        for (int i = 1; i <= batch; i *= 2) {
            for (auto msvc: cont_msvcsList) {
                msvc->msvc_idealBatchSize = i;
            }
            while (true) {
                spdlog::get("container_agent")->info("{0:s} waiting for profiling of model with a max batch of {1:d} and real batch of {2:d}.",
                             __func__, batch, i);
                if (cont_msvcsList[0]->checkPause()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }

        spdlog::get("container_agent")->info("===============================================================================================");
        if (stepMode == 0) {
            batch += step;
        } else {
            batch *= 2;
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
        cont_batchInferProfileList[batch].p95inferLat = profile[i + 3].get<BatchSizeType>();

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
    cont_taskName = containerConfigs["cont_taskName"].get<std::string>();
    cont_hostDevice = containerConfigs["cont_hostDevice"].get<std::string>();
    cont_hostDeviceType = containerConfigs["cont_hostDeviceType"].get<std::string>();
    cont_systemName = containerConfigs["cont_systemName"].get<std::string>();

    cont_deviceIndex = containerConfigs["cont_device"];

    cont_RUNMODE = containerConfigs["cont_RUNMODE"];

    cont_logDir = containerConfigs["cont_logPath"].get<std::string>();

    setupLogger(
        cont_logDir,
        cont_name,
        containerConfigs["cont_loggingMode"],
        containerConfigs["cont_logLevel"],
        cont_loggerSinks,
        cont_logger
    );

    // bool readProfile = readModelProfile(containerConfigs["cont_modelProfile"]);

    // if (!readProfile && cont_RUNMODE == RUNMODE::DEPLOYMENT && cont_taskName != "dsrc" && cont_taskName != "datasource") {
    //     spdlog::get("container_agent")->error("{0:s} No model profile found.", __func__);
    //     exit(1);
    // }

    // if (cont_RUNMODE == RUNMODE::EMPTY_PROFILING) {
    //     // Create the logDir for this container
    //     cont_logDir = (std::string) containerConfigs.at("cont_logPath");
    //     std::filesystem::create_directory(
    //             std::filesystem::path(cont_logDir)
    //     );
    // } else {
    //     cont_logDir = (std::string) containerConfigs["cont_logPath"];
    // }

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
            // cont_batchInferProfileList = queryBatchInferLatency(
            //     *cont_metricsServerConn,
            //     cont_experimentName,
            //     cont_systemName,
            //     cont_pipeName,
            //     "stream",
            //     cont_inferModel,
            //     cont_hostDeviceType,
            //     cont_inferModel
            // );
            
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
         * @brief Table for full arrival records, hence the suffix `_f`
         * 
         */
        // std::string tableName = cont_arrivalTableName + "_f";
        // if (!tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, tableName)) {
        //     sql_statement = "CREATE TABLE IF NOT EXISTS " + tableName + " (arrival_timestamps BIGINT NOT NULL, "
        //                                                                         "model_name TEXT NOT NULL, "
        //                                                                         "stream TEXT NOT NULL, "
        //                                                                         "sender_host TEXT NOT NULL, "
        //                                                                         "receiver_host TEXT NOT NULL, "
        //                                                                         "out_queueing_duration_us BIGINT NOT NULL, "
        //                                                                         "transfer_duration_us BIGINT NOT NULL, "
        //                                                                         "queueing_duration_us BIGINT NOT NULL, "
        //                                                                         "total_package_size_b INTEGER NOT NULL, "
        //                                                                         "request_size_b INTEGER NOT NULL)";

        //     pushSQL(*cont_metricsServerConn, sql_statement);

        //     sql_statement = "SELECT create_hypertable('" + tableName + "', 'arrival_timestamps', if_not_exists => TRUE);";
            
        //     pushSQL(*cont_metricsServerConn, sql_statement);

        //     sql_statement = "CREATE INDEX ON " + tableName + " (arrival_timestamps);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);
            
        //     sql_statement = "CREATE INDEX ON " + tableName + " (stream);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);

        //     sql_statement = "CREATE INDEX ON " + tableName + " (sender_host);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);

        //     sql_statement = "CREATE INDEX ON " + tableName + " (receiver_host);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);
        // }

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
                             "p95_total_package_size_b INTEGER NOT NULL)";

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
         * @brief Table for full process records, hence the suffix `_f`
         * 
         */
        // tableName = cont_processTableName + "_f";
        // if (!tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, tableName)) {
        //     sql_statement = "CREATE TABLE IF NOT EXISTS " + tableName + " ("
        //                                                                         "postprocess_timestamps BIGINT NOT NULL, "
        //                                                                         "stream TEXT NOT NULL, "
        //                                                                         "prep_duration_us INTEGER NOT NULL, "
        //                                                                         "batch_duration_us INTEGER NOT NULL, "
        //                                                                         "infer_duration_us INTEGER NOT NULL, "
        //                                                                         "post_duration_us INTEGER NOT NULL, "
        //                                                                         "infer_batch_size INT2 NOT NULL, "
        //                                                                         "input_size_b INTEGER NOT NULL, "
        //                                                                         "output_size_b INTEGER NOT NULL)";

        //     pushSQL(*cont_metricsServerConn, sql_statement);

        //     sql_statement = "SELECT create_hypertable('" + tableName + "', 'postprocess_timestamps', if_not_exists => TRUE);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);
            
        //     sql_statement = "CREATE INDEX ON " + tableName + " (postprocess_timestamps);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);

        //     sql_statement += "CREATE INDEX ON " + tableName + " (stream);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);

        //     sql_statement += "CREATE INDEX ON " + tableName + " (infer_batch_size);";
        //     pushSQL(*cont_metricsServerConn, sql_statement);
        // }

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

    setenv("GRPC_DNS_RESOLVER", "native", 1);

    int own_port = containerConfigs.at("cont_port");

    std::string server_address = absl::StrFormat("%s:%d", "0.0.0.0", own_port);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    server_cq = builder.AddCompletionQueue();
    server = builder.BuildAndStart();

    server_address = absl::StrFormat("%s:%d", "localhost", INDEVICE_CONTROL_PORT  + absl::GetFlag(FLAGS_port_offset));
    stub = InDeviceCommunication::NewStub(grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));
    sender_cq = new CompletionQueue();

    run = true;
    reportHwMetrics = false;
    profiler = nullptr;
    std::thread receiver(&ContainerAgent::HandleRecvRpcs, this);
    receiver.detach();
}

void ContainerAgent::ReportStart() {
    ProcessData request;
    request.set_msvc_name(cont_name);
    ProcessData reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<ProcessData>> rpc(
            stub->AsyncReportMsvcStart(&context, request, sender_cq));
    finishGrpc(rpc, reply, status, sender_cq);
    pid = reply.pid();
    spdlog::get("container_agent")->info("Container Agent started with pid: {0:d}", pid);
    if (cont_taskName != "dsrc" && cont_taskName != "sink" && cont_RUNMODE == RUNMODE::PROFILING) {
        profiler = new Profiler({pid});
        reportHwMetrics = true;
    }
}


void ContainerAgent::runService(const json &pipeConfigs, const json &configs) {
    if (configs["container"]["cont_RUNMODE"] == RUNMODE::EMPTY_PROFILING) {
        profiling(pipeConfigs, configs["profiling"]);
    } else {
        this->dispatchMicroservices();

        this->waitReady();
        this->START();

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
    unsigned int lateCount, queueDrops, totalRequests;
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

    if (cont_msvcsList[0]->msvc_type == MicroserviceType::DataReader) {
        while (run) {
            if (cont_msvcsList[0]->STOP_THREADS) {
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
                cont_hwMetrics = {stats.cpuUsage, stats.memoryUsage, stats.rssMemory, stats.gpuUtilization,
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

        if (timePointCastMillisecond(startTime) >= timePointCastMillisecond(cont_metricsServerConfigs.nextArrivalRateScrapeTime)) {;
            perSecondArrivalRecords.addRecord(cont_msvcsList[0]->getPerSecondArrivalRecord());
            // secondIndex = (secondIndex + 1) % maxNumSeconds;
            metricsStopwatch.stop();
            auto localScrapeLatencyMilisec = (uint64_t) std::ceil(metricsStopwatch.elapsed_microseconds() / 1000.f);
            scrapeLatencyMillisec += localScrapeLatencyMilisec;

            cont_metricsServerConfigs.nextArrivalRateScrapeTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(1000 - localScrapeLatencyMilisec);
            metricsStopwatch.start();
        }

        startTime = std::chrono::high_resolution_clock::now();
        if (timePointCastMillisecond(startTime) >= 
                timePointCastMillisecond(cont_metricsServerConfigs.nextMetricsReportTime)) {
            Stopwatch pushMetricsStopWatch;
            pushMetricsStopWatch.start();
            lateCount = cont_msvcsList[0]->GetDroppedReqCount();
            totalRequests = cont_msvcsList[0]->GetTotalReqCount();
            queueDrops = 0;
            for (auto msvc: cont_msvcsList) {
                queueDrops += msvc->GetQueueDrops();
            }

            spdlog::get("container_agent")->info("{0:s} had {1:d} late requests of {2:d} total requests. ({3:d} queue drops)", cont_name, lateCount, totalRequests, queueDrops);

            std::string modelName = cont_msvcsList[2]->getModelName();
            if (cont_RUNMODE == RUNMODE::PROFILING) {
                if (reportHwMetrics && cont_hwMetrics.metricsAvailable) {
                    sql = "INSERT INTO " + cont_hwMetricsTableName +
                        " (timestamps, batch_size, cpu_usage, mem_usage, rss_mem_usage, gpu_usage, gpu_mem_usage) VALUES ";
                    sql += "(" + timePointToEpochString(std::chrono::high_resolution_clock::now()) + ", ";
                    sql += std::to_string(cont_msvcsList[1]->msvc_idealBatchSize) + ", ";
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
                if (cont_msvcsList[0]->STOP_THREADS) {
                    run = false;
                    continue;
                }
            }

            updateArrivalRecords(perSecondArrivalRecords);
            updateProcessRecords();

            batchInferRecords = cont_msvcsList[3]->getBatchInferRecords();
            for (auto& [keys, records] : batchInferRecords) {
                uint32_t numEntries = records.inferDuration.size();
                // Check if there are any records
                if (numEntries == 0) { continue; }

                std::string reqOriginStream = keys.first;
                BatchSizeType inferBatchSize = keys.second;

                std::map<uint8_t, PercentilesBatchInferRecord> percentilesRecord = records.findPercentileAll({95});

                // Construct the SQL statement
                sql = absl::StrFormat("INSERT INTO %s (timestamps, stream, infer_batch_size, p95_infer_duration_us) "
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

            pushMetricsStopWatch.stop();
            auto pushMetricsLatencyMillisec = (uint64_t) std::ceil(pushMetricsStopWatch.elapsed_microseconds() / 1000.f);
            spdlog::get("container_agent")->trace("{0:s} pushed BATCH INFER METRICS to the database", cont_name);
            spdlog::get("container_agent")->trace("{0:s} pushed ALL METRICS to the database. Latency {1:d}ms. Next push in {2:d}ms",
                                                 cont_name,
                                                 pushMetricsLatencyMillisec, 
                                                 cont_metricsServerConfigs.metricsReportIntervalMillisec - pushMetricsLatencyMillisec);
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
        timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextTime - std::chrono::high_resolution_clock::now()).count();
        std::chrono::milliseconds sleepPeriod(timeDiff - (reportLatencyMillisec) + 2);
        spdlog::get("container_agent")->trace("{0:s} Container Agent's Metric Reporter sleeps for {1:d} milliseconds.", cont_name, sleepPeriod.count());
        std::this_thread::sleep_for(sleepPeriod);
    }

    for (auto msvc: cont_msvcsList) {
        msvc->stopThread();
    }
}

void ContainerAgent::updateArrivalRecords(RunningArrivalRecord &perSecondArrivalRecords) {
    std::string sql;
    ArrivalRecordType arrivalRecords = cont_msvcsList[3]->getArrivalRecords();
    // Keys value here is std::pair<std::string, std::string> for stream and sender_host
    NetworkRecordType networkRecords;
    for (auto &[keys, records]: arrivalRecords) {
        uint32_t numEntries = records.arrivalTime.size();
        if (numEntries == 0) {
            continue;
        }

        std::string stream = keys.first;
        std::string senderHostAbbr = abbreviate(keys.second);

        std::vector<uint8_t> percentiles = {95};
        std::map<uint8_t, PercentilesArrivalRecord> percentilesRecord = records.findPercentileAll(percentiles);

        sql = absl::StrFormat("INSERT INTO %s (timestamps, stream, model_name, sender_host, receiver_host, ", cont_arrivalTableName);

        for (auto &period : cont_metricsServerConfigs.queryArrivalPeriodMillisec) {
            sql += "arrival_rate_" + std::to_string(period/1000) + "s, ";
            sql += "coeff_var_" + std::to_string(period/1000) + "s, ";
        }
        perSecondArrivalRecords.aggregateArrivalRecord(cont_metricsServerConfigs.queryArrivalPeriodMillisec);
        std::vector<float> requestRates = perSecondArrivalRecords.getArrivalRatesInPeriods();
        std::vector<float> coeffVars = perSecondArrivalRecords.getCoeffVarsInPeriods();
        sql += absl::StrFormat("p95_out_queueing_duration_us, p95_transfer_duration_us, p95_queueing_duration_us, p95_total_package_size_b) "
                               "VALUES ('%s', '%s', '%s', '%s', '%s'",
                               timePointToEpochString(std::chrono::system_clock::now()),
                               stream,
                               cont_inferModel,
                               senderHostAbbr,
                               abbreviate(cont_hostDevice));
        for (auto i = 0; i < requestRates.size(); i++) {
            sql += ", " + std::to_string(requestRates[i]);
            sql += ", " + std::to_string(coeffVars[i]);
        }
        sql += absl::StrFormat(", %ld, %ld, %ld, %d);",
                               percentilesRecord[95].outQueueingDuration,
                               percentilesRecord[95].transferDuration,
                               percentilesRecord[95].queueingDuration,
                               percentilesRecord[95].totalPkgSize);

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

    arrivalRecords.clear();
    spdlog::get("container_agent")->trace("{0:s} pushed arrival metrics to the database.", cont_name);
}

void ContainerAgent::updateProcessRecords() {
    ProcessRecordType processRecords = cont_msvcsList[3]->getProcessRecords();
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
}

void ContainerAgent::updateProfileTable() {
    std::string profileTableName = abbreviate("prof__" + cont_inferModel + "__" + cont_hostDeviceType);
    std::string procTableName = profileTableName + "_proc";
    std::string hwTableName = profileTableName + "_hw";
    
    BatchInferProfileListType batchInferProfile;

    pqxx::nontransaction curl(*cont_metricsServerConn);
    std::string query = absl::StrFormat(
        "SELECT "
        "   infer_batch_size, "
        "   percentile_disc(0.95) WITHIN GROUP (ORDER BY infer_duration) AS p95_infer_duration "
        "FROM %s "
        "GROUP BY infer_batch_size", procTableName
    );
    pqxx::result res = curl.exec(query);
    for (const auto& row : res) {
        BatchSizeType batchSize = row[0].as<BatchSizeType>();
        batchInferProfile[batchSize].p95inferLat = (uint64_t)(row[1].as<uint64_t>() / batchSize);
    }

    query = absl::StrFormat(
        "SELECT "
        "   batch_size, "
        "   MAX(cpu_usage) AS cpu_usage, "
        "   MAX(mem_usage) AS mem_usage, "
        "   MAX(rss_mem_usage) AS rss_mem_usage, "
        "   MAX(gpu_usage) AS gpu_usage, "
        "   MAX(gpu_mem_usage) AS gpu_mem_usage "
        "FROM %s "
        "GROUP BY batch_size", hwTableName
    );
    res =  curl.exec(query);
    for (const auto& row : res) {
        BatchSizeType batchSize = row[0].as<BatchSizeType>();
        batchInferProfile[batchSize].cpuUtil = row[1].as<CpuUtilType>();
        batchInferProfile[batchSize].memUsage = row[2].as<MemUsageType>();
        batchInferProfile[batchSize].rssMemUsage = row[3].as<MemUsageType>();
        batchInferProfile[batchSize].gpuUtil = row[4].as<GpuUtilType>();
        batchInferProfile[batchSize].gpuMemUsage = row[5].as<GpuMemUsageType>();
    }

    // Delete old profile entries
    if (tableExists(*cont_metricsServerConn, cont_metricsServerConfigs.schema, profileTableName)) {
        query = "DROP TABLE " + profileTableName + ";";
        pushSQL(*cont_metricsServerConn, query);
    }
    query = absl::StrFormat(
        "CREATE TABLE %s ("
        "   infer_batch_size INT PRIMARY KEY, "
        "   p95_infer_duration BIGINT NOT NULL, "
        "   cpu_usage INT2 NOT NULL, "
        "   mem_usage INT4 NOT NULL, "
        "   rss_mem_usage INT4 NOT NULL, "
        "   gpu_usage INT2 NOT NULL, "
        "   gpu_mem_usage INT4 NOT NULL"
        ");", profileTableName
    );
    pushSQL(*cont_metricsServerConn, query);

    // Insert new profile entries
    query = absl::StrFormat(
        "INSERT INTO %s (infer_batch_size, p95_infer_duration, cpu_usage, mem_usage, rss_mem_usage, gpu_usage, gpu_mem_usage) "
        "VALUES ", profileTableName
    );
    for (const auto& [batchSize, profile] : batchInferProfile) {
        query += absl::StrFormat(
            "(%d,%d,%d,%ld,%ld, %d,%ld),",
            batchSize, profile.p95inferLat, (int) profile.cpuUtil,
            profile.memUsage, profile.rssMemUsage, profile.gpuUtil, profile.gpuMemUsage
        );
    }

    pushSQL(*cont_metricsServerConn, query.substr(0, query.size() - 1) + ";");
}

void ContainerAgent::HandleRecvRpcs() {
    new StopRequestHandler(&service, server_cq.get(), &run);
    new UpdateSenderRequestHandler(&service, server_cq.get(), &cont_msvcsList);
    new UpdateBatchSizeRequestHandler(&service, server_cq.get(), &cont_msvcsList);
    new UpdateResolutionRequestHandler(&service, server_cq.get(), this);
    new UpdateTimeKeepingRequestHandler(&service, server_cq.get(), this);
    new SyncDatasourcesRequestHandler(&service, server_cq.get(), this);
    void *tag;
    bool ok;
    while (run) {
        if (!server_cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}

void ContainerAgent::StopRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestStopExecution(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        *run = false;
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::UpdateSenderRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateSender(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateSenderRequestHandler(service, cq, msvcs);
        // pause processing except senders to clear out the queues
        for (auto msvc: *msvcs) {
            if (msvc->dnstreamMicroserviceList[0].name == request.name()) {
                continue;
            }
            msvc->pauseThread();
        }
        json config;
        std::vector<ThreadSafeFixSizedDoubleQueue *> inqueue;
        for (auto msvc: *msvcs) {
            if (msvc->dnstreamMicroserviceList[0].name == request.name()) {
                config = msvc->msvc_configs;
                config["msvc_dnstreamMicroservices"][0]["nb_link"][0] = absl::StrFormat("%s:%d", request.ip(),
                                                                                        request.port());
                inqueue = msvc->GetInQueue();
                msvc->stopThread();
                msvcs->erase(std::remove(msvcs->begin(), msvcs->end(), msvc), msvcs->end());
                break;
            }
        }
//        if (request.ip() == "localhost") {
//            // change postprocessing to keep the data on gpu
//
//            // start new GPU sender
//            msvcs->push_back(new GPUSender(config));
//        } else {
        // change postprocessing to offload data from gpu

        // start new serialized sender
        msvcs->push_back(new RemoteCPUSender(config));
//        }
        // align the data queue from postprocessor to new sender
        msvcs->back()->SetInQueue(inqueue);
        //start the new sender
        msvcs->back()->dispatchThread();
        for (auto msvc: *msvcs) {
            if (msvc->dnstreamMicroserviceList[0].name == request.name()) {
                continue;
            }
            msvc->unpauseThread();
        }

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::UpdateBatchSizeRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateBatchSize(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateBatchSizeRequestHandler(service, cq, msvcs);
        for (auto msvc : *msvcs) {
            // The batch size of the data reader (aka FPS) should be updated by `UpdateBatchSizeRequestHandler`
            if (msvc->msvc_type == msvcconfigs::MicroserviceType::DataReader) {
                continue;
            }
            msvc->msvc_idealBatchSize = request.value();
        }
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::UpdateResolutionRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateResolution(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateResolutionRequestHandler(service, cq, container_agent);
        std::vector<int> resolution = {};
        resolution.push_back(request.channels());
        resolution.push_back(request.height());
        resolution.push_back(request.width());
        if (container_agent->cont_msvcsList[0]->msvc_type == msvcconfigs::MicroserviceType::DataReader){
            container_agent->cont_msvcsList[0]->msvc_dataShape = {resolution};
        } else {
            container_agent->cont_msvcsList[1]->dnstreamMicroserviceList[0].expectedShape = {resolution};
        }

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::UpdateTimeKeepingRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestUpdateTimeKeeping(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        new UpdateTimeKeepingRequestHandler(service, cq, container_agent);
        container_agent->cont_msvcsList[1]->msvc_contSLO = request.cont_slo();
        container_agent->cont_msvcsList[1]->msvc_pipelineSLO = request.slo();
        container_agent->cont_msvcsList[1]->msvc_timeBudgetLeft = request.time_budget();
        container_agent->cont_msvcsList[1]->msvc_contStartTime = request.start_time();
        container_agent->cont_msvcsList[1]->msvc_contEndTime = request.end_time();
        container_agent->cont_msvcsList[1]->msvc_localDutyCycle = request.local_duty_cycle();
        container_agent->cont_msvcsList[1]->msvc_cycleStartTime = ClockType(TimePrecisionType(request.cycle_start_time()));
        container_agent->cont_msvcsList[1]->updateCycleTiming();
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void ContainerAgent::transferFrameID(std::string url) {
    indevicecommunication::Int32 request;
    EmptyMessage reply;
    ClientContext context;
    Status status;
    auto dsrc_stub = InDeviceCommunication::NewStub(grpc::CreateChannel(url, grpc::InsecureChannelCredentials()));
    auto dsrc_cq = new CompletionQueue();
    cont_msvcsList[0]->pauseThread();
    request.set_value(cont_msvcsList[0]->msvc_currFrameID);
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            dsrc_stub->AsyncSyncDatasources(&context, request, dsrc_cq));
    finishGrpc(rpc, reply, status, dsrc_cq);
    run = false;
    for (auto msvc: cont_msvcsList) {
        msvc->stopThread();
    }
}

void ContainerAgent::SyncDatasourcesRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSyncDatasources(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        containerAgent->transferFrameID(absl::StrFormat("localhost:%d/", request.value()));
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

/**
 * @brief Check if all the microservices are paused
 * 
 * @return true 
 * @return false 
 */
bool ContainerAgent::checkPause() {
    for (auto msvc: cont_msvcsList) {
        if (msvc->checkPause()) {
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
        for (auto msvc: cont_msvcsList) {
            if (!msvc->checkPause()) {
                paused = false;
                break;
            }
        }
        if (paused) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

}

/**
 * @brief Check if all the microservices are ready
 * 
 * @return true 
 * @return false 
 */
bool ContainerAgent::checkReady() {
    for (auto msvc: cont_msvcsList) {
        if (!msvc->checkReady()) {
            return true;
        }
    }
    return true;

}

/**
 * @brief Wait for all the microservices to be ready
 * 
 */
void ContainerAgent::waitReady() {
    ReportStart();
    bool ready = false;
    while (!ready) {
        ready = true;

        spdlog::get("container_agent")->info("{0:s} waiting for all microservices to be ready.", __func__);
        for (auto msvc: cont_msvcsList) {
            if (!msvc->checkReady()) {
                ready = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}
