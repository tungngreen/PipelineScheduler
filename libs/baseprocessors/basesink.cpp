#include "baseprocessor.h"

using namespace spdlog;

void BaseSink::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOANDING configs...", __func__);
    if (!isConstructing) {
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }
    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);
}

BaseSink::BaseSink(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_name = "sink";
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", __func__);
}

void BaseSink::sink() {
    Request<LocalCPUReqDataType> inferTimeReport;
    BatchSizeType batchSize;

    while (true) {
        if (STOP_THREADS) {
            if (STOP_THREADS) {
                spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
                break;
            }
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                /**
                 * @brief Opening a new log file
                 * During runtime: log file should come with a new timestamp everytime the microservice is reloaded
                 * 
                 */

                if (msvc_logFile.is_open()) {
                    msvc_logFile.close();
                }
                msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);

                setDevice();
                RELOADING = false;
                READY = true;
                spdlog::get("container_agent")->info("{0:s} is reloaded.", msvc_name);
            }
            continue;
        }
        inferTimeReport = msvc_InQueue.at(0)->pop1();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(inferTimeReport.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        }

        /**
         * @brief During profiling mode, there are six important timestamps to be recorded:
         * 1. When the request was generated
         * 2. When the request was received by the preprocessor
         * 3. When the request was done preprocessing by the preprocessor
         * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
         * 5. When the batch inferencer was completed by the inferencer 
         * 6. When each request was completed by the postprocessor
         */
        batchSize = inferTimeReport.req_batchSize;

        if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
            for (BatchSizeType i = 0; i < batchSize; i++) {
                msvc_logFile << inferTimeReport.req_travelPath[i] << "|";
                for (unsigned int j = 0; j < inferTimeReport.req_origGenTime[i].size() - 1; j++) {
                    msvc_logFile << timePointToEpochString(inferTimeReport.req_origGenTime[i].at(j)) << ",";
                }
                msvc_logFile << timePointToEpochString(inferTimeReport.req_origGenTime[i].back()) << "|";

                for (BatchSizeType j = 1; j < inferTimeReport.req_origGenTime[i].size(); j++) {
                    msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(inferTimeReport.req_origGenTime[i].at(j) - inferTimeReport.req_origGenTime[i].at(j-1)).count() << ",";
                }
                msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(inferTimeReport.req_origGenTime[i].back() - inferTimeReport.req_origGenTime[i].front()).count() << std::endl;
            }

        /**
         * @brief 
         * 
         */
        } else if (msvc_RUNMODE == RUNMODE::DEPLOYMENT || msvc_RUNMODE == RUNMODE::PROFILING) {
            spdlog::get("container_agent")->trace("{0:s} is processing request {1:s}...", msvc_name, inferTimeReport.req_travelPath[0]);
            msvc_logFile << inferTimeReport.req_travelPath[0] << "|";
            for (unsigned int j = 0; j < inferTimeReport.req_origGenTime[0].size() - 1; j++) {
                msvc_logFile << timePointToEpochString(inferTimeReport.req_origGenTime[0].at(j)) << ",";
            }
            msvc_logFile << timePointToEpochString(inferTimeReport.req_origGenTime[0].back()) << "|";

            for (BatchSizeType j = 1; j < inferTimeReport.req_origGenTime[0].size(); j++) {
                msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(inferTimeReport.req_origGenTime[0].at(j) - inferTimeReport.req_origGenTime[0].at(j-1)).count() << ",";
            }
            msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(inferTimeReport.req_origGenTime[0].back() - inferTimeReport.req_origGenTime[0].front()).count() << std::endl;
        }
    }
    msvc_logFile.close();
    STOPPED = true;
}