#include "baseprocessor.h"

using namespace spdlog;

void BaseSink::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::trace("{0:s} is LOANDING configs...", __func__);
    if (!isConstructing) {
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }
    spdlog::trace("{0:s} FINISHED loading configs...", __func__);
}

BaseSink::BaseSink(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_name = "sink";
    info("{0:s} is created.", __func__);
}

void BaseSink::sink() {
    Request<LocalCPUReqDataType> inferTimeReport;
    BatchSizeType batchSize;
    int keepProfiling = 0;

    while (true) {
        if (this->STOP_THREADS) {
            if (this->STOP_THREADS) {
                info("{0:s} STOPS.", msvc_name);
                break;
            }
        } else if (this->PAUSE_THREADS) {
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
                keepProfiling = 1;
                RELOADING = false;
                READY = true;
                info("{0:s} is reloaded.", msvc_name);
            }
            continue;
        }
        inferTimeReport = msvc_InQueue.at(0)->pop1();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(inferTimeReport.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
        }

        if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
        }

        /**
         * @brief During profiling mode, there are six important timestamps to be recorded:
         * 1. When the request was generated
         * 2. When the request was received by the batcher
         * 3. When the request was done preprocessing by the batcher
         * 4. When the request, along with all others in the batch, was batched together and sent to the inferencer
         * 5. When the batch inferencer was completed by the inferencer 
         * 6. When each request was completed by the postprocessor
         */
        if (msvc_RUNMODE == RUNMODE::PROFILING) {
            batchSize = inferTimeReport.req_batchSize;
            if (inferTimeReport.req_travelPath[batchSize - 1].find("BATCH_ENDS") != std::string::npos) {
                inferTimeReport.req_travelPath[batchSize - 1] = removeSubstring(inferTimeReport.req_travelPath[batchSize - 1], "BATCH_ENDS");
                keepProfiling = 0;
            }
            BatchSizeType numTimeStamps = (BatchSizeType)(inferTimeReport.req_origGenTime.size() / batchSize);
            for (BatchSizeType i = 0; i < batchSize; i++) {
                msvc_logFile << inferTimeReport.req_travelPath[i] << ",";
                for (auto j = 0; j < inferTimeReport.req_origGenTime[i].size() - 1; j++) {
                    msvc_logFile << timePointToEpochString(inferTimeReport.req_origGenTime[i].at(j)) << ",";
                }
                msvc_logFile << timePointToEpochString(inferTimeReport.req_origGenTime[i].back()) << "|";

                for (BatchSizeType j = 1; j < inferTimeReport.req_origGenTime[i].size(); j++) {
                    msvc_logFile << std::chrono::duration_cast<std::chrono::nanoseconds>(inferTimeReport.req_origGenTime[i].at(j) - inferTimeReport.req_origGenTime[i].at(j-1)).count() << ",";
                }
                msvc_logFile << std::chrono::duration_cast<std::chrono::nanoseconds>(inferTimeReport.req_origGenTime[i].back() - inferTimeReport.req_origGenTime[i].front()).count() << std::endl;
            }

            // it transfers a dummy request back to the data generator to keep the profiling mode running
            msvc_OutQueue.at(0)->emplace(
                Request<LocalCPUReqDataType>(
                    inferTimeReport.req_origGenTime,
                    inferTimeReport.req_e2eSLOLatency,
                    inferTimeReport.req_travelPath,
                    inferTimeReport.req_batchSize,
                    {
                        {
                            {{1}},
                            {cv::Mat(1, 1, CV_8U, cv::Scalar(keepProfiling))}
                        }
                    }
                )
            );
        /**
         * @brief 
         * 
         */
        } else if (msvc_RUNMODE == RUNMODE::DEPLOYMENT) {
            if (inferTimeReport.req_travelPath[0].find("PROFILE_ENDS") != std::string::npos) {
                this->pauseThread();
            }
        }
    }
    msvc_logFile.close();
}