#ifndef PIPEPLUSPLUS_RECEIVER_H
#define PIPEPLUSPLUS_RECEIVER_H

#include "communicator.h"
#include <fstream>
#include <random>

using boost::interprocess::read_only;
using boost::interprocess::open_only;
using json = nlohmann::ordered_json;

struct ReceiverConfigs : BaseMicroserviceConfigs {
    uint8_t msvc_inputRandomizeScheme;
    std::string msvc_dataShape;
};

class Receiver : public Microservice {
public:
    Receiver(const json &jsonConfigs);

    ~Receiver() override {
        waitStop();
        spdlog::get("container_agent")->info("{0:s} has stopped", msvc_name);
    }

    template<typename ReqDataType>
    void processInferTimeReport(Request<ReqDataType> &timeReport);

    void dispatchThread() override {
        std::thread handler(&Receiver::HandleRpcs, this);
        handler.detach();
    }

    ReceiverConfigs loadConfigsFromJson(const json &jsonConfigs);

    void loadConfigs(const json &jsonConfigs, bool isConstructing = true) override;

    ClockType msvc_lastReqTime;
    std::atomic<int64_t> msvc_interReqTimeRunningMean = 0;
    std::atomic<int64_t> msvc_interReqTimeRunningVar = 0;

    virtual PerSecondArrivalRecord getPerSecondArrivalRecord() override {
        auto reqCount = msvc_totalReqCount.exchange(0);
        MsvcSLOType mean = msvc_interReqTimeRunningMean.exchange(0);
        MsvcSLOType var = msvc_interReqTimeRunningVar.exchange(0);
        return {reqCount, mean, var};
    }
    
private:
    /**
     * @brief update the statistics of the receiver including the inter-request time mean/std, total request count
     * All the statistics except overallTotalReqCount are reset after each second by ContainerAgent's calling
     * `getPerSecondArrivalRecord()` method, which clears out msvc_totalReqCount
     *
     * @param receiveTime
     */
    inline void updateStats(ClockType &receiveTime) {
        msvc_overallTotalReqCount++;
        uint32_t totalReqCount = ++msvc_totalReqCount;
        if (msvc_totalReqCount == 1) {
            msvc_interReqTimeRunningMean.store(0);
            msvc_interReqTimeRunningVar.store(0);
            msvc_lastReqTime = receiveTime;
            return;
        }
        int64_t interReqTime = std::chrono::duration_cast<std::chrono::microseconds>(receiveTime - msvc_lastReqTime).count();
        if (interReqTime < 0) {
            return;
        }
        int64_t mean = msvc_interReqTimeRunningMean.load();
        int64_t var = msvc_interReqTimeRunningVar.load();
        auto oldMean = mean;
        // std::cout << "totalReqCount: " << msvc_totalReqCount.load() << " mean: " << mean << std::endl;
        mean += ((interReqTime - oldMean) / totalReqCount);
        // std::cout << " var: " << var << std::endl;
        var += ((interReqTime - oldMean) * (interReqTime - mean));
        // var /= msvc_totalReqCount;
        msvc_interReqTimeRunningMean.exchange(mean);
        msvc_interReqTimeRunningVar.exchange(var);
        msvc_lastReqTime = receiveTime;
        // std::cout << "totalReqCount: " << msvc_totalReqCount.load() << " interReqTime: " << interReqTime << " mean: " << mean << " var: " << var << std::endl;
    }

    /**
     * @brief Check if this request is still valid or its too old and should be discarded
     *
     * @param timestamps
     * @return true
     * @return false
     */
    inline bool validateReq(ClockType originalGenTime, const std::string &path) {
        auto now = std::chrono::high_resolution_clock::now();
        uint64_t diff = std::chrono::duration_cast<TimePrecisionType>(now - originalGenTime).count();
        if (msvc_RUNMODE == RUNMODE::PROFILING) {
            if (checkProfileEnd(path)) {
                STOP_THREADS = true;
                return false;
            };
            return true;
        }
        if (diff > msvc_pipelineSLO - msvc_timeBudgetLeft &&
            msvc_DROP_MODE == DROP_MODE::LAZY) {
            msvc_droppedReqCount++;
            spdlog::get("container_agent")->trace("{0:s} drops a request with time {1:d}", containerName, diff);
            return false;
        } else if (msvc_DROP_MODE == DROP_MODE::NO_DROP) {
            return true;
        }
        return true;
    }

    void HandleRpcs();
    void SerializedDataRequestHandler(const std::string &msg);
    void SharedMemoryRequestHandler(const std::string &msg);
    void GpuPointerRequestHandler(const std::string &msg);

    context_t comm_ctx;
    socket_t socket;
    std::unordered_map<std::string, std::function<void(const std::string&)>> msg_handlers;

    std::string containerName;
};

#endif //PIPEPLUSPLUS_RECEIVER_H
