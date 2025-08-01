#ifndef PIPEPLUSPLUS_PROFILER_H
#define PIPEPLUSPLUS_PROFILER_H

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <queue>
#include <unistd.h>
#include <nvml.h>
#include <spdlog/spdlog.h>

class LimitedPairQueue {
public:
    LimitedPairQueue(unsigned int limit = 10) : limit(limit) {}

    void push(std::pair<long, long> value) {
        if (q.size() == limit) q.pop();
        q.push(value);
    }

    std::pair<long, long> front() { return q.front(); }

    int size() { return q.size(); }

private:
    std::queue<std::pair<long, long>> q;
    unsigned int limit;
};

class Profiler {
public:
    Profiler(const std::vector<unsigned int> &pids, std::string mode);
    ~Profiler();

    void run() {};

    struct sysStats {
        int cpuUsage = 0;
        int processMemoryUsage = 0;
        int rssMemory = 0;
        int deviceMemoryUsage = 0;
        unsigned int gpuUtilization = 0;
        unsigned int gpuMemoryUsage = 0;
        int energyConsumption = 0;
    };


    void addPid(unsigned int pid) { setPidOnDevices(pid); };
    void removePid(unsigned int pid) { pidOnDevices.erase(pid); };
    int getGpuCount();
    std::vector<long> getGpuMemory(int device_count);

    sysStats reportAtRuntime(unsigned int cpu_pid, unsigned int gpu_pid);

    std::vector<Profiler::sysStats> reportDeviceStats();

private:

    bool initializeNVML();

    static bool setAccounting(nvmlDevice_t device);

    std::vector<nvmlDevice_t> getDevices();

    void setPidOnDevices(unsigned int pid, bool first = true);

    bool cleanupNVML();

    int getCPUInfo(unsigned int pid);

    int getDeviceCPUInfo();

    std::pair<int, int> getMemoryInfo(unsigned int pid);

    int getDeviceMemoryInfo();

    nvmlUtilization_t getGPUInfo(unsigned int pid, nvmlDevice_t device);

    int getEnergyConsumption(nvmlDevice_t device);

    unsigned int getPcieInfo(nvmlDevice_t device);

    bool nvmlInitialized;
    std::map<nvmlDevice_t ,bool> accountingEnabled;

    std::map<unsigned int, LimitedPairQueue> prevCpuTimes;
    std::vector<nvmlDevice_t> cuda_devices;
    std::map<unsigned int, nvmlDevice_t> pidOnDevices;
};


#endif //PIPEPLUSPLUS_PROFILER_H
