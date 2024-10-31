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
    Profiler(const std::vector<unsigned int> &pids);

    ~Profiler();

    void addPid(unsigned int pid);

    void removePid(unsigned int pid);

    struct sysStats {
        uint64_t timestamp = 0;
        int cpuUsage = 0;
        int memoryUsage = 0;
        int rssMemory = 0;
        unsigned int gpuUtilization = 0;
        unsigned int gpuMemoryUsage = 0;
    };

    int getGpuCount();

    std::vector<long> getGpuMemory(int device_count);

    sysStats reportAtRuntime(unsigned int cpu_pid, unsigned int gpu_pid);

    std::vector<Profiler::sysStats> reportDeviceStats();

private:

    bool initializeNVML();

    static bool setAccounting(nvmlDevice_t device);

    static std::vector<nvmlDevice_t> getDevices();

    void setPidOnDevices(unsigned int pid);

    bool cleanupNVML();

    int getCPUInfo(unsigned int pid);

    int getDeviceCPUInfo();

    std::pair<int, int> getMemoryInfo(unsigned int pid);

    int getDeviceMemoryInfo();

    nvmlUtilization_t getGPUInfo(unsigned int pid, nvmlDevice_t device);

    unsigned int getPcieInfo(nvmlDevice_t device);

    bool nvmlInitialized;

    std::map<unsigned int, LimitedPairQueue> prevCpuTimes;
    std::vector<nvmlDevice_t> cuda_devices;
    std::map<unsigned int, nvmlDevice_t> pidOnDevices;
};


#endif //PIPEPLUSPLUS_PROFILER_H
