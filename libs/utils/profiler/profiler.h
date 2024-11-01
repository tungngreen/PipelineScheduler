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
#include <queue>
#include <unistd.h>
#include <Python.h>
#include <mutex>
#include <condition_variable>
#include <termios.h>
#include <sys/select.h>

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

    void run();

    struct sysStats {
        int cpuUsage = 0;
        int processMemoryUsage = 0;
        int deviceMemoryUsage = 0;
        unsigned int gpuUtilization = 0;
        unsigned int gpuMemoryUsage = 0;
    };

    static int getGpuCount() { return 1; };
    void addPid(unsigned int pid) { stats[pid] = sysStats(); };
    static std::vector<unsigned int> getGpuMemory(int device_count) { return {0}; };

    sysStats reportAtRuntime(unsigned int pid) {
        std::lock_guard<std::mutex> lock(m); return stats[pid]; };
    sysStats reportAnyMetrics() {
        std::lock_guard<std::mutex> lock(m); return stats.begin()->second; };

    int getDeviceCPUInfo();

private:
    void jtop(const std::string &cmd);
    LimitedPairQueue prevCpuTimes;
    std::thread t;
    std::mutex m;
    std::map<unsigned int, sysStats> stats;
};


#endif //PIPEPLUSPLUS_PROFILER_H
