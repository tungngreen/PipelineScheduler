#include "profiler.h"

Profiler::Profiler(const std::vector<unsigned int> &pids, std::string mode) {
    if (!initializeNVML()) {
        spdlog::error("Failed to initialize NVML");
        return;
    }
    pidOnDevices = std::map<unsigned int, nvmlDevice_t>();
    cuda_devices = getDevices();
    if (!pids.empty() && mode == "profile") {
        for (const auto &pid: pids) {
            setPidOnDevices(pid);
        }
    }
    nvmlInitialized = true;
}

Profiler::~Profiler() {
    if (nvmlInitialized) {
        if (!cleanupNVML()) {
            spdlog::get("container_agent")->error("Failed to shutdown NVML");
        }
    }
}

int Profiler::getGpuCount() {
    unsigned int device_count;
    nvmlReturn_t result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        spdlog::error("Failed to get device count: {}", nvmlErrorString(result));
        return -1;
    }
    return device_count;
}

std::vector<long> Profiler::getGpuMemory(int device_count) {
    std::vector<long> totalMemory;
    for (int i = 0; i < device_count; i++) {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            spdlog::error("Failed to get handle for device {}: {}", i, nvmlErrorString(result));
            return {-1};
        }
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (result != NVML_SUCCESS) {
            spdlog::get("container_agent")->error("Failed to get memory info for device {}: {}", i,
                                                  nvmlErrorString(result));
            return {-1};
        }
        totalMemory.push_back((long) memory.total / 1000000); // convert to MB
    }
    return totalMemory;
}

Profiler::sysStats Profiler::reportAtRuntime(unsigned int cpu_pid, unsigned int gpu_pid) {
    sysStats value{};
    value.cpuUsage = getCPUInfo(cpu_pid);
    auto mem = getMemoryInfo(cpu_pid);
    value.processMemoryUsage = mem.first;
    value.rssMemory = mem.second;
    nvmlUtilization_t gpu = getGPUInfo(gpu_pid, pidOnDevices[gpu_pid]);
    value.gpuUtilization = gpu.gpu;
    value.gpuMemoryUsage = gpu.memory;
    return value;
}

std::vector<Profiler::sysStats> Profiler::reportDeviceStats() {
    std::vector<Profiler::sysStats> deviceStats;
    for (unsigned int i = 0; i < cuda_devices.size(); i++) {
        sysStats value{};
        value.cpuUsage = getDeviceCPUInfo();
        value.processMemoryUsage = getDeviceMemoryInfo();
        nvmlUtilization_t gpu = getGPUInfo(0, cuda_devices[i]);
        value.gpuUtilization = gpu.gpu;
        value.gpuMemoryUsage = gpu.memory;
        deviceStats.push_back(value);
    }
    return deviceStats;
}

bool Profiler::initializeNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        spdlog::get("container_agent")->error("Failed to initialize NVML: {}", nvmlErrorString(result));
        return false;
    }
    return true;
}

bool Profiler::setAccounting(nvmlDevice_t device) {
    nvmlEnableState_t state;
    nvmlReturn_t result = nvmlDeviceGetAccountingMode(device, &state);
    if (result != NVML_SUCCESS) {
        spdlog::get("container_agent")->error("Failed to get accounting mode: {}", nvmlErrorString(result));
        return false;
    }
    if (state == NVML_FEATURE_DISABLED) {
        result = nvmlDeviceSetAccountingMode(device, NVML_FEATURE_ENABLED);
        if (result != NVML_SUCCESS) {
            spdlog::error("Failed to enable accounting mode: {}", nvmlErrorString(result));
            return false;
        }
    }
    return true;
}

std::vector<nvmlDevice_t> Profiler::getDevices() {
    std::vector<nvmlDevice_t> devices;
    unsigned int deviceCount;
    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);
    if (NVML_SUCCESS != result) {
        spdlog::get("container_agent")->error("Failed to query device count: {}", nvmlErrorString(result));
        return std::vector<nvmlDevice_t>();
    }
    for (unsigned int i = 0; i < deviceCount; i++) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result) {
            spdlog::get("container_agent")->error("Failed to get handle for device {}: {}", i, nvmlErrorString(result));
            return std::vector<nvmlDevice_t>();
        }
        setAccounting(device);
        devices.push_back(device);
    }
    return devices;
}

void Profiler::setPidOnDevices(unsigned int pid) {
    for (const auto &device: cuda_devices) {
        nvmlProcessInfo_t processes[64];
        unsigned int infoCount = 64;
        nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, processes);
        if (NVML_SUCCESS != result) {
            spdlog::get("container_agent")->error("Failed to get compute running processes for a device: {}",
                                                  nvmlErrorString(result));
            break;
        }
        for (unsigned int j = 0; j < infoCount; j++) {
            if (processes[j].pid == pid) {
                pidOnDevices[pid] = device;
                break;
            }
        }
    }
}

bool Profiler::cleanupNVML() {
    for (const auto &[pid, device]: pidOnDevices) {
        nvmlReturn_t result = nvmlDeviceClearAccountingPids(device);
        if (result != NVML_SUCCESS) {
            spdlog::get("container_agent")->error("Failed to clear accounting PIDs: {}", nvmlErrorString(result));
            return false;
        }
    }
    return true;
}

int Profiler::getCPUInfo(unsigned int pid) {
    std::vector<std::string> timers;
    std::string timer, line, skip, utime, stime;
    std::ifstream stream("/proc/stat");
    if (stream.is_open()) {
        std::getline(stream, line);
        std::istringstream linestream(line);
        linestream >> skip;
        for (int i = 0; i < 10; ++i) {
            linestream >> timer;
            timers.push_back(timer);
        }
    }
    stream.close();
    long total_active = 0;
    for (unsigned int i = 0; i < timers.size(); ++i) {
        if (i != 3 && i != 4) total_active += std::stol(timers[i]);
    }
    stream = std::ifstream("/proc/" + std::to_string(pid) + "/stat");
    if (stream.is_open()) {
        std::getline(stream, line);
        std::istringstream linestream(line);
        for (int i = 0; i < 13; ++i) linestream >> skip;
        linestream >> utime >> stime;
    }
    long process_active = 0;
    try {
        process_active = std::stol(utime) + std::stol(stime);
    } catch (const std::invalid_argument &ia) {
    }

    double cpuUsage = 0.0;
    if (prevCpuTimes[pid].size() > 0) {
        std::pair<long, long> prev_active = prevCpuTimes[pid].front();
        cpuUsage = 100.0 * (process_active - prev_active.first) / (total_active - prev_active.second);
    }

    prevCpuTimes[pid].push(std::make_pair(process_active, total_active));
    if (std::isinf(cpuUsage) || std::isnan(cpuUsage)) {
        return 0.0;
    }
    return (int) cpuUsage;
}

int Profiler::getDeviceCPUInfo() {
    std::string line;
    std::string cpu;
    std::ifstream stream("/proc/stat");
    if (stream.is_open()) {
        std::getline(stream, line);
        std::istringstream linestream(line);
        linestream >> cpu;
        long active = 0, total = 0;
        for (int i = 0; i < 8; ++i) {
            linestream >> cpu;
            if (i < 3 || i > 4) {
                active += std::stol(cpu);
            }
            total += std::stol(cpu);
        }
        double cpuUsage = 100.0 * (double) (active - prevCpuTimes[0].front().first) / (total - prevCpuTimes[0].front().second);
        prevCpuTimes[0].push(std::make_pair(active, total));
        if (std::isinf(cpuUsage) || std::isnan(cpuUsage)) {
            return 0.0;
        }
        return (int) cpuUsage;
    }
    return 0;
}

std::pair<int, int> Profiler::getMemoryInfo(unsigned int pid) {
    std::string mem = "0";
    std::string rss = "0";
    bool search = true;
    std::string line;
    std::string tmp;
    std::ifstream stream("/proc/" + std::to_string(pid) + "/status");
    if (stream.is_open()) {
        while (search && stream.peek() != EOF) {
            std::getline(stream, line);
            std::istringstream linestream(line);
            linestream >> tmp;
            if (tmp == "VmSize:") {
                linestream >> tmp;
                mem = tmp;
            }
            if (tmp == "VmRSS:") {
                linestream >> tmp;
                rss = tmp;
                search = false;
            }
        }
    }
    return std::pair((int) std::atoi(mem.c_str()) / 1000, (int) std::atoi(rss.c_str()) / 1000); //convert to MB
}

int Profiler::getDeviceMemoryInfo() {
    std::string line;
    std::string memTotal;
    std::string memAvailable;
    std::ifstream stream("/proc/meminfo");
    if (stream.is_open()) {
        std::getline(stream, line);
        std::istringstream linestream(line);
        linestream >> memTotal;
        linestream >> memTotal;
        std::getline(stream, line);
        std::getline(stream, line);
        linestream = std::istringstream(line);
        linestream >> memAvailable;
        linestream >> memAvailable;
    }
    return (std::stoi(memTotal) - std::stoi(memAvailable)) / 1000; //convert to MB
}

nvmlUtilization_t Profiler::getGPUInfo(unsigned int pid, nvmlDevice_t device) {
    nvmlUtilization_t util;
    nvmlReturn_t result;
    if (pid != 0) {
        nvmlAccountingStats_t stats;
        result = nvmlDeviceGetAccountingStats(device, pid, &stats);
        if (result == NVML_SUCCESS) {
            util.gpu = stats.gpuUtilization;
            util.memory = stats.memoryUtilization;
            return util;
        }
    }
    result = nvmlDeviceGetUtilizationRates(device, &util);
    if (result != NVML_SUCCESS) {
        util.gpu = 0;
        util.memory = 0;
    }
    nvmlMemory_t mem;
    result = nvmlDeviceGetMemoryInfo(device, &mem);
    if (result == NVML_SUCCESS) {
        util.memory = mem.used / 1000000;
    }
    return util;
}

unsigned int Profiler::getPcieInfo(nvmlDevice_t device) {
    unsigned int pcie;
    nvmlReturn_t result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_COUNT, &pcie);
    if (result != NVML_SUCCESS) {
        spdlog::get("container_agent")->error("Failed to get PCIe throughput: {}", nvmlErrorString(result));
        pcie = 0;
    }
    return pcie;
}
