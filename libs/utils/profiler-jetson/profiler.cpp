#include "profiler.h"

Profiler::Profiler(const std::vector<unsigned int> &pids, std::string mode) {
    if (!pids.empty()) {
        int pid = pids[0];
        std::string command = "python3 ../jetson_profiler.py " + mode + " " + std::to_string(pid);
        t = std::thread(&Profiler::jtop, this, command);
        t.detach();
    }
}

Profiler::~Profiler() {
    if (t.joinable()) {
        t.join();
    }
}

void Profiler::jtop(const std::string &cmd) {
    std::array<char, 128> buffer;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    std::vector<std::string> result;
    std::string token;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        std::stringstream ss(buffer.data());
        while (std::getline(ss, token, '|')) {
            result.push_back(token);
        }
        std::lock_guard<std::mutex> lock(m);
        stats[std::stoi(result[0])] = {std::stoi(result[1]), std::stoi(result[2]), std::stoi(result[5]), std::stoi(result[4]),
                                       std::stoi(result[3])};
        m.unlock();
        result = {};
    }
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
        double cpuUsage = 100.0 * (double) (active - prevCpuTimes.front().first) / (total - prevCpuTimes.front().second);
        prevCpuTimes.push(std::make_pair(active, total));
        if (std::isinf(cpuUsage) || std::isnan(cpuUsage)) {
            return 0.0;
        }
        return (int) cpuUsage;
    }
    return 0;
}