#include <variant>
#include "communicator.h"
#include "receiver.h"
#include "sender.h"
#include "absl/flags/flag.h"
#include "spdlog/spdlog.h"

ABSL_FLAG(std::string, type, "", "type of communicator to start");
ABSL_FLAG(int, size, 0, "size of the quadratic image matrix");
ABSL_FLAG(int, overhead, 0, "factor that impacts the length of path and other header data");
ABSL_FLAG(std::string, url, "", "url of the receiver - only required for sender initialization");

void run_receiver(Receiver &receiver, bool gpu, int size) {
    ThreadSafeFixSizedDoubleQueue *queue = receiver.GetOutQueue()[0];
    std::ofstream file("./communicator_profiling" + std::to_string(size) + ".log");
    int i = 0;
    receiver.PAUSE_THREADS = false;
    while (i < 100) {
        if (gpu) {
            auto tmp = queue->pop2();
            if (strcmp(tmp.req_travelPath[0].c_str(), "empty") == 0) {
                continue;
            }
            spdlog::info("Receiving gpu image {}", i);
            tmp.req_origGenTime[0].push_back(std::chrono::high_resolution_clock::now());
            for (auto time : tmp.req_origGenTime[0]) {
                file << std::chrono::system_clock::to_time_t(time) << "|";
            }
            file << std::endl;
        } else {
            auto tmp = queue->pop1();
            if (strcmp(tmp.req_travelPath[0].c_str(), "empty") == 0) {
                continue;
            }
            spdlog::info("Receiving cpu image {}", i);
            tmp.req_origGenTime[0].push_back(std::chrono::high_resolution_clock::now());
            for (auto time : tmp.req_origGenTime[0]) {
                file << std::chrono::system_clock::to_time_t(time) << "|";
            }
            file << std::endl;
        }
        i++;
    }
    file.close();
}

int main(int argc, char *argv[]) {
    absl::ParseCommandLine(argc, argv);
    std::string type = absl::GetFlag(FLAGS_type);
    int size = absl::GetFlag(FLAGS_size);
    if (type == "gpu-receiver") {
        Receiver receiver(json::parse(
                "{\"msvc_name\": \"receiver\", \"msvc_contName\": \"receiver\", \"msvc_idealBatchSize\": 0, "
                "\"msvc_dnstreamMicroservices\": [{\"nb_classOfInterest\": -1,\"nb_commMethod\": 4, \"nb_link\": [\"\"], "
                "\"nb_maxQueueSize\": 10, \"nb_name\": \"::preprocessor\", \"nb_expectedShape\": [[-1, -1, -1]]}], "
                "\"msvc_dataShape\": [[0, 0]], \"msvc_svcLevelObjLatency\": 1, \"msvc_type\": 0, "
                "\"msvc_upstreamMicroservices\": [ {\"nb_classOfInterest\": -2, \"nb_commMethod\": 2, \"nb_link\": ["
                "\"0.0.0.0:50000\"],\"nb_maxQueueSize\": 10, \"nb_name\": \"datasource_0\", \"nb_expectedShape\": [[-1, -1, -1]]}], "
                "\"msvc_maxQueueSize\": 100, \"msvc_deviceIndex\": 0, \"msvc_containerLogPath\" : \"\", \"msvc_RUNMODE\" : 0}"));
        receiver.dispatchThread();
        run_receiver(receiver, true, size);
    } else if (type == "cpu-receiver") {
        Receiver receiver(json::parse(
                "{\"msvc_name\": \"receiver\", \"msvc_contName\": \"receiver\", \"msvc_idealBatchSize\": 0, "
                "\"msvc_dnstreamMicroservices\": [{\"nb_classOfInterest\": -1,\"nb_commMethod\": 4, \"nb_link\": [\"\"], "
                "\"nb_maxQueueSize\": 10, \"nb_name\": \"::preprocessor\", \"nb_expectedShape\": [[-1, -1, -1]]}], "
                "\"msvc_dataShape\": [[0, 0]], \"msvc_svcLevelObjLatency\": 1, \"msvc_type\": 0, "
                "\"msvc_upstreamMicroservices\": [ {\"nb_classOfInterest\": -2, \"nb_commMethod\": 2, \"nb_link\": ["
                "\"0.0.0.0:50000\"],\"nb_maxQueueSize\": 10, \"nb_name\": \"datasource_0\", \"nb_expectedShape\": [[-1, -1, -1]]}], "
                "\"msvc_maxQueueSize\": 100, \"msvc_deviceIndex\": 0, \"msvc_containerLogPath\" : \"\", \"msvc_RUNMODE\" : 0}"));
        receiver.dispatchThread();
        run_receiver(receiver, false, size);
    } else {
        auto *queue = new ThreadSafeFixSizedDoubleQueue(100, -1, "sender");
        bool gpu = false;
        Sender *sender;
        if (type == "gpu-sender") {
            gpu = true;
            sender = new GPUSender(json::parse(
                    "{\"msvc_name\": \"sender\", \"msvc_contName\": \"sender\", \"msvc_idealBatchSize\": 20, "
                    "\"msvc_dnstreamMicroservices\": [{\"nb_classOfInterest\": -1, \"nb_commMethod\": 3, \"nb_link\": [\"" +
                    absl::GetFlag(FLAGS_url) +
                    ":50000\"],\"nb_maxQueueSize\": 10, \"nb_name\": \"dummy_receiver_0\", \"nb_expectedShape\": [[-1, -1]]}], "
                    "\"msvc_dataShape\": [[0, 0]],\"msvc_svcLevelObjLatency\": 1,\"msvc_type\": 4003, "
                    "\"msvc_upstreamMicroservices\": [{\"nb_classOfInterest\": -2, \"nb_commMethod\": 3, \"nb_link\": [\"\"], "
                    "\"nb_maxQueueSize\": 10, \"nb_name\": \"::postprocessor\", \"nb_expectedShape\": [[-1, -1]]}], "
                    "\"msvc_maxQueueSize\": 100, \"msvc_deviceIndex\": 0, \"msvc_containerLogPath\" : \"\", \"msvc_RUNMODE\" : 0}"));
            sender->SetInQueue({queue});
            sender->dispatchThread();
        } else if (type == "local-cpu-sender") {
            sender = new LocalCPUSender(json::parse(
                    "{\"msvc_name\": \"sender\", \"msvc_contName\": \"sender\", \"msvc_idealBatchSize\": 20, "
                    "\"msvc_dnstreamMicroservices\": [{\"nb_classOfInterest\": -1, \"nb_commMethod\": 3, \"nb_link\": [\"" +
                    absl::GetFlag(FLAGS_url) +
                    ":50000\"],\"nb_maxQueueSize\": 10, \"nb_name\": \"dummy_receiver_0\", \"nb_expectedShape\": [[-1, -1]]}], "
                    "\"msvc_dataShape\": [[0, 0]],\"msvc_svcLevelObjLatency\": 1,\"msvc_type\": 4001, "
                    "\"msvc_upstreamMicroservices\": [{\"nb_classOfInterest\": -2, \"nb_commMethod\": 3, \"nb_link\": [\"\"], "
                    "\"nb_maxQueueSize\": 10, \"nb_name\": \"::postprocessor\", \"nb_expectedShape\": [[-1, -1]]}], "
                    "\"msvc_maxQueueSize\": 100, \"msvc_deviceIndex\": 0, \"msvc_containerLogPath\" : \"\", \"msvc_RUNMODE\" : 0}"));
            sender->SetInQueue({queue});
            sender->dispatchThread();
        } else if (type == "remote-cpu-sender") {
            sender = new RemoteCPUSender(json::parse(
                    "{\"msvc_name\": \"sender\", \"msvc_contName\": \"sender\", \"msvc_idealBatchSize\": 20, "
                    "\"msvc_dnstreamMicroservices\": [{\"nb_classOfInterest\": -1, \"nb_commMethod\": 3, \"nb_link\": [\"" +
                    absl::GetFlag(FLAGS_url) +
                    ":50000\"],\"nb_maxQueueSize\": 10, \"nb_name\": \"dummy_receiver_0\", \"nb_expectedShape\": [[-1, -1]]}], "
                    "\"msvc_dataShape\": [[0, 0]],\"msvc_svcLevelObjLatency\": 1,\"msvc_type\": 4002, "
                    "\"msvc_upstreamMicroservices\": [{\"nb_classOfInterest\": -2, \"nb_commMethod\": 3, \"nb_link\": [\"\"], "
                    "\"nb_maxQueueSize\": 10, \"nb_name\": \"::postprocessor\", \"nb_expectedShape\": [[-1, -1]]}], "
                    "\"msvc_maxQueueSize\": 100, \"msvc_deviceIndex\": 0, \"msvc_containerLogPath\" : \"\", \"msvc_RUNMODE\" : 0}"));
            sender->SetInQueue({queue});
            sender->dispatchThread();
            while (true) {
                std::cin >> type;
                if (type == "start") {
                    // Needs this to make sender start running
                    sender->PAUSE_THREADS = false;
                    break;
                }
            }
        } else {
            std::cerr << "Invalid type of communicator" << std::endl;
        }
        if (gpu) {
            cv::cuda::GpuMat img;
            img.create(size, size, CV_8UC3);
            img.setTo(cv::Scalar::all(0));
            Request<LocalGPUReqDataType> payload = {{{std::chrono::high_resolution_clock::now()}}, {0}, {""}, 0,
                                                    {{{3, size, size}, img}}};
            for (int i = 0; i < absl::GetFlag(FLAGS_overhead); i++) {
                payload.req_e2eSLOLatency.push_back(1337);
                payload.req_travelPath.push_back("task_name::mode_name::msvc_name");
            }
            for (int i = 0; i < 100; i++) {
                spdlog::info("Sending image {}", i);
                payload.req_origGenTime = {{std::chrono::high_resolution_clock::now()}};
                queue->emplace(payload);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        } else {
            cv::Mat img = cv::Mat::zeros(size, size, CV_8UC3);
            Request<LocalCPUReqDataType> payload = {{{std::chrono::high_resolution_clock::now()}}, {0}, {""}, 0,
                                                    {{{3, size, size}, img}}};
            for (int i = 0; i < absl::GetFlag(FLAGS_overhead); i++) {
                payload.req_e2eSLOLatency.push_back(1337);
                payload.req_travelPath.push_back("task_name::mode_name::msvc_name");
            }
            for (int i = 0; i < 100; i++) {
                spdlog::info("Sending image {}", i);
                payload.req_origGenTime = {{std::chrono::high_resolution_clock::now()}};
                queue->emplace(payload);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        while (true) {
            std::cin >> type;
            if (type == "exit") {
                break;
            }
        }
    }

    return 0;
}