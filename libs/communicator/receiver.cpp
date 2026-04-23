#include "receiver.h"

ReceiverConfigs Receiver::loadConfigsFromJson(const json &jsonConfigs) {
    ReceiverConfigs configs;
    return configs;
}

void Receiver::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOANDING configs...", __func__);

    if (!isConstructing) { // If this is not called from the constructor, then we are loading configs from a file for Microservice class
        Microservice::loadConfigs(jsonConfigs);
    }

    ReceiverConfigs configs = loadConfigsFromJson(jsonConfigs);

    if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
    } else if (msvc_RUNMODE == RUNMODE::DEPLOYMENT || msvc_RUNMODE == RUNMODE::PROFILING) {
        comm_ctx = context_t(1);
        socket = socket_t(comm_ctx, ZMQ_REP);
        socket.bind("tcp://" + upstreamMicroserviceList.front().link[0]);
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
    }
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);
}

Receiver::Receiver(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name); 
}

template<typename ReqDataType>
void Receiver::processInferTimeReport(Request<ReqDataType> &timeReport) {
    BatchSizeType batchSize = timeReport.req_batchSize;

    BatchSizeType numTimeStamps = (BatchSizeType) (timeReport.req_origGenTime.size() / batchSize);
    for (BatchSizeType i = 0; i < batchSize; i++) {
        msvc_logFile << timeReport.req_travelPath[i] << ",";
        for (BatchSizeType j = 0; j < numTimeStamps - 1; j++) {
            msvc_logFile << timePointToEpochString(timeReport.req_origGenTime[i * numTimeStamps + j]) << ",";
        }
        msvc_logFile << timePointToEpochString(timeReport.req_origGenTime[i * numTimeStamps + numTimeStamps - 1])
                     << "|";

        for (BatchSizeType j = 1; j < numTimeStamps - 1; j++) {
            msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(
                    timeReport.req_origGenTime[i * numTimeStamps + j] -
                    timeReport.req_origGenTime[i * numTimeStamps + j - 1]).count() << ",";
        }
        msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(
                timeReport.req_origGenTime[(i + 1) * numTimeStamps - 1] -
                timeReport.req_origGenTime[(i + 1) * numTimeStamps - 2]).count() << std::endl;
    }
}

void Receiver::GpuPointerRequestHandler(const std::string &msg) {
    ImageDataPayload request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("{} failed to unpack ImageDataPayload from msg: {}", __func__, msg);
        socket.send(zmq::buffer("ERROR", 5), zmq::send_flags::none);
        return;
    }
    spdlog::get("container_agent")->trace("{0:s} is processing request {1:s} ...", __func__, request.mutable_elements()->at(0).path());
    if (msvc_OutQueue[0]->getActiveQueueIndex() != 2) msvc_OutQueue[0]->setActiveQueueIndex(2);
    std::vector<Request<LocalGPUReqDataType>> requests = {};
    std::vector<RequestData<LocalGPUReqDataType>> elements = {};

    for (const auto &el: *request.mutable_elements()) {
        auto timestamps = std::vector<ClockType>();
        for (auto ts: el.timestamp()) {
            timestamps.emplace_back(TimePrecisionType(ts));
        }
        auto receivedTime = std::chrono::system_clock::now();
        updateStats(receivedTime);
        timestamps.push_back(receivedTime);
        void* data;
        cudaIpcMemHandle_t ipcHandle;
        memcpy(&ipcHandle, el.data().c_str(), sizeof(cudaIpcMemHandle_t));
        cudaError_t cudaStatus = cudaIpcOpenMemHandle(&data, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaIpcOpenMemHandle failed: " << cudaStatus << std::endl;
            continue;
        }
        auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3, data).clone();
        elements = {{{gpu_image.channels(), el.height(), el.width()}, gpu_image}};

        cudaIpcCloseMemHandle(data);

        if (elements.empty()) continue;

        Request<LocalGPUReqDataType> req = {
                {timestamps},
                {el.slo()},
                {el.path()},
                1,
                elements
        };
        msvc_OutQueue[0]->emplace(req);
        spdlog::get("container_agent")->trace("GpuPointerRequestHandler::{0:s} emplaced request with path: {1:s}", __func__, el.path());
    }
    socket.send(zmq::buffer("ACK", 3), zmq::send_flags::none);
}

void Receiver::SharedMemoryRequestHandler(const std::string &msg) {
    ImageDataPayload request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("{} failed to unpack ImageDataPayload from msg: {}", __func__, msg);
        socket.send(zmq::buffer("ERROR", 5), zmq::send_flags::none);
        return;
    }
    spdlog::get("container_agent")->trace("{0:s} is processing request {1:s} ...", __func__, request.mutable_elements()->at(0).path());
    if (msvc_OutQueue[0]->getActiveQueueIndex() != 1) msvc_OutQueue[0]->setActiveQueueIndex(1);
    std::vector<Request<LocalCPUReqDataType>> requests = {};
    std::vector<RequestData<LocalCPUReqDataType>> elements = {};

    for (const auto &el: *request.mutable_elements()) {
        auto timestamps = std::vector<ClockType>();
        for (auto ts: el.timestamp()) {
            timestamps.emplace_back(TimePrecisionType(ts));
        }
        auto receivedTime = std::chrono::system_clock::now();
        updateStats(receivedTime);
        timestamps.push_back(receivedTime);
        auto name = el.data().c_str();
        boost::interprocess::shared_memory_object shm{open_only, name, read_only};
        boost::interprocess::mapped_region region{shm, read_only};
        auto image = static_cast<cv::Mat *>(region.get_address());
        elements = {{{image->channels(), el.height(), el.width()}, *image}};

        boost::interprocess::shared_memory_object::remove(name);

        Request<LocalCPUReqDataType> req = {
                {timestamps},
                {el.slo()},
                {el.path()},
                1,
                elements
        };
        msvc_OutQueue[0]->emplace(req);
        spdlog::get("container_agent")->trace("SharedMemoryRequestHandler::{0:s} emplaced request with path: {1:s}", __func__, el.path());
    }
    socket.send(zmq::buffer("ACK", 3), zmq::send_flags::none);
}

void Receiver::SerializedDataRequestHandler(const std::string &msg) {
    ImageDataPayload request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("{} failed to unpack ImageDataPayload from msg: {}", __func__, msg);
        socket.send(zmq::buffer("ERROR", 5), zmq::send_flags::none);
        return;
    }
    spdlog::get("container_agent")->trace("{0:s} is processing request {1:s} ...", __func__, request.mutable_elements()->at(0).path());
    std::vector<Request<LocalCPUReqDataType>> requests = {};
    std::vector<RequestData<LocalCPUReqDataType>> elements = {};

    for (const auto &el: *request.mutable_elements()) {
        if (!validateReq(ClockType(TimePrecisionType(el.timestamp(0))), el.path())) {
            continue;
        }
        uint length = el.data().length();
        if (length != el.datalen()) {
            spdlog::get("container_agent")->error("SerializedDataRequestHandler::{0:s} data length does not match", __func__);
            continue;
        }
        auto timestamps = std::vector<ClockType>();
        for (auto ts: el.timestamp()) {
            timestamps.emplace_back(TimePrecisionType(ts));
        }
        auto receivedTime = std::chrono::system_clock::now();
        updateStats(receivedTime);
        timestamps.push_back(receivedTime);
        cv::Mat image;
        if (el.is_encoded()){
            std::vector<uchar> buf(el.data().c_str(), el.data().c_str() + length);
            image = cv::imdecode(buf, cv::IMREAD_COLOR);
        } else {
            image = cv::Mat(el.height(), el.width(), CV_8UC3,const_cast<char *>(el.data().c_str())).clone();
        }
        elements = {{{image.channels(), el.height(), el.width()}, image}};
        requests.emplace_back(Request<LocalCPUReqDataType>{
                {timestamps},
                {el.slo()},
                {el.path()},
                1,
                elements
        });

        spdlog::get("container_agent")->trace("SerializedDataRequestHandler::{0:s} unpacked request with path: {1:s}", __func__, el.path());

        /**
         * @brief Request now should carry 4 timestamps
         * 1. The very moment request is originally generated at the beggining of the pipeline. (FIRST_TIMESTAMP)
         * 2. The moment request is put into outqueue of last immediate upstream processor. (SECOND_TIMESTAMP)
         * 3. The moment request is sent by immediate upstream sender. (THIRD_TIMESTAMP)
         * 4. The moment request is received by the receiver. (FOURTH_TIMESTAMP)
         */
    }
    msvc_OutQueue[0]->emplace(requests);
    socket.send(zmq::buffer("ACK", 3), zmq::send_flags::none);
}

void Receiver::HandleRpcs() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    msg_handlers = {
            {std::to_string(pipelinescheduler::REMOTE), std::bind(&Receiver::SerializedDataRequestHandler, this, std::placeholders::_1)},
            {std::to_string(pipelinescheduler::MEMORY), std::bind(&Receiver::SharedMemoryRequestHandler, this, std::placeholders::_1)},
            {std::to_string(pipelinescheduler::GPU), std::bind(&Receiver::GpuPointerRequestHandler, this, std::placeholders::_1)},
    };
    READY = true;
    while (true) {
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
                setDevice();
                /*void* target;
                auto test = cv::cuda::GpuMat(1, 1, CV_8UC3);
                cudaIpcMemHandle_t ipcHandle;
                cudaIpcGetMemHandle(&ipcHandle, test.data);
                cudaError_t cudaStatus = cudaIpcOpenMemHandle(&target, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
                cudaIpcCloseMemHandle(target);
                test.release();
                if (cudaStatus != cudaSuccess) {
                    std::cout << "cudaIpcOpenMemHandle failed: " << cudaStatus << std::endl;
                    setDevice();
                }*/
                RELOADING = false;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        message_t message;
        if (socket.recv(message, recv_flags::none)) {
            std::string raw = message.to_string();
            std::istringstream iss(raw);
            std::string type;
            iss >> type;
            iss.get(); // skip the space after the type
            std::string payload((std::istreambuf_iterator<char>(iss)),
                                std::istreambuf_iterator<char>());
            if (msg_handlers.count(type)) {
                msg_handlers[type](payload);
            } else {
                spdlog::get("container_agent")->error("Received unknown Transfer type: {}", type);
            }
        }
    }
    msvc_logFile.close();
    STOPPED = true;
}