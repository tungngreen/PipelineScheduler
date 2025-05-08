#include "misc.h"
#include <fstream>
#include <torch/torch.h>
#include <boost/algorithm/string.hpp>
#include <random>
#include <cmath>
#include <chrono>
#include "indevicecommands.grpc.pb.h"
#include "indevicemessages.grpc.pb.h"
#include "controlcommands.grpc.pb.h"

#ifndef PIPEPLUSPLUS_MICRO_OPTIMIZATION_H
#define PIPEPLUSPLUS_MICRO_OPTIMIZATION_H

using controlcommands::ControlCommands;
using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using indevicemessages::InDeviceMessages;
using indevicecommands::FlData;
using EmptyMessage = google::protobuf::Empty;
using T = torch::Tensor;

enum threadingAction {
    NoMultiThreads = 0,
    MultiPreprocess = 1,
    MultiPostprocess = 2,
    BothMultiThreads = 3
};

const std::unordered_map<std::string, torch::Dtype> DTYPE_MAP = {
        {"float", torch::kFloat32},
        {"double", torch::kDouble},
        {"half", torch::kFloat16},
        {"int", torch::kInt32},
        {"long", torch::kInt64},
        {"short", torch::kInt16},
        {"char", torch::kInt8},
        {"byte", torch::kUInt8},
        {"bool", torch::kBool}
};

#endif //PIPEPLUSPLUS_MICRO_OPTIMIZATION_H
