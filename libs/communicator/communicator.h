#ifndef PIPEPLUSPLUS_COMMUNICATOR_H
#define PIPEPLUSPLUS_COMMUNICATOR_H

#include <cstdint>
#include <utility>
#include <string>
#include <random>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include <google/protobuf/empty.pb.h>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

#include "microservice.h"
#include "dataexchange.grpc.pb.h"

using pipelinescheduler::ImageDataPayload;
using pipelinescheduler::ImageData;
using EmptyMessage = google::protobuf::Empty;

#endif //PIPEPLUSPLUS_COMMUNICATOR_H