#pragma once

#include "controller.h"

/*
Jellyfish variable implementation.

Compared with the original jellyfish paper, we make some slight modification for our case:
(1) the req rate is the very first model of the pipeline(usually yolo in our case)
(2) we only map the datasource to different very first analysis model in the pipeline

The following code implement the scheduling algorithm of jellyfish which contains 1.data adaptation, 2.client-DNN mapping 3. dynamic batching
*/

/**
 * @brief ModelInfo is a collection of a single running model. A helper class for scheduling algorithm
 */
struct ModelInfoJF {
    int batch_size;
    float inference_latency;
    int throughput;
    int width;
    int height;
    std::string name;
    float accuracy;
    std::weak_ptr<PipelineModel> model;

    ModelInfoJF(int bs, float il, int w, int h, const std::string& n, float acc, std::weak_ptr<PipelineModel> m);

    bool operator==(const ModelInfoJF& other) const {
        auto this_model = model.lock();
        auto other_model = other.model.lock();
        return batch_size == other.batch_size && 
               inference_latency == other.inference_latency && 
               throughput == other.throughput && 
               width == other.width && 
               height == other.height && 
               name == other.name && 
               accuracy == other.accuracy &&
               this_model == other_model;
    }
};

/**
 * @brief comparison of the key of ModelProfiles, for sorting in the ModelProfiles::infos
 */
struct ModelSetCompare {
    bool operator()(const std::tuple<std::string, float> &lhs, const std::tuple<std::string, float> &rhs) const {
        // Strict weak ordering: If accuracies are the same, tie-break by name to prevent map collisions.
        if (std::get<1>(lhs) != std::get<1>(rhs)) {
            return std::get<1>(lhs) > std::get<1>(rhs); // Descending accuracy
        }
        return std::get<0>(lhs) > std::get<0>(rhs); // Tie-breaker
    }
};

/**
 * @brief ModelProfiles is a collection of all runing models' profile.
 */
class ModelProfilesJF {
public:
    void add(const std::string& name, float accuracy, int batch_size, float inference_latency, int width, int height, std::weak_ptr<PipelineModel> model);
    void add(const ModelInfoJF &model_info);
    
    // Thread-safe getter to replace direct public access
    std::map<std::tuple<std::string, float>, std::vector<ModelInfoJF>, ModelSetCompare> getInfos() const {
        std::lock_guard<std::mutex> lock(mtx);
        return infos;
    }
    
    void debugging();

private:
    std::map<std::tuple<std::string, float>, std::vector<ModelInfoJF>, ModelSetCompare> infos;
    mutable std::mutex mtx;
};

/**
 * @brief ClientInfo is a collection of single client's information. A helper class for scheduling algorithm
 */
struct ClientInfoJF
{
    std::string name;         // can be anything, just a unique identification for differenct clients(datasource)
    float budget;             // slo
    int req_rate;             // request rate (how many frame are sent to remote per second)
    std::weak_ptr<PipelineModel> model;     // pointer to that component
    int transmission_latency; // networking time, useful for scheduling
    std::string task_name;
    std::string task_source;
    NetworkEntryType network_entry;

    ClientInfoJF(const std::string& _name, float _budget, int _req_rate, std::weak_ptr<PipelineModel> _model,
                 const std::string& _task_name, const std::string& _task_source, NetworkEntryType _network_entry);

    bool operator==(const ClientInfoJF &other) const {
        return name == other.name && budget == other.budget && req_rate == other.req_rate;
    }

    void set_transmission_latency(int lat) {
        this->transmission_latency = lat;
    }
};

/**
 * @brief ClientProfiles is a collection of all clients' profile.
 */
class ClientProfilesJF {
public:
    static void sortBudgetDescending(std::vector<ClientInfoJF> &clients);
    
    void add(const std::string &name, float budget, int req_rate, std::weak_ptr<PipelineModel> model,
             const std::string& task_name, const std::string& task_source, NetworkEntryType network_entry);
             
    // Thread-safe getter
    std::vector<ClientInfoJF> getInfos() const {
        std::lock_guard<std::mutex> lock(mtx);
        return infos;
    }
             
    void debugging();

private:
    std::vector<ClientInfoJF> infos;
    mutable std::mutex mtx;
};

// the accuracy value here is dummy, just use for ranking models
inline const std::map<std::string, float> ACC_LEVEL_MAP = {
        {"yolov5n320", 0.30},
        {"yolov5n512", 0.40},
        {"yolov5n640", 0.50},
        {"yolov5s640", 0.55},
        {"yolov5m640", 0.60},
};

// --------------------------------------------------------------------------------------------------------
//                                     start of jellyfish scheduling implementation
// --------------------------------------------------------------------------------------------------------
namespace Jlf {
    // Note: Adjusted signature to take std::vector<ClientInfoJF>& directly so it can be mutated 
    // safely in the .cpp file without modifying the locked class internals.
    std::vector<std::tuple<std::tuple<std::string, float>, std::vector<ClientInfoJF>, int>>
    mapClient(std::vector<ClientInfoJF> &clients, ModelProfilesJF &model_profiles);

    std::vector<ClientInfoJF>
    findOptimalClients(const std::vector<ModelInfoJF> &models, std::vector<ClientInfoJF> &clients);

    int check_and_assign(std::vector<ModelInfoJF> &model, std::vector<ClientInfoJF> &selected_clients);

    std::tuple<int, int> findMaxBatchSize(const std::vector<ModelInfoJF> &models, const ClientInfoJF &client,
                                          int max_available_batch_size = 16);

    void differenceClients(std::vector<ClientInfoJF> &src, const std::vector<ClientInfoJF> &diff);
}

// --------------------------------------------------------------------------------------------------------
//                                      end of jellyfish scheduling implementation
// --------------------------------------------------------------------------------------------------------