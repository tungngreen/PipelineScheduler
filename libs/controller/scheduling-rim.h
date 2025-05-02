#include "controller.h"

// ====================================================== IMPLEMENTATION OF RIM  ===========================================================

// Helper functions
namespace Rim {
    const BatchSizeType RIM_BATCH_SIZE = 8;

    std::string device_type_str(NodeHandle *device) {
        std::string device_type;
        switch (device->type) {
            case OrinAGX:
                device_type = "orinagx";
                break;
            case OrinNX:
                device_type = "orinnx";
                break;
            case OrinNano:
                device_type = "orinano";
                break;
            case AGXXavier:
                device_type = "agxavier";
                break;
            case NXXavier:
                device_type = "nxavier";
                break;
            default:
                device_type = "server";
                break;
        }
        return device_type;
    }

    uint64_t calc_model_fps(PipelineModel *currModel, NodeHandle *device) {
        std::lock_guard<std::mutex> model_lock(currModel->pipelineModelMutex);

        uint64_t batchSize = RIM_BATCH_SIZE;

        currModel->batchSize = batchSize;

        std::string device_type = device_type_str(device);

        ModelProfile profile = currModel->processProfiles[device_type];
        uint64_t preprocessLatency = profile.batchInfer[batchSize].p95prepLat;
        uint64_t inferLatency = profile.batchInfer[batchSize].p95inferLat;
        uint64_t postprocessLatency = profile.batchInfer[batchSize].p95postLat;

        currModel->expectedAvgPerQueryLatency = preprocessLatency + inferLatency * batchSize + postprocessLatency;
        return static_cast<uint64_t>(std::round(
                1000000.0 / static_cast<double>(currModel->expectedAvgPerQueryLatency)));
    }

    double calc_pipe_utilization(const std::list<std::string> &subgraph,
                                 const std::unordered_map<std::string, PipelineModel *> &model_map,
                                 NodeHandle *device, uint64_t desiredFps) {
        double device_utilization = 0.0;
        for (const auto &node_name: subgraph) {
            auto it = model_map.find(node_name);
            if (it != model_map.end()) {
                PipelineModel *node = it->second;
                if (node_name == "datasource" || node_name == "sink") {
                    continue;
                }
                uint64_t max_model_fps = calc_model_fps(node, device);
                device_utilization += static_cast<double>(desiredFps) / max_model_fps;
            }
        }
        return device_utilization;
    }

    void generate_subgraphs_helper(PipelineModel *node, std::vector<PipelineModel *> &current_path,
                                   std::vector<std::list<std::string>> &subgraphs) {
        std::unique_lock<std::mutex> lock(node->pipelineModelMutex);

        if (!node || current_path.size() >= 3) return;

        if (node->name.find("sink") == std::string::npos) {
            current_path.push_back(node);

            if (current_path.size() <= 3) {
                std::list<std::string> subgraph;
                for (auto *n: current_path) {
                    subgraph.push_back(n->name);
                }
                subgraphs.push_back(subgraph);
            }

            // Continue with downstream nodes
            auto downstreams = node->downstreams;
            lock.unlock();

            for (const auto &downstream: downstreams) {
                generate_subgraphs_helper(downstream.first, current_path, subgraphs);
            }

            current_path.pop_back();
        }
    }

    void generate_split_subgraphs(PipelineModel *node, std::vector<std::list<std::string>> &subgraphs) {
        std::unique_lock<std::mutex> lock(node->pipelineModelMutex);

        if (!node || node->downstreams.size() < 2) return;

        for (size_t i = 0; i < node->downstreams.size(); ++i) {
            for (size_t j = i + 1; j < node->downstreams.size(); ++j) {
                std::list<std::string> subgraph;
                subgraph.push_back(node->name);
                subgraph.push_back(node->downstreams[i].first->name);
                subgraph.push_back(node->downstreams[j].first->name);
                subgraphs.push_back(subgraph);
            }
        }

        lock.unlock();
    }

    std::vector<std::list<std::string>> generate_subgraphs(PipelineModel *root) {
        std::unique_lock<std::mutex> lock(root->pipelineModelMutex);
        std::vector<std::list<std::string>> subgraphs;
        std::vector<PipelineModel *> current_path;

        auto downstreams = root->downstreams;
        lock.unlock();

        for (const auto &downstream: downstreams) {
            generate_subgraphs_helper(downstream.first, current_path, subgraphs);
            generate_split_subgraphs(downstream.first, subgraphs);
        }

        return subgraphs;
    }

    std::unordered_map<std::string, PipelineModel *> build_model_map(PipelineModel *root) {
        std::unordered_map<std::string, PipelineModel *> model_map;
        if (!root) return model_map;

        std::queue<PipelineModel *> q;
        q.push(root);

        while (!q.empty()) {
            PipelineModel *current = q.front();
            q.pop();

            std::unique_lock<std::mutex> lock(current->pipelineModelMutex);

            if (model_map.find(current->name) == model_map.end()) {
                model_map[current->name] = current;

                for (const auto &downstream: current->downstreams) {
                    q.push(downstream.first);
                }
            }
            lock.unlock();
        }
        return model_map;
    }

    // End of helper functions

    std::list<std::string> *choosing_subgraph_for_edge(const std::vector<std::list<std::string>> &available_subgraphs,
                                                       PipelineModel *root,
                                                       std::map<std::string, NodeHandle *> devices,
                                                       uint64_t desiredFps) {
        double best_score = 0.0;
        auto model_map = build_model_map(root);
        std::list<std::string> best_subgraph;
        NodeHandle *best_device = nullptr;

        std::unique_lock<std::mutex> root_lock(root->pipelineModelMutex);
        auto possible_devices = root->possibleDevices;
        root_lock.unlock();

        for (std::map<std::string, NodeHandle *>::iterator it = devices.begin(); it != devices.end(); ++it) {
            if (std::find(possible_devices.begin(), possible_devices.end(), it->first) != possible_devices.end()) {
                for (const auto &subgraph: available_subgraphs) {
                    double score = calc_pipe_utilization(subgraph, model_map, it->second, desiredFps);
                    if (score > best_score && score <= 1.0) {
                        best_score = score;
                        best_subgraph = subgraph;
                        std::unique_lock<std::mutex> device_lock(it->second->nodeHandleMutex);
                        best_device = it->second;
                        device_lock.unlock();
                    }
                }
            }
        }

        if (best_score == 0.0) {
            return nullptr;
        }

        std::unique_lock<std::mutex> device_lock(best_device->nodeHandleMutex);

        // Update the device attribute for nodes in the best subgraph
        for (const auto &node_name: best_subgraph) {
            auto it = model_map.find(node_name);
            if (it != model_map.end()) {
                PipelineModel *node = it->second;
                std::unique_lock<std::mutex> node_lock(node->pipelineModelMutex);
                node->deviceAgent = best_device;
                node->device = best_device->name;
                node->deviceTypeName = device_type_str(best_device);
                node_lock.unlock();
            }
        }
        device_lock.unlock();

        return new std::list<std::string>(best_subgraph);
    }

    void collect_all_nodes(PipelineModel *node, std::set<std::string> &all_nodes) {
        if (!node || all_nodes.find(node->name) != all_nodes.end()) return;

        all_nodes.insert(node->name);

        for (const auto &downstream: node->downstreams) {
            collect_all_nodes(downstream.first, all_nodes);
        }
    }

    std::set<std::string> get_remaining_nodes(PipelineModel *root, const std::list<std::string> *chosen_subgraph) {
        std::set<std::string> all_nodes;
        collect_all_nodes(root, all_nodes);

        if (chosen_subgraph) {
            // Remove nodes that are in the chosen subgraph
            for (const auto &node: *chosen_subgraph) {
                all_nodes.erase(node);
            }
        }

        // Remove "datasource" and any nodes containing "sink"
        all_nodes.erase("datasource");
        for (auto it = all_nodes.begin(); it != all_nodes.end();) {
            if (it->find("sink") != std::string::npos) {
                it = all_nodes.erase(it);
            } else {
                ++it;
            }
        }

        return all_nodes;
    }

    void place_on_server(PipelineModel *root, std::set<std::string> &remaining_nodes,
                         std::map<std::string, NodeHandle *> &devices) {
        auto model_map = build_model_map(root);
        NodeHandle *server_node = devices.at("server");

        for (const auto &node_name: remaining_nodes) {
            auto it = model_map.find(node_name);
            if (it != model_map.end()) {
                PipelineModel *node = it->second;
                std::unique_lock<std::mutex> node_lock(node->pipelineModelMutex);
                node->deviceAgent = server_node;
                node->device = server_node->name;
                node->deviceTypeName = device_type_str(server_node);
            }
        }
    }

    void print_subgraphs(const std::vector<std::list<std::string>> &subgraphs) {
        for (size_t i = 0; i < subgraphs.size(); ++i) {
            std::cout << i + 1 << ". ";
            bool first = true;
            for (const auto &node: subgraphs[i]) {
                if (!first)
                    std::cout << "-";
                std::cout << node;
                first = false;
            }
            std::cout << std::endl;
        }
    }

    void print_remaining_nodes(const std::set<std::string> &nodes) {
        for (const auto &node: nodes) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }

    void print_pipeline_levels(PipelineModel *root) {
        std::cout << "Pipeline Graph Levels:" << std::endl;

        std::queue<PipelineModel *> current_level;
        current_level.push(root);

        int level = 0;
        while (!current_level.empty()) {
            std::cout << "Level " << level << ": ";

            std::queue<PipelineModel *> next_level;
            std::unordered_set<PipelineModel *> level_set;

            while (!current_level.empty()) {
                PipelineModel *node = current_level.front();
                current_level.pop();

                if (level_set.find(node) == level_set.end()) {
                    std::cout << node->name << " ";
                    level_set.insert(node);

                    for (const auto &[downstream, _]: node->downstreams) {
                        if (level_set.find(downstream) == level_set.end()) {
                            next_level.push(downstream);
                        }
                    }
                }
            }

            std::cout << std::endl;
            current_level = next_level;
            level++;
        }
    }

    void rim_action(TaskHandle *task, Devices &devices) {
        // The desired fps
        std::lock_guard<std::mutex> task_lock(task->tk_mutex);

        uint64_t desiredFps = static_cast<uint64_t>(std::round(1000000.0 / static_cast<double>(task->tk_slo)));
        PipelineModel *root = task->tk_pipelineModels[0];
        std::vector<std::list<std::string>> suitable_subgraphs = generate_subgraphs(root);

        print_subgraphs(suitable_subgraphs);

        std::map<std::string, NodeHandle *> edges;
        std::map<std::string, NodeHandle *> servers;


        for (const auto &pair: devices.getMap()) {
            if (pair.second->name == "server") {
                servers[pair.first] = pair.second;
            } else {
                edges[pair.first] = pair.second;
            }
        }



        // update device for datasource
        // std::lock_guard<std::mutex> source_lock(root->pipelineModelMutex);
        std::unique_lock<std::mutex> source_lock(root->pipelineModelMutex);
        root->deviceAgent = edges.at(root->device);
        root->deviceTypeName = device_type_str(root->deviceAgent);
        source_lock.unlock();

        std::list<std::string> *chosen_subgraph = choosing_subgraph_for_edge(suitable_subgraphs, root, edges,
                                                                             desiredFps);
        std::set<std::string> remaining_nodes = get_remaining_nodes(root, chosen_subgraph);

        print_remaining_nodes(remaining_nodes);

        place_on_server(root, remaining_nodes, servers);

        // update device for sink
        PipelineModel *sink = task->tk_pipelineModels[task->tk_pipelineModels.size() - 1];
        std::lock_guard<std::mutex> sink_lock(sink->pipelineModelMutex);
        sink->device = sink->possibleDevices[0];
        sink->deviceAgent = devices.getDevice(sink->device);
        sink->deviceTypeName = device_type_str(sink->deviceAgent);
    }
}

// ====================================================== END OF IMPLEMENTATION OF RIM  ===========================================================
