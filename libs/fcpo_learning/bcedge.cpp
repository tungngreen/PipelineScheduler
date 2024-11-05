#include "bcedge.h"

BCEdgeAgent::BCEdgeAgent(std::string& dev_name, uint state_size, uint max_batch, uint scaling_size, uint memory_size,
                         CompletionQueue *cq, std::shared_ptr<InDeviceMessages::Stub> stub, torch::Dtype precision,
                         uint update_steps, double lambda, double gamma, double clip_epsilon)
                         : dev_name(dev_name), precision(precision), lambda(lambda), gamma(gamma),
                           clip_epsilon(clip_epsilon), state_size(state_size), max_batch(max_batch),
                           scaling_size(scaling_size), memory_size(memory_size), update_steps(update_steps) {
    path = "../models/bcedge/" + dev_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    model = std::make_shared<BCEdgeNet>(state_size, max_batch, scaling_size, memory_size);
    std::string model_save = path + "/latest_model.pt";
    if (std::filesystem::exists(model_save)) torch::load(model, model_save);
    model->to(precision);
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

    cumu_reward = 0.0;
    states = {};
    batching_actions = {};
    scaling_actions = {};
    memory_actions = {};
    rewards = {};
    log_probs = {};
    values = {};
}