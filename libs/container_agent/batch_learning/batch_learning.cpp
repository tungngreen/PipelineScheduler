#include "batch_learning.h"

PPOAgent::PPOAgent(std::string& cont_name, uint state_size, uint max_batch, uint resolution_size, uint threading_size,
                   CompletionQueue *cq, std::shared_ptr<InDeviceMessages::Stub> stub, uint update_steps,
                   uint federated_steps, double lambda, double gamma, const std::string& model_save)
                   : cont_name(cont_name), cq(cq), stub(stub), lambda(lambda), gamma(gamma), max_batch(max_batch),
                   update_steps(update_steps), federated_steps(federated_steps) {
    path = "../models/batch_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    std::random_device rd;
    re = std::mt19937(rd());
    model = std::make_shared<MultiPolicyNetwork>(state_size, resolution_size, max_batch, threading_size);
    model->to(torch::kF64);
    if (!model_save.empty()) torch::load(model, model_save);
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

    avg_reward = 0.0;
    penalty_weight = 0.1;
    states = {};
    resolution_actions = {};
    batching_actions = {};
    scaling_actions = {};
    rewards = {};
    log_probs = {};
    values = {};
}

void PPOAgent::update() {
    steps_counter = 0;
    if (federated_steps_counter == 0) {
        spdlog::trace("Waiting for federated update, cancel !");
        return;
    }
    if (federated_steps_counter++ % federated_steps == 0) {
        std::cout << "Federated Training: " << avg_reward << std::endl;
        spdlog::info("Federated training RL agent at average Reward {}!", avg_reward);
        federated_steps_counter = 0;  // 0 means that we are waiting for federated update to come back
        federatedUpdate();
        return;
    }
    spdlog::info("Locally training RL agent at average Reward {}!", avg_reward);
    Stopwatch sw;
    sw.start();

    auto [policy1, policy2, policy3, v] = model->forward(torch::stack(states));
    T action1_probs = torch::softmax(policy1, -1);
    T action1_log_probs = torch::log(action1_probs.gather(-1, torch::tensor(resolution_actions).view({-1, 1, 1})).squeeze(-1));
    T action2_probs = torch::softmax(policy2, -1);
    T action2_log_probs = torch::log(action2_probs.gather(-1, torch::tensor(batching_actions).view({-1, 1, 1})).squeeze(-1));
    T action3_probs = torch::softmax(policy3, -1);
    T action3_log_probs = torch::log(action3_probs.gather(-1, torch::tensor(batching_actions).view({-1, 1, 1})).squeeze(-1));
    T new_log_probs = action1_log_probs + action2_log_probs + action3_log_probs;

    T ratio = torch::exp(new_log_probs - torch::stack(log_probs));
    T clipped_ratio = torch::clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon);
    T advantages = computeGae();

    T policy_loss = -torch::min(ratio * advantages, clipped_ratio * advantages).to(torch::kF64).mean();
    T value_loss = torch::mse_loss(v, computeCumuRewards()).to(torch::kF64);
    T policy1_penalty = penalty_weight * torch::mean(torch::clamp(torch::tensor(resolution_actions), 0, 1)).to(torch::kF64);
    T policy3_penalty = penalty_weight * torch::mean(torch::clamp(torch::tensor(scaling_actions), 0, 1)).to(torch::kF64);
    T loss = policy_loss + 0.5 * value_loss + policy1_penalty + policy3_penalty;

    // Backpropagation
    optimizer->zero_grad();
    loss.to(torch::kF64).backward();
    std::lock_guard<std::mutex> lock(model_mutex);
    optimizer->step();
    sw.stop();

    std::cout << "Training: " << sw.elapsed_microseconds() << std::endl;

    reset();
}

void PPOAgent::federatedUpdate() {
    FlData request;
    // save model information in uchar* pointer and then reload it from that information
    std::ostringstream oss;
    torch::save(model, oss);
    request.set_name(cont_name);
    request.set_network(oss.str());
    request.set_states(torch::stack(states).to(torch::kF64).data_ptr<uchar>(), states.size() * states[0].numel() * sizeof(double));
    request.set_values(torch::stack(values).to(torch::kF64).data_ptr<uchar>(), values.size() * values[0].numel() * sizeof(double));
    request.set_resolution_actions(torch::tensor(resolution_actions).to(torch::kF64).data_ptr<uchar>(), resolution_actions.size() * sizeof(int));
    request.set_batching_actions(torch::tensor(batching_actions).to(torch::kF64).data_ptr<uchar>(), batching_actions.size() * sizeof(int));
    request.set_scaling_actions(torch::tensor(scaling_actions).to(torch::kF64).data_ptr<uchar>(), scaling_actions.size() * sizeof(int));
    request.set_rewards(computeCumuRewards().to(torch::kF64).data_ptr<uchar>(), rewards.size() * sizeof(double));
    request.set_log_probs(torch::stack(log_probs).to(torch::kF64).data_ptr<uchar>(), log_probs.size() * log_probs[0].numel() * sizeof(double));
    reset();
    EmptyMessage reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stub->AsyncStartFederatedLearning(&context, request, cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (cq != nullptr) GPR_ASSERT(cq->Next(&got_tag, &ok));
    if (!status.ok()){
        spdlog::error("Federated update failed: {}", status.error_message());
        federated_steps_counter = 1; // 1 means that we are starting local updates again until the next federation
    }
}

void PPOAgent::federatedUpdateCallback(FlData &response) {
    std::istringstream iss(response.network());
    std::lock_guard<std::mutex> lock(model_mutex);
    torch::load(model, iss);
    steps_counter = 0;
    federated_steps_counter = 1; // 1 means that we are starting local updates again until the next federation
    reset();
}

void PPOAgent::rewardCallback(double throughput, double drops, double latency_penalty, double oversize_penalty) {
    states.push_back(state);
    rewards.push_back(2 * throughput - drops - latency_penalty + (1 - oversize_penalty));
}

void PPOAgent::setState(double curr_batch, double curr_resolution_choice, double arrival, double pre_queue_size, double inf_queue_size) {
    state = torch::tensor({{curr_batch / max_batch, curr_resolution_choice, arrival, pre_queue_size, inf_queue_size}}, torch::kF64);
}

std::tuple<int, int, int> PPOAgent::selectAction() {
    std::lock_guard<std::mutex> lock(model_mutex);
    auto [policy1, policy2, policy3, value] = model->forward(state);

    T action_dist = torch::multinomial(policy1, 1);  // Sample from policy (discrete distribution)
    int resolution = action_dist.item<int>();  // Convert tensor to int action
    action_dist = torch::multinomial(policy2, 1);
    int batching = action_dist.item<int>();
    action_dist = torch::multinomial(policy3, 1);
    int scaling = action_dist.item<int>();

    log_probs.push_back(torch::log(policy1.squeeze(0)[resolution] + torch::log(policy2.squeeze(0)[batching]) + torch::log(policy3.squeeze(0)[scaling])));
    values.push_back(value);
    return std::make_tuple(resolution, batching, scaling);
}

T PPOAgent::computeCumuRewards(double last_value) const {
    std::vector<double> discounted_rewards;
    double cumulative = last_value;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        cumulative = *it + gamma * cumulative;
        discounted_rewards.insert(discounted_rewards.begin(), cumulative);
    }
    return torch::tensor(discounted_rewards).to(torch::kF64);
}

T PPOAgent::computeGae(double last_value) const {
    std::vector<double> advantages;
    double gae = 0.0;
    double next_value = last_value;
    for (int t = rewards.size() - 1; t >= 0; --t) {
        double delta = rewards[t] + gamma * next_value - values[t].item<double>();
        gae = delta + gamma * lambda * gae;
        advantages.insert(advantages.begin(), gae);
        next_value = values[t].item<double>();
    }
    return torch::tensor(advantages).to(torch::kF64);
}

std::tuple<int, int, int> PPOAgent::runStep() {
    Stopwatch sw;
    sw.start();
    std::tuple<int, int, int> action = selectAction();
    avg_reward += rewards[steps_counter] / update_steps;

    steps_counter++;
    resolution_actions.push_back(std::get<0>(action));
    batching_actions.push_back(std::get<1>(action));
    scaling_actions.push_back(std::get<2>(action));
    sw.stop();
    out << sw.elapsed_microseconds() << "," << federated_steps_counter << "," << steps_counter << "," << avg_reward << "," << std::get<0>(action) << "," << std::get<1>(action) << "," << std::get<2>(action) << std::endl;

    if (steps_counter%update_steps == 0) {
        std::thread t(&PPOAgent::update, this);
        t.detach();
    }
    return std::make_tuple(std::get<0>(action) + 1, std::get<1>(action) + 1, std::get<2>(action));
}