#include "fcpo_learning.h"

FCPOAgent::FCPOAgent(std::string& cont_name, uint state_size, uint timeout_size, uint max_batch,  uint scaling_size,
                     CompletionQueue *cq, std::shared_ptr<InDeviceMessages::Stub> stub,
                     BatchInferProfileListType profile, int base_batch, torch::Dtype precision, uint update_steps,
                     uint update_steps_inc, uint federated_steps, double lambda, double gamma, double clip_epsilon,
                     double penalty_weight, double theta, double sigma, double phi, double rho, int seed)
        : precision(precision), batchInferProfileList(profile), cont_name(cont_name), cq(cq), stub(stub), lambda(lambda),
          gamma(gamma), clip_epsilon(clip_epsilon), penalty_weight(penalty_weight), theta(theta), sigma(sigma), phi(phi),
          rho(rho), state_size(state_size), timeout_size(timeout_size), max_batch(max_batch),
          base_batch(base_batch), scaling_size(scaling_size), update_steps(update_steps),
          update_steps_inc(update_steps_inc), federated_steps(federated_steps) {
    path = "../models/fcpo_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log_" + getTimestampString() + ".csv");

    model = std::make_shared<MultiPolicyNet>(state_size, max_batch, timeout_size, scaling_size);
    std::string model_save = path + "/latest_model.pt";
    if (std::filesystem::exists(model_save)) {
        torch::load(model, model_save);
    } else {
        torch::manual_seed(seed);
        for (auto& p : model->named_parameters()) {
            if(p.key().find("norm") != std::string::npos) continue;
            // Initialize weights and biases
            if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::xavier_uniform_(p.value());
            } else if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::constant_(p.value(), 0);
            }
        }
    };
    model->to(precision);
    optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(1e-3));

    cumu_reward = 0.0;
    last_batching = 0;
    last_slo = 0.0;
    steps_counter = 0;
    experiences = ExperienceBuffer(200);
}

void FCPOAgent::update() {
    steps_counter = 0;
    if (federated_steps_counter == 0) {
        experiences.clear();
        spdlog::get("container_agent")->trace("Waiting for federated update!");
        return;
    }
    spdlog::get("container_agent")->info("Locally training RL Agent at cumulative Reward {}!", cumu_reward);
    Stopwatch sw;
    sw.start();

    auto [policy1, policy2, policy3, val] = model->forward(torch::stack(experiences.get_states()));
    T action1_probs = torch::softmax(policy1, -1);
    T action1_log_probs = torch::log(action1_probs.gather(-1, torch::tensor(experiences.get_batching()).reshape({-1, 1})).squeeze(-1));
    T action2_probs = torch::softmax(policy2, -1);
    T action2_log_probs = torch::log(action2_probs.gather(-1, torch::tensor(experiences.get_timeout()).reshape({-1, 1})).squeeze(-1));
    T action3_probs = torch::softmax(policy3, -1);
    T action3_log_probs = torch::log(action3_probs.gather(-1, torch::tensor(experiences.get_scaling()).reshape({-1, 1})).squeeze(-1));
    T new_log_probs = (1 * action1_log_probs + 1 * action2_log_probs + 1 * action3_log_probs).squeeze(-1);
    // code if using SinglePolicyNet
    // auto [policy, val] = model->forward(torch::stack(experiences.get_states()));
    // T actions = model->combine_actions(experiences.get_timeout(), experiences.get_batching(), experiences.get_scaling(), max_batch, scaling_size).to(torch::kInt64);
    // T new_log_probs = torch::log(policy.gather(-1, actions.reshape({-1, 1})).squeeze(-1));

    T ratio = torch::exp(new_log_probs - torch::stack(experiences.get_log_probs()));

    if (experiences.get_rewards().size() < experiences.get_states().size()) {
        ratio = ratio.slice(0, 1, ratio.size(0), 1);
        val = val.slice(0, 1, val.size(0), 1);
    }

    T loss = torch::tensor(0.0), policy_loss = torch::tensor(0.0), value_loss = torch::tensor(0.0);
    try {
        T clipped_ratio = torch::clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon);
        T advantages = computeGae();
        T policy_loss_1 = -torch::min(ratio * advantages, clipped_ratio * advantages).to(precision);
        T policy_loss_2 = torch::exp(-torch::tensor(experiences.get_rewards()).to(precision));
        policy_loss_2 = torch::min((10 / ratio) * policy_loss_2, (10 / clipped_ratio) * policy_loss_2);
        policy_loss = 10 * torch::mean(policy_loss_1 + policy_loss_2);

        value_loss = torch::mse_loss(val.squeeze(), computeCumuRewards());
        spdlog::get("container_agent")->info("RL Agent Policy Loss: {}, Value Loss: {}",
                                             policy_loss.item<double>(), value_loss.item<double>());
        loss = (policy_loss + value_loss);
    } catch (const std::exception& e) {
        spdlog::get("container_agent")->error("Error in loss computation: {}", e.what());
        reset();
        return;
    }

    // Backpropagation
    std::unique_lock<std::mutex> lock(model_mutex);
    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
    sw.stop();

    double avg_reward = cumu_reward / (double) update_steps;
    out << "episodeEnd," << sw.elapsed_microseconds() << "," << federated_steps_counter << "," << steps_counter
        << "," << cumu_reward << "," << avg_reward << "," << loss.item<double>() << "," << policy_loss.item<double>() << "," << value_loss.item<double>() << std::endl;
    if (last_avg_reward < avg_reward) {
        last_avg_reward = avg_reward;
        torch::save(model, path + "/latest_model.pt");
    }

    if (federated_steps > 0 && ++federated_steps_counter % federated_steps == 0) {
        spdlog::get("container_agent")->info("Federated training RL agent!");
        federated_steps_counter = 0; // 0 means that we are waiting for federated update to come back
        federatedUpdate(loss.item<double>());
    }
    reset();
}

void FCPOAgent::federatedUpdate(const double loss) {
    FlData request;
    // save model information in uchar* pointer and then reload it from that information
    std::ostringstream oss;
    request.set_name(cont_name);
    request.set_state_size(state_size);
    request.set_timeout_size(timeout_size);
    request.set_max_batch(max_batch);
    request.set_threading_size(scaling_size);
    request.set_latest_loss(loss);
    torch::save(model, oss);
    request.set_network(oss.str());
    oss.str("");
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
        spdlog::get("container_agent")->error("Federated update failed: {}", status.error_message());
        federated_steps_counter = 1; // 1 means that we are starting local updates again until the next federation
    } else {
        federatedStartTime = std::chrono::high_resolution_clock::now();
    }
}

void FCPOAgent::federatedUpdateCallback(FlData &response) {
    spdlog::get("container_agent")->info("Federated Update received!");
    std::istringstream iss(response.network());
    std::unique_lock<std::mutex> lock(model_mutex);
    torch::load(model, iss);
    auto states = torch::stack(experiences.get_states()).to(precision);
    auto policy1_history = torch::tensor(experiences.get_batching()).to(precision);
    auto policy2_history = torch::tensor(experiences.get_timeout()).to(precision);
    auto policy3_history = torch::tensor(experiences.get_scaling()).to(precision);
    if (policy1_history.size(0) != policy2_history.size(0) || policy1_history.size(0) != policy3_history.size(0) || policy2_history.size(0) != policy3_history.size(0)) {
        spdlog::get("container_agent")->error("Mismatch in experience buffer sizes during federated update callback!");
        federated_steps_counter = 1;
        return;
    }
    if (states.size(0) > policy1_history.size(0)) {
        spdlog::get("container_agent")->warn("Mismatch in experience buffer sizes during federated update callback! Removing last state.");
        states = states.slice(0, 0, policy1_history.size(0), 1);
        if (states.size(0) != policy1_history.size(0)) {
            spdlog::get("container_agent")->error("Mismatch in experience buffer sizes during federated update callback after slicing!");
            federated_steps_counter = 1;
            return;
        }
    }

    model->train();
    optimizer->zero_grad();
    auto [policy1_output, policy2_output, policy3_output, value_estimated] = model->forward(states);

    auto policy1_loss = torch::nll_loss(policy1_output, policy1_history);
    auto policy2_loss = torch::nll_loss(policy2_output, policy2_history);
    auto policy3_loss = torch::nll_loss(policy3_output, policy3_history);

    auto loss = policy1_loss + policy2_loss + policy3_loss;
    value_estimated.detach();
    loss.to(precision).backward();
    optimizer->step();

    uint64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - federatedStartTime).count();
    out << "federatedAggregation," << latency << "," << federated_steps_counter << "," << steps_counter << std::endl;
    steps_counter = 0;
    federated_steps_counter = 1; // 1 means that we are starting local updates again until the next federation
    if (update_steps < 300) update_steps += update_steps_inc; // Increase the number of steps for the next updates every time we finish federation
    reset();
}

void FCPOAgent::rewardCallback(double throughput, double latency, double oversize, double memory_use) {
    if (update_steps > 0 ) {
        if (first) { // First reward is not valid and needs to be discarded
            first = false;
            return;
        }
        double reward = theta * throughput - sigma * latency - phi * oversize
                - rho * (memory_use / (double) (batchInferProfileList[max_batch].gpuMemUsage + 1.0));
        spdlog::get("container_agent")->trace("RL Agent Reward: throughput: {}, latency_penalty: {}, oversize_penalty: {}, memory_use: {}, profile_max_memory: {}, penalty: {}, reward: {}",
                                             throughput, latency, oversize, memory_use,
                                             batchInferProfileList[max_batch].gpuMemUsage, penalty, reward);
        reward = std::max(-1.0, std::min(1.0, reward)); // Clip reward to [-1, 1]
        if (penalty) reward *= (reward < 0) ? 1.1 : -0.9;
        experiences.add_reward(reward);
        cumu_reward += reward;
    }
}

void FCPOAgent::setState(double curr_timeout, double curr_batch, double curr_scaling,  double arrival,
                         double pre_queue_size, double inf_queue_size, double post_queue_size, double slo,
                         double memory_use) {
    last_slo = slo;
    double maxMemory;
    if (batchInferProfileList.empty())
        maxMemory = memory_use * 2.0 + 1;
    else
        maxMemory = (double) batchInferProfileList[max_batch].gpuMemUsage + 1.0;
    spdlog::get("container_agent")->trace("RL Agent State: curr_timeout: {}, curr_batch: {}, curr_scaling: {}, arrival: {}, pre_queue_size: {}, inf_queue_size: {}, post_queue_size: {}, slo: {}, memory_use: {}, maxMemory: {}",
                                         curr_timeout, curr_batch, curr_scaling, arrival, pre_queue_size, inf_queue_size, post_queue_size, slo, memory_use, maxMemory);
    state = torch::tensor({curr_batch / max_batch, curr_timeout / timeout_size,
                           curr_scaling / scaling_size, arrival, pre_queue_size / 100.0, inf_queue_size / 100.0,
                           post_queue_size / 100.0, slo / 100000.0,
                           memory_use / maxMemory}, precision);
}

void FCPOAgent::selectAction() {
    std::unique_lock<std::mutex> lock(model_mutex);
    auto [policy1, policy2, policy3, val] = model->forward(state);
    T action_dist = torch::multinomial(policy1, 1);
    batching = action_dist.item<int>();
    action_dist = torch::multinomial(policy2, 1);
    timeout = action_dist.item<int>();
    action_dist = torch::multinomial(policy3, 1);
    scaling = action_dist.item<int>();
    // code if using SinglePolicyNet
    // auto [policy, val] = model->forward(state);
    // T action_dist = torch::multinomial(policy, 1);  // Sample from policy (discrete distribution)
    // std::tie(timeout, batching, scaling) = model->interpret_action(action_dist.item<int>(), max_batch, scaling_size);
    // log_prob = torch::log(policy[action_dist.item<int>()]);

    // ensure that the action is within the valid range of the given SLO
    if (batching > 0 && batchInferProfileList[batching].p95inferLat > last_slo) {
        penalty = true;
        batching = std::max(last_batching, base_batch);
    }
    last_batching = batching;
    if (update_steps > 0 ) {
        log_prob = torch::log(policy1[batching]) + torch::log(policy2[timeout]) + torch::log(policy3[scaling]);
        // log_prob = torch::log(policy[action_dist.item<int>()]);
        experiences.add(state, log_prob, val, timeout, batching, scaling);
    }
}

T FCPOAgent::computeCumuRewards() const {
    std::vector<double> discounted_rewards, rewards = experiences.get_rewards();
    double cumulative = 0.0;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        cumulative = *it + gamma * cumulative;
        discounted_rewards.insert(discounted_rewards.begin(), cumulative);
    }
    return torch::tensor(discounted_rewards).to(precision);
}

T FCPOAgent::computeGae() const {
    std::vector<double> advantages, rewards = experiences.get_rewards();
    std::vector<T> values = experiences.get_values();
    double gae = 0.0;
    double next_value = 0.0;
    for (int t = rewards.size() - 1; t >= 0; --t) {
        double delta = rewards[t] + gamma * next_value - values[t].item<double>();
        gae = delta + gamma * lambda * gae;
        advantages.insert(advantages.begin(), gae);
        next_value = values[t].item<double>();
    }
    return torch::tensor(advantages).to(precision);
}

std::tuple<int, int, int> FCPOAgent::runStep() {
    Stopwatch sw;
    sw.start();
    selectAction();

    if (update_steps > 0 ) steps_counter++;
    sw.stop();
    out << "step," << sw.elapsed_microseconds() << "," << federated_steps_counter << "," << steps_counter << "," << cumu_reward  << "," << timeout << "," << batching << "," << scaling << std::endl;

    if (update_steps > 0 ) {
        if (steps_counter % update_steps == 0) {
            update();
        }
    }
    return std::make_tuple((timeout + 1) % (timeout_size), batching + 1, scaling);
}


FCPOServer::FCPOServer(std::string run_name, nlohmann::json parameters, uint state_size, torch::Dtype precision)
        : precision(precision), state_size(state_size) {
    path = "../models/fcpo_learning/" + run_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    std::random_device rd;
    re = std::mt19937(rd());
    model = std::make_shared<MultiPolicyNet>(state_size, 1, 1, 1);
    std::string model_save = path + "/latest_model.pt";
    if (std::filesystem::exists(model_save)) torch::load(model, model_save);
    model->to(precision);
    optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(1e-3));

    lambda = parameters["lambda"];
    gamma = parameters["gamma"];
    clip_epsilon = parameters["clip_epsilon"];
    penalty_weight = parameters["penalty_weight"];

    theta = parameters["theta"];
    sigma = parameters["sigma"];
    phi = parameters["phi"];

    update_steps = parameters["update_steps"];
    update_step_incs = parameters["update_step_incs"];
    federated_steps = parameters["federated_steps"];
    seed = parameters["seed"];

    federated_clients = {};
    run = true;
    std::thread t(&FCPOServer::proceed, this);
    t.detach();
}

bool FCPOServer::addClient(FlData &request, std::shared_ptr<ControlCommands::Stub> stub, CompletionQueue *cq) {
    std::istringstream iss(request.network());
    federated_clients.push_back({request,
                                 std::make_shared<MultiPolicyNet>(request.state_size(), request.timeout_size(),
                                                                  request.max_batch(), request.threading_size()),
                                 stub, cq});
    try {
        torch::load(federated_clients.back().model, iss);
        federated_clients.back().model->to(precision);
    } catch (const std::exception& e) {
        spdlog::get("container_agent")->error("Error in loading model: {}", e.what());
        federated_clients.pop_back();
        return false;
    }
    return true;
}

void FCPOServer::proceed() {
    while (run) {
        if (client_counter == 0 || federated_clients.size() < (client_counter / 2)) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        for (unsigned int i = 0; i < client_counter; i++) {
            if (federated_clients.size() == client_counter) break;
            std::this_thread::sleep_for(std::chrono::seconds(1)); // add some wait to potentially receive more clients
        }

        spdlog::get("container_agent")->info("Starting Federated Aggregation of FCPO Agents!");
        auto participants = federated_clients;
        federated_clients.clear();

        std::vector<torch::Tensor> aggregated_shared_params;
        for (const auto& param : model->parameters()) {
            aggregated_shared_params.push_back(param.clone());
        }
        std::vector<std::map<std::vector<int64_t>, T>> action_heads_by_size =
                std::vector<std::map<std::vector<int64_t>, T>>(aggregated_shared_params.size());

        double last_loss = 0.0;
        for (auto& client : participants) {
            size_t i = 0;
            for (const auto& param : client.model->parameters()) {
                if (aggregated_shared_params[i].sizes() == param.sizes()) {
                    aggregated_shared_params[i] = aggregated_shared_params[i] + param;
                } else {
                    auto size_key = param.sizes().vec();
                    if (action_heads_by_size[i].find(size_key) == action_heads_by_size[i].end()) {
                        action_heads_by_size[i][size_key] = param.clone();
                    } else {
                        double factor = 1.0 / (client.data.latest_loss() - last_loss);
                        action_heads_by_size[i][size_key] = action_heads_by_size[i][size_key] + param * factor;
                    }
                }
                i++;
                last_loss = client.data.latest_loss();
            }
        }

        for (auto& param : aggregated_shared_params) {
            param /= static_cast<int64_t>(participants.size() + 1);
        }

        for (auto& action_head : action_heads_by_size) {
            for (auto &[key, value]: action_head) {
                action_head[key] = value / static_cast<int64_t>(action_head.size());
            }
        }

        for (auto client: participants) {
            for (int i = 0; i <= 5; i++) client.model->parameters()[i] = aggregated_shared_params[i].clone();
            returnFLModel(client);
        }

        for (unsigned int i = 0; i < aggregated_shared_params.size(); i++) {
            model->parameters()[i] = aggregated_shared_params[i].clone();
        }
    }
}

void FCPOServer::returnFLModel(ClientModel &client) {
    FlData request;
    std::ostringstream oss;
    torch::save(client.model, oss);
    request.set_name(client.data.name());
    request.set_network(oss.str());
    ClientContext context;
    EmptyMessage reply;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            client.stub->AsyncReturnFl(&context, request, client.cq));
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    if (client.cq != nullptr) GPR_ASSERT(client.cq->Next(&got_tag, &ok));
}