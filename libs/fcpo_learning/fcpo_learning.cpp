#include "fcpo_learning.h"

FCPOAgent::FCPOAgent(std::string& cont_name, uint state_size, uint resolution_size, uint max_batch,  uint threading_size,
                   CompletionQueue *cq, std::shared_ptr<InDeviceMessages::Stub> stub, torch::Dtype precision, 
                   uint update_steps, uint update_steps_inc, uint federated_steps, double lambda, double gamma,
                   double clip_epsilon, double penalty_weight)
                   : precision(precision), cont_name(cont_name), cq(cq), stub(stub), lambda(lambda), gamma(gamma),
                     clip_epsilon(clip_epsilon), penalty_weight(penalty_weight), state_size(state_size),
                     resolution_size(resolution_size), max_batch(max_batch), threading_size(threading_size),
                     update_steps(update_steps), update_steps_inc(update_steps_inc), federated_steps(federated_steps) {
    path = "../models/fcpo_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    std::random_device rd;
    re = std::mt19937(rd());
    model = std::make_shared<MultiPolicyNet>(state_size, resolution_size, max_batch, threading_size);
    std::string model_save = path + "/latest_model.pt";
    if (std::filesystem::exists(model_save)) torch::load(model, model_save);
    model->to(precision);
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

    cumu_reward = 0.0;
    states = {};
    resolution_actions = {};
    batching_actions = {};
    scaling_actions = {};
    rewards = {};
    log_probs = {};
    values = {};
}

void FCPOAgent::update() {
    steps_counter = 0;
    if (federated_steps_counter == 0) {
        spdlog::get("container_agent")->trace("Waiting for federated update, cancel !");
        return;
    }
    spdlog::get("container_agent")->info("Locally training RL agent at cumulative Reward {}!", cumu_reward);
    Stopwatch sw;
    sw.start();

    auto [policy1, policy2, policy3, val] = model->forward(torch::stack(states));
    T action1_probs = torch::softmax(policy1, -1);
    T action1_log_probs = torch::log(action1_probs.gather(-1, torch::tensor(resolution_actions).reshape({-1, 1})).squeeze(-1));
    T action2_probs = torch::softmax(policy2, -1);
    T action2_log_probs = torch::log(action2_probs.gather(-1, torch::tensor(batching_actions).reshape({-1, 1})).squeeze(-1));
    T action3_probs = torch::softmax(policy3, -1);
    T action3_log_probs = torch::log(action3_probs.gather(-1, torch::tensor(scaling_actions).reshape({-1, 1})).squeeze(-1));
    T new_log_probs = (action1_log_probs + action2_log_probs + action3_log_probs).squeeze(-1);

    T ratio = torch::exp(new_log_probs - torch::stack(log_probs));

    if (rewards.size() < states.size()) {
        ratio = ratio.slice(0, 1, ratio.size(0), 1);
        val = val.slice(0, 1, val.size(0), 1);
    }

    T clipped_ratio = torch::clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon);
    T advantages = computeGae();
    T policy_loss = -torch::min(ratio * advantages, clipped_ratio * advantages).to(precision).mean();

    T value_loss = torch::mse_loss(val.squeeze(), computeCumuRewards());
    T policy1_penalty = penalty_weight * torch::mean(torch::tensor(resolution_actions).to(precision));
    T policy3_penalty = penalty_weight * torch::mean(torch::tensor(scaling_actions).to(precision));
    T loss = (policy_loss + 0.5 * value_loss + policy1_penalty + policy3_penalty);

    // Backpropagation
    optimizer->zero_grad();
    loss.backward();
    std::lock_guard<std::mutex> lock(model_mutex);
    optimizer->step();
    sw.stop();

    std::cout << "Training: " << sw.elapsed_microseconds() << std::endl;
    out << "episodeEnd," << sw.elapsed_microseconds() << "," << federated_steps_counter << "," << steps_counter << "," << cumu_reward << "," << cumu_reward / (double) steps_counter << std::endl;

    if (federated_steps_counter++ % federated_steps == 0) {
        spdlog::get("container_agent")->info("Federated training RL agent!");
        federated_steps_counter = 0; // 0 means that we are waiting for federated update to come back
        federatedUpdate();
    }
    reset();
}

void FCPOAgent::federatedUpdate() {
    FlData request;
    // save model information in uchar* pointer and then reload it from that information
    std::ostringstream oss;
    request.set_name(cont_name);
    request.set_state_size(state_size);
    request.set_resolution_size(resolution_size);
    request.set_max_batch(max_batch);
    request.set_threading_size(threading_size);
    torch::save(model, oss);
    request.set_network(oss.str());
    oss.str("");
    torch::save(torch::stack(states).to(precision), oss);
    request.set_states(oss.str());
    oss.str("");
    torch::save(torch::tensor(resolution_actions).to(precision), oss);
    request.set_resolution_actions(oss.str());
    oss.str("");
    torch::save(torch::tensor(batching_actions).to(precision), oss);
    request.set_batching_actions(oss.str());
    oss.str("");
    torch::save(torch::tensor(scaling_actions).to(precision), oss);
    request.set_scaling_actions(oss.str());
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
        spdlog::get("container_agent")->error("Federated update failed: {}", status.error_message());
        federated_steps_counter = 1; // 1 means that we are starting local updates again until the next federation
    }
}

void FCPOAgent::federatedUpdateCallback(FlData &response) {
    std::istringstream iss(response.network());
    std::lock_guard<std::mutex> lock(model_mutex);
    torch::load(model, iss);
    steps_counter = 0;
    update_steps += update_steps_inc; // Increase the number of steps for the next updates every time we finish federation
    federated_steps_counter = 1; // 1 means that we are starting local updates again until the next federation
    reset();
}

void FCPOAgent::rewardCallback(double throughput, double drops, double latency_penalty, double oversize_penalty) {
    if (first) { // First reward is not valid and needs to be discarded
        first = false;
        return;
    }
    rewards.push_back(2 * throughput - drops - latency_penalty + (1 - oversize_penalty));
}

void FCPOAgent::setState(double curr_batch, double curr_resolution_choice, double arrival, double pre_queue_size, double inf_queue_size) {
    state = torch::tensor({curr_batch / max_batch, curr_resolution_choice, arrival, pre_queue_size, inf_queue_size}, precision);
}

std::tuple<int, int, int> FCPOAgent::selectAction() {
    std::lock_guard<std::mutex> lock(model_mutex);
    auto [policy1, policy2, policy3, val] = model->forward(state);

    T action_dist = torch::multinomial(policy1, 1);  // Sample from policy (discrete distribution)
    int resolution = action_dist.item<int>();  // Convert tensor to int action
    action_dist = torch::multinomial(policy2, 1);
    int batching = action_dist.item<int>();
    action_dist = torch::multinomial(policy3, 1);
    int scaling = action_dist.item<int>();


    log_prob = torch::log(policy1[resolution]) + torch::log(policy2[batching]) + torch::log(policy3[scaling]);
    states.push_back(state);
    log_probs.push_back(log_prob);
    values.push_back(val);
    resolution_actions.push_back(resolution);
    batching_actions.push_back(batching);
    scaling_actions.push_back(scaling);
    return std::make_tuple(resolution, batching, scaling);
}

T FCPOAgent::computeCumuRewards() const {
    std::vector<double> discounted_rewards;
    double cumulative = 0.0;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        cumulative = *it + gamma * cumulative;
        discounted_rewards.insert(discounted_rewards.begin(), cumulative);
    }
    return torch::tensor(discounted_rewards).to(precision);
}

T FCPOAgent::computeGae() const {
    std::vector<double> advantages;
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
    actions = selectAction();
    cumu_reward += (steps_counter) ? rewards[steps_counter - 1] : 0;

    steps_counter++;
    sw.stop();
    out << "step," << sw.elapsed_microseconds() << "," << federated_steps_counter << "," << steps_counter << "," << cumu_reward  << "," << std::get<0>(actions) << "," << std::get<1>(actions) << "," << std::get<2>(actions) << std::endl;

    if (steps_counter%update_steps == 0) {
        std::thread t(&FCPOAgent::update, this);
        t.detach();
    }
    return std::make_tuple(std::get<0>(actions) + 1, std::get<1>(actions) + 1, std::get<2>(actions));
}


FCPOServer::FCPOServer(std::string run_name, uint state_size, torch::Dtype precision)
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
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

    lambda = 0.95;
    gamma = 0.99;
    penalty_weight = 0.1;
    clip_epsilon = 0.2;
    federated_clients = {};
    run = true;
    std::thread t(&FCPOServer::proceed, this);
    t.detach();
}

void FCPOServer::addClient(FlData &request, std::shared_ptr<ControlCommands::Stub> stub, CompletionQueue *cq) {
    std::istringstream iss(request.network());
    federated_clients.push_back({request,
                                 std::make_shared<MultiPolicyNet>(request.state_size(), request.resolution_size(),
                                                                  request.max_batch(), request.threading_size()),
                                                                      stub, cq});
    torch::load(federated_clients.back().model, iss);
    federated_clients.back().model->to(precision);
}

void FCPOServer::proceed() {
    while (run) {
        if (client_counter == 0 || federated_clients.size() < client_counter) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        spdlog::get("container_agent")->info("Starting Federated Aggregation of FCPO Agents!");

        std::vector<torch::Tensor> aggregated_params;
        for (auto param: {model->shared_layer1->weight, model->shared_layer1->bias, model->shared_layer2->weight, model->shared_layer2->bias, model->value_head->weight, model->value_head->bias}) {
            aggregated_params.push_back(param);
        }
        for (auto client: federated_clients) {
            aggregated_params[0] += client.model->shared_layer1->weight;
            aggregated_params[1] += client.model->shared_layer1->bias;
            aggregated_params[2] += client.model->shared_layer2->weight;
            aggregated_params[3] += client.model->shared_layer2->bias;
            aggregated_params[4] += client.model->value_head->weight;
            aggregated_params[5] += client.model->value_head->bias;
        }
        for (auto& param : aggregated_params) {
          param /= static_cast<int64_t>(federated_clients.size() + 1);
        }

        const std::vector<torch::Tensor> new_params = aggregated_params;
        T states, resolution_actions, batching_actions, scaling_actions, rewards, values;
        for (auto client: federated_clients) {
            client.model->shared_layer1->weight.copy_(new_params[0]);
            client.model->shared_layer1->bias.copy_(new_params[1]);
            client.model->shared_layer2->weight.copy_(new_params[2]);
            client.model->shared_layer2->bias.copy_(new_params[3]);
            client.model->value_head->weight.copy_(new_params[4]);
            client.model->value_head->bias.copy_(new_params[5]);
            client.model->train();
            optimizer->zero_grad();
            std::istringstream iss(client.data.states());
            torch::load(states, iss);
            auto [policy1_output, policy2_output, policy3_output, value] = client.model->forward(states);

            auto policy1_loss = torch::nn::functional::nll_loss(policy1_output, resolution_actions);
            auto policy2_loss = torch::nn::functional::nll_loss(policy2_output, batching_actions);
            auto policy3_loss = torch::nn::functional::nll_loss(policy3_output, scaling_actions);

            auto loss = policy1_loss + policy2_loss + policy3_loss;
            loss.to(precision).backward();
            optimizer->step();

            returnFLModel(client);
        }
        federated_clients.clear();
        model->shared_layer1->weight.copy_(new_params[0]);
        model->shared_layer1->bias.copy_(new_params[1]);
        model->shared_layer2->weight.copy_(new_params[2]);
        model->shared_layer2->bias.copy_(new_params[3]);
        model->value_head->weight.copy_(new_params[4]);
        model->value_head->bias.copy_(new_params[5]);
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