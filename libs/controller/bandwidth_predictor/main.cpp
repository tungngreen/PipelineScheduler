#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "bandwidth_predictor.h"
#include <random>

// Command-line flags
ABSL_FLAG(uint16_t, epochs, 10, "Number of epochs");
ABSL_FLAG(uint16_t, batch_size, 32, "Batch size for training");
ABSL_FLAG(double, learning_rate, 0.001, "Learning rate");
ABSL_FLAG(int, seq_len, 10, "Sequence length for LSTM input");
ABSL_FLAG(int, input_size, 1, "Number of input features");
ABSL_FLAG(int, hidden_size, 64, "Hidden size for LSTM");
ABSL_FLAG(int, num_layers, 2, "Number of LSTM layers");

std::vector<double> load_latency_data() {
    nlohmann::json metricsCfgs = nlohmann::json::parse(std::ifstream("../jsons/metricsserver.json"));
    MetricsServerConfigs metricsServerConfigs;
    metricsServerConfigs.from_json(metricsCfgs);
    metricsServerConfigs.schema = "pf15_ppp";
    metricsServerConfigs.user = "controller";
    metricsServerConfigs.password = "agent";
    std::vector<double> data;
    try {
        std::unique_ptr<pqxx::connection> conn = connectToMetricsServer(metricsServerConfigs, "controller");
        pqxx::work txn(*conn);
        pqxx::result r = txn.exec("SELECT transmission_latency FROM pf15_netw_traces ORDER BY timestamp ASC");
        for (auto row : r) {
            data.push_back(row["transmission_latency"].as<double>());
        }
    } catch (const std::exception& e) {
        spdlog::error("Database error: {}", e.what());
    }
    return data;
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> create_batches(
        const std::vector<double>& data, int seq_len, int batch_size) {

    std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
    int num_sequences = data.size() - seq_len;
    for (int i = 0; i < num_sequences; i++) {
        std::vector<double> x_seq(data.begin() + i, data.begin() + i + seq_len);
        double y_val = data[i + seq_len];

        torch::Tensor x_tensor = torch::from_blob(x_seq.data(), {seq_len, 1}).clone();
        torch::Tensor y_tensor = torch::full({1, 1}, y_val);

        batches.push_back({x_tensor, y_tensor});
    }

    std::shuffle(batches.begin(), batches.end(), std::mt19937(std::random_device{}()));
    std::vector<std::pair<torch::Tensor, torch::Tensor>> batch_groups;
    for (size_t i = 0; i < batches.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, batches.size());
        std::vector<torch::Tensor> x_batch, y_batch;
        for (size_t j = i; j < end; j++) {
            x_batch.push_back(batches[j].first.unsqueeze(0)); // shape [1, seq_len, 1]
            y_batch.push_back(batches[j].second);
        }
        batch_groups.push_back({torch::cat(x_batch, 0), torch::cat(y_batch, 0)}); // shape [B, seq_len, 1], [B, 1]
    }

    return batch_groups;
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    int epochs = absl::GetFlag(FLAGS_epochs);
    int batch_size = absl::GetFlag(FLAGS_batch_size);
    double learning_rate = absl::GetFlag(FLAGS_learning_rate);
    int seq_len = absl::GetFlag(FLAGS_seq_len);
    int input_size = absl::GetFlag(FLAGS_input_size);
    int hidden_size = absl::GetFlag(FLAGS_hidden_size);
    int num_layers = absl::GetFlag(FLAGS_num_layers);

    std::vector<spdlog::sink_ptr> loggerSinks = {};
    std::shared_ptr<spdlog::logger> logger;
    setupLogger(
            "../logs",
            "controller",
            0,
            0,
            loggerSinks,
            logger
    );

    spdlog::info("Loading data from database...");
    std::vector<double> latency_data = load_latency_data();
    if (latency_data.size() < seq_len + 1) {
        spdlog::error("Not enough data for training.");
        return 1;
    }

    spdlog::info("Creating batches...");
    auto batches = create_batches(latency_data, seq_len, batch_size);

    // Model setup
    spdlog::info("Initializing TPA-LSTM model...");
    TPAArgs args = {
        .window = seq_len,
        .hidRNN = hidden_size,
        .hidden_state_features = hidden_size,
        .hidCNN = 32,
        .hidSkip = 16,
        .CNN_kernel = 3,
        .skip = 1,
        .highway_window = 1,
        .num_layers_lstm = num_layers,
    };
    TPADataShape shape = {input_size};
    auto model = std::make_shared<TPALSTMImpl>(args, shape);
    model->to(torch::kF32);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    auto criterion = torch::nn::MSELoss();

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        model->train();
        double epoch_loss = 0.0;
        for (auto& batch : batches) {
            auto X = batch.first;
            auto Y = batch.second;

            optimizer.zero_grad();
            auto output = model->forward(X);

            auto loss = criterion(output, Y);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>() * X.size(0);
        }
        double avg_loss = epoch_loss / (batches.size() * batch_size);
        spdlog::info("Epoch [{}/{}], Loss: {:.6f}", epoch + 1, epochs, avg_loss);
    }

    spdlog::info("Training completed successfully.");
    torch::save(model, "tpa_lstm_model.pt");
    std::string path = "../models/bandwidth_prediction";
    std::filesystem::create_directories(std::filesystem::path(path));
    path += "/latest_model.pt";
    torch::save(model, path);
    return 0;
}