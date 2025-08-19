#include "bandwidth_predictor.h"

torch::Tensor TPALSTMImpl::forward(torch::Tensor input) {
    // (Optional) mimic .cuda() in Python if requested
    if (A.cuda && input.device().is_cpu()) {
        input = input.to(torch::kCUDA);
    }

    //const int64_t B_in = input.size(0);
    const int64_t T    = input.size(1);
    const int64_t F    = input.size(2);
    TORCH_CHECK(F == D.original_columns, "Input feature size must equal original_columns");

    // ---- Step 1: LSTM ----
    // LSTM expects [T, B, F]
    auto input_to_lstm = input.permute({1, 0, 2}).contiguous();
    torch::Tensor lstm_out; // [T,B,H]
    torch::Tensor h_n, c_n; // [num_layers,B,H]
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> output;
    output = lstm->forward(input_to_lstm);
    std::tie(lstm_out, std::ignore) = output;
    std::tie(h_n, c_n) = std::get<1>(output);
    // hn: last layer hidden -> [B,1,H] (match Python)
    auto hn_last = h_n.index({h_n.size(0) - 1}); // [B,H]
    auto hn = hn_last.view({1, hn_last.size(0), hn_last.size(1)}); // [1,B,H]
    hn = hn.permute({1, 0, 2}).contiguous(); // [B,1,H]

    // ---- Step 2: Convolution over hidden states ----
    // Align to [B,T,H]
    auto output_realigned = lstm_out.permute({1, 0, 2}).contiguous(); // [B,T,H]

    // Conv2d expects [N, C_in=1, H=time, W=Hfeat]
    auto input_to_conv = output_realigned.view({-1, 1, T, H}); // [B,1,T,H]
    auto conv_out = torch::relu(compute_convolution->forward(input_to_conv)); // [B,hidC,(T-Ck+1),1]
    conv_out = dropout->forward(conv_out);
    conv_out = conv_out.squeeze(3); // [B,hidC,(T-Ck+1)]
    const int64_t L = conv_out.size(2); // temporal length after conv

    // ---- Step 3: Attention over conv outputs ----
    // We will potentially pad up to parameter batch size like the Python code.
    const int64_t B_param = attention_matrix.size(0);

    auto device = attention_matrix.device();
    auto final_hn = torch::zeros({B_param, 1, H}, input.options().device(device));
    auto final_conv_out = torch::zeros({B_param, hidC, L}, input.options().device(device));

    int64_t diff = 0;
    if (hn.size(0) < B_param) {
        final_hn.index_put_({torch::indexing::Slice(0, hn.size(0)), "..."},
                            hn.to(device));
        final_conv_out.index_put_({torch::indexing::Slice(0, conv_out.size(0)), "..."},
                                  conv_out.to(device));
        diff = B_param - static_cast<int64_t>(hn.size(0));
    } else {
        final_hn = hn.to(device);
        final_conv_out = conv_out.to(device);
    }

    // Shapes for scoring
    auto conv_for_scoring = final_conv_out.permute({0, 2, 1}).contiguous(); // [B,L,hidC]
    auto final_hn_realigned = final_hn.permute({0, 2, 1}).contiguous();     // [B,H,1]

    // mat1 = [B,L,hidC] x [B,hidC,H] -> [B,L,H]
    auto mat1 = torch::bmm(conv_for_scoring, attention_matrix); // [B,L,H]
    // scoring = [B,L,H] x [B,H,1] -> [B,L,1]
    auto scoring = torch::bmm(mat1, final_hn_realigned);        // [B,L,1]
    auto alpha = torch::sigmoid(scoring);                       // [B,L,1]
    // context = sum_t alpha_t * conv_for_scoring_t  -> [B,hidC]
    auto context_vec = (alpha * conv_for_scoring).sum(1);       // [B,hidC]
    context_vec = context_vec.view({-1, hidC, 1});              // [B,hidC,1]

    // ---- Step 4: Combine with final state and project ----
    // h_intermediate = F * hn + C * context ;  F:[B,H,H], hn:[B,H,1], C:[B,H,hidC], ctx:[B,hidC,1]
    auto h_intermediate =
            torch::bmm(final_state_matrix, final_hn_realigned) +
            torch::bmm(context_vector_matrix, context_vec);         // [B,H,1]

    // result = W * h_intermediate ; W:[B,orig,F=H]  -> [B,orig,1] -> squeeze -> [B,orig]
    auto result = torch::bmm(final_matrix, h_intermediate);     // [B,orig,1]
    result = result.permute({0, 2, 1}).contiguous();            // [B,1,orig]
    result = result.squeeze(1);                                 // [B,orig]

    // remove padded tail if any
    if (diff > 0) {
        const int64_t keep = result.size(0) - diff;
        result = result.index({torch::indexing::Slice(0, keep), torch::indexing::Slice()}); // [B_in, orig]
    }

    // ---- Highway ----
    torch::Tensor res = result;
    if (hw > 0) {
        TORCH_CHECK(T >= hw, "Sequence length T must be >= highway_window");
        // z = x[:, -hw:, :] -> [B, F, hw] -> view(-1, hw) -> Linear(hw->1) -> reshape [B, F]
        auto z = input.index({torch::indexing::Slice(), torch::indexing::Slice(T - hw, T), torch::indexing::Slice()}); // [B,hw,F]
        z = z.permute({0, 2, 1}).contiguous();                 // [B,F,hw]
        z = z.view({-1, hw});                                   // [B*F, hw]
        z = highway->forward(z);                               // [B*F, 1]
        z = z.view({-1, D.original_columns});                   // [B,F]
        res = res + z;
    }

    return torch::sigmoid(res);
}

BandwidthPredictor::BandwidthPredictor() {
    std::string model_save = "../models/bandwidth_prediction";
    std::filesystem::create_directories(std::filesystem::path(model_save));
    model_save += "/latest_model.pt";
    if (std::filesystem::exists(model_save)) {
        torch::load(model, model_save);
    } else {
        torch::manual_seed(42);
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
}

float BandwidthPredictor::predict(const std::vector<float> &input) {
    if (input.size() < args.window) {
        spdlog::get("container_agent")->warn("Input size {} is smaller than the model window size {}. Cannot predict bandwidth.", input.size(), args.window);
        return 0.0f;
    }

    torch::Tensor input_tensor = torch::tensor(input, torch::kFloat32).view({1, -1, 1}); // [1, T, 1]
    input_tensor = input_tensor.to(torch::kCUDA); // Move to GPU if available

    torch::Tensor output = model->forward(input_tensor);
    output = output.squeeze(0).squeeze(0); // [T]

    return output.item<float>();
}