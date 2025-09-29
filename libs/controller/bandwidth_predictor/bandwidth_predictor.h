#ifndef PIPEPLUSPLUS_BANDWIDTH_PREDICTOR_H
#define PIPEPLUSPLUS_BANDWIDTH_PREDICTOR_H
#include "misc.h"
#include <torch/torch.h>

struct TPAArgs {
    bool   cuda = true;
    int64_t window = 168;                 // temporal window size
    int64_t hidRNN = 64;                  // (not used in this forward; parity)
    int64_t hidden_state_features = 64;   // LSTM hidden size
    int64_t hidCNN = 1;                   // number of conv filters
    int64_t hidSkip = 0;                  // (not used in this forward; parity)
    int64_t CNN_kernel = 1;               // temporal kernel
    int64_t skip = 0;                     // (not used in this forward; parity)
    int64_t highway_window = 0;           // HW size
    int64_t num_layers_lstm = 1;
    int64_t batch_size = 32;              // used to size attention matrices
    double  dropout = 0.0;
    std::string output_fun = "sigmoid";   // kept for parity (we always return sigmoid at end)
};

struct TPADataShape {
    int64_t original_columns; // feature dimension of inputs and outputs
};

// ========== Model ==========
struct TPALSTM : torch::nn::Module {
    TPAArgs A;
    TPADataShape D;

    int64_t window_length;
    int64_t hidR;
    int64_t H;       // hidden_state_features
    int64_t hidC;
    int64_t hidS;
    int64_t Ck;
    int64_t skip;
    int64_t pt;      // (A.window - Ck) // skip  (parity; unused here)
    int64_t hw;
    int64_t num_layers_lstm;

    // modules
    torch::nn::LSTM lstm{nullptr};                      // input_size = original_columns, hidden_size = H
    torch::nn::Conv2d compute_convolution{nullptr};     // (1 -> hidC) kernel = (Ck, H)
    torch::nn::Conv2d conv1{nullptr};                   // parity (unused in forward)
    torch::nn::GRU GRU1{nullptr};                       // parity (unused in forward)
    torch::nn::Dropout dropout{nullptr};
    torch::nn::GRU GRUskip{nullptr};                    // parity (unused in forward)
    torch::nn::Linear linear1{nullptr};                 // parity (unused in forward)
    torch::nn::Linear highway{nullptr};                 // hw -> 1

    // batch-dependent parameters
    torch::Tensor attention_matrix;     // [B, hidC, H]
    torch::Tensor context_vector_matrix;// [B, H, hidC]
    torch::Tensor final_state_matrix;   // [B, H, H]
    torch::Tensor final_matrix;         // [B, original_columns, H]

    // actor
    TPALSTM(const TPAArgs& args, const TPADataShape& data)
            : A(args), D(data), window_length(args.window), hidR(args.hidRNN), H(args.hidden_state_features),
              hidC(args.hidCNN), hidS(args.hidSkip), Ck(args.CNN_kernel), skip(args.skip),
              pt((args.window - args.CNN_kernel) / (args.skip == 0 ? 1 : args.skip)),
              hw(args.highway_window), num_layers_lstm(args.num_layers_lstm) {

        // LSTM
        lstm = torch::nn::LSTM(torch::nn::LSTMOptions(D.original_columns, H).num_layers(num_layers_lstm).bidirectional(false));

        // Conv across rows of hidden states (time x H): kernel (Ck, H)
        compute_convolution = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(1, hidC, {Ck, H}));

        // Parity modules
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, hidC, {Ck, D.original_columns}));
        GRU1 = torch::nn::GRU(torch::nn::GRUOptions(hidC, hidR));
        if (skip > 0) {
            GRUskip = torch::nn::GRU(torch::nn::GRUOptions(hidC, hidS));
            linear1 = torch::nn::Linear(hidR + skip * hidS, D.original_columns);
            register_module("GRUskip", GRUskip);
            register_module("linear1", linear1);
        } else {
            linear1 = torch::nn::Linear(hidR, D.original_columns);
            register_module("linear1", linear1);
        }

        dropout = torch::nn::Dropout(torch::nn::DropoutOptions(A.dropout));
        if (hw > 0) {
            highway = torch::nn::Linear(hw, 1);
            register_module("highway", highway);
        }

        // Register main modules
        register_module("lstm", lstm);
        register_module("compute_convolution", compute_convolution);
        register_module("conv1", conv1);
        register_module("GRU1", GRU1);
        register_module("dropout", dropout);

        // Batch-dependent parameters (match Python shapes exactly)
        auto dev = (A.cuda ? torch::kCUDA : torch::kCPU);
        attention_matrix = register_parameter(
                "attention_matrix",
                torch::empty({A.batch_size, hidC, H}, torch::device(dev)));
        context_vector_matrix = register_parameter(
                "context_vector_matrix",
                torch::empty({A.batch_size, H, hidC}, torch::device(dev)));
        final_state_matrix = register_parameter(
                "final_state_matrix",
                torch::empty({A.batch_size, H, H}, torch::device(dev)));
        final_matrix = register_parameter(
                "final_matrix",
                torch::empty({A.batch_size, D.original_columns, H}, torch::device(dev)));

        torch::nn::init::xavier_uniform_(attention_matrix);
        torch::nn::init::xavier_uniform_(context_vector_matrix);
        torch::nn::init::xavier_uniform_(final_state_matrix);
        torch::nn::init::xavier_uniform_(final_matrix);
    }

    // forward: input x -> [B, T, F]
    torch::Tensor forward(torch::Tensor input);
};

class BandwidthPredictor {
public:
    BandwidthPredictor();
    ~BandwidthPredictor() = default;

    float predict(const std::vector<float> &input);

    int64_t getWindowSize() const {
        return args.window;
    }

private:
    TPAArgs args;
    std::shared_ptr<TPALSTM> model;
};

#endif //PIPEPLUSPLUS_BANDWIDTH_PREDICTOR_H
