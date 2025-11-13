#pragma once
#include "onnxruntime_cxx_api.h"
#include <string>
#include <vector>
#include <memory>

class OpusModel {
#if defined(_WIN32) || defined(_WIN64)
    using ModelPathString = std::wstring;
#else
    using ModelPathString = std::string;
#endif

public:
    OpusModel(const ModelPathString &encoder_model_path, const ModelPathString &decoder_model_path, int num_threads = 4, GraphOptimizationLevel level = ORT_ENABLE_BASIC);

    std::vector<int64_t> infer(std::vector<int64_t>& input_ids, int max_length = 128) const;

    static constexpr int64_t ENCODER_HIDDEN_SIZE = 512;
    static constexpr int64_t EOS_TOKEN_ID = 0;
    static constexpr int64_t PAD_TOKEN_ID = 65000;
    static constexpr int64_t VOCAB_SIZE = 65001;

    Ort::Value runEncoder(std::vector<int64_t> &input_ids, std::vector<int64_t> &attention_mask) const;
    Ort::Value runDecoder(std::vector<int64_t> &dec_input_ids, std::vector<int64_t> &enc_attention_mask, Ort::Value &enc_hidden_states) const;

    static const Ort::Env env_;
    static const char *enc_input_names_[2];
    static const char *enc_output_names_[1];
    static const char *dec_input_names_[3];
    static const char *dec_output_names_[1];

    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;

    Ort::MemoryInfo memory_info_;
};
