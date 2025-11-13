#include "OpusModel.hpp"
#include <algorithm>
#include <iostream>
#include <dml_provider_factory.h>

const Ort::Env OpusModel::env_ = {ORT_LOGGING_LEVEL_WARNING, "LLM"};
const char *OpusModel::enc_input_names_[2] = {"input_ids", "attention_mask"};
const char *OpusModel::enc_output_names_[1] = {"last_hidden_state"};
const char *OpusModel::dec_input_names_[3] = {"encoder_attention_mask", "input_ids", "encoder_hidden_states"};
const char *OpusModel::dec_output_names_[1] = {"logits"};

OpusModel::OpusModel(const ModelPathString &encoder_model_path, const ModelPathString &decoder_model_path, const int num_threads, const GraphOptimizationLevel level) : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(num_threads);
    options.SetGraphOptimizationLevel(level);

    // Enable DirectML execution provider (Windows GPU acceleration)
    // try {
    //     OrtApi const& ortApi = Ort::GetApi();
    //     OrtDmlApi const* ortDmlApi = nullptr;
    //     ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<void const**>(&ortDmlApi));

    //     options.SetExecutionMode(ExecutionMode::ORT_PARALLEL); // For DML EP
    //     options.DisableMemPattern();
    //     ortApi.AddFreeDimensionOverrideByName(options, "batch_size", 1);
    //     ortDmlApi->SessionOptionsAppendExecutionProvider_DML(options, /*device index*/ 0);

    //     std::cout << "DirectML provider enabled successfully" << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "Warning: Failed to enable DirectML, falling back to CPU: " << e.what() << std::endl;
    // }

#if defined(_WIN32) || defined(_WIN64)
    encoder_session_ = std::make_unique<Ort::Session>(env_, encoder_model_path.c_str(), options);
    decoder_session_ = std::make_unique<Ort::Session>(env_, decoder_model_path.c_str(), options);
#else
    encoder_session_ = std::make_unique<Ort::Session>(env_, encoder_model_path.c_str(), options);
    decoder_session_ = std::make_unique<Ort::Session>(env_, decoder_model_path.c_str(), options);
#endif
}

std::vector<int64_t> OpusModel::infer(std::vector<int64_t> &input_ids, int max_length) const {
    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    auto enc_outputs = runEncoder(input_ids, attention_mask);

    std::vector<int64_t> dec_input_ids {PAD_TOKEN_ID};
    std::vector<int64_t> generated_tokens;

    while (max_length-- > 0) {
        auto logits = runDecoder(dec_input_ids, attention_mask, enc_outputs);

        const int64_t last_token_offset = (static_cast<int64_t>(dec_input_ids.size()) - 1) * VOCAB_SIZE;
        const float *start = logits.GetTensorMutableData<float>() + last_token_offset;
        auto it = std::max_element(start, start + VOCAB_SIZE);
        const int64_t token = it - start;

        if (token != EOS_TOKEN_ID) {
            generated_tokens.push_back(token);
            dec_input_ids.push_back(token);
        } else {
            break;
        }
    }

    return generated_tokens;
}

Ort::Value OpusModel::runEncoder(std::vector<int64_t> &input_ids, std::vector<int64_t> &attention_mask) const {
    const int64_t input_shape[2] = {1, static_cast<int64_t>(input_ids.size())};
    std::vector<Ort::Value> input_tensors;
    {
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            input_ids.data(),
            input_ids.size(),
            input_shape,
            2
        );
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            attention_mask.data(),
            attention_mask.size(),
            input_shape,
            2
        );
        input_tensors.emplace_back(std::move(input_ids_tensor));
        input_tensors.emplace_back(std::move(attention_mask_tensor));
    }
    std::vector<Ort::Value> output_tensors = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        enc_input_names_,
        input_tensors.data(),
        input_tensors.size(),
        enc_output_names_,
        1
    );
    return std::move(output_tensors.front());
}

Ort::Value OpusModel::runDecoder(std::vector<int64_t> &dec_input_ids, std::vector<int64_t> &enc_attention_mask, Ort::Value &enc_hidden_states) const {
    const int64_t dec_input_shape[2] = {1, static_cast<int64_t>(dec_input_ids.size())};
    const int64_t enc_attention_mask_shape[2] = {1, static_cast<int64_t>(enc_attention_mask.size())};
    const int64_t enc_hidden_states_shape[3] = {1, static_cast<int64_t>(enc_attention_mask.size()), ENCODER_HIDDEN_SIZE};
    std::vector<Ort::Value> input_tensors;
    {
        Ort::Value dec_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            dec_input_ids.data(),
            dec_input_ids.size(),
            dec_input_shape,
            2
        );
        Ort::Value enc_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            enc_attention_mask.data(),
            enc_attention_mask.size(),
            enc_attention_mask_shape,
            2
        );
        Ort::Value enc_hidden_states_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            enc_hidden_states.GetTensorMutableData<float>(),
            enc_attention_mask.size() * static_cast<size_t>(ENCODER_HIDDEN_SIZE),
            enc_hidden_states_shape,
            3
        );
        input_tensors.emplace_back(std::move(enc_attention_mask_tensor));
        input_tensors.emplace_back(std::move(dec_input_ids_tensor));
        input_tensors.emplace_back(std::move(enc_hidden_states_tensor));
    }
    std::vector<Ort::Value> output_tensors = decoder_session_->Run(
        Ort::RunOptions{nullptr},
        dec_input_names_,
        input_tensors.data(),
        input_tensors.size(),
        dec_output_names_,
        1
    );
    return std::move(output_tensors.front());
}
