#include "SentencePieceTokenizer.hpp"
#include <stdexcept>
#include <fstream>

SentencePieceTokenizer::SentencePieceTokenizer(const std::string &en_model_path, const std::string &zh_model_path, const std::string &vocab_file_path) {
    if (const auto status = en_processor_.Load(en_model_path); !status.ok()) {
        throw std::runtime_error("Failed to load En SentencePiece model: " + status.ToString());
    }
    if (const auto status = zh_processor_.Load(zh_model_path); !status.ok()) {
        throw std::runtime_error("Failed to load Zh SentencePiece model: " + status.ToString());
    }
    std::ifstream vocab_file(vocab_file_path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_file_path);
    }
    int64_t index = 0;
    std::string line;
    while (getline(vocab_file, line, '\n')) {
        bimap_.insert(line, index++);
    }
}

std::vector<int64_t> SentencePieceTokenizer::encode(const std::string &sentence) {
    std::vector<std::string> pieces;
    if (const auto status = en_processor_.Encode(sentence, &pieces); !status.ok()) {
        throw std::runtime_error("Failed to encode sentence: " + status.ToString());
    }
    std::vector<int64_t> tokens;
    tokens.reserve(pieces.size() + 1);
    for (const auto &piece : pieces) {
        if (const auto it = bimap_[piece]; it != nullptr) {
            tokens.emplace_back(it->second);
        } else {
            tokens.emplace_back(1);
        }
    }
    return tokens;
}

std::string SentencePieceTokenizer::decode(const std::vector<int64_t> &tokens) {
    std::vector<std::string> pieces;
    pieces.reserve(tokens.size());
    for (const auto &token : tokens) {
        if (const auto it = bimap_[token]; it != nullptr) {
            pieces.emplace_back(it->first);
        }
    }
    std::string text;
    if (const auto status = zh_processor_.Decode(pieces, &text); !status.ok()) {
        throw std::runtime_error("Failed to decode sentence: " + status.ToString());
    }
    return text;
}
