#pragma once
#include <sentencepiece_processor.h>
#include <vector>
#include <string>
#include "bimap.h"

class SentencePieceTokenizer {
public:
    SentencePieceTokenizer(const std::string &en_model_path, const std::string &zh_model_path, const std::string &vocab_file_path);

    std::vector<int64_t> encode(const std::string &sentence);
    std::string decode(const std::vector<int64_t> &tokens);

private:
    sentencepiece::SentencePieceProcessor en_processor_;
    sentencepiece::SentencePieceProcessor zh_processor_;
    maxy::bimap<std::string, int64_t> bimap_;
};
