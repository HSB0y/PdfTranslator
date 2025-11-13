#include <algorithm>
#include <drogon/HttpAppFramework.h>
#include <drogon/drogon.h>
#include <fstream>
#include <optional>
#include <iostream>
#include "OpusModel.hpp"
#include "SentencePieceTokenizer.hpp"
#include <webview/types.h>
#include <webview/webview.h>
#include "nlohmann/json.hpp"

std::optional<SentencePieceTokenizer> tokenizer;
std::optional<OpusModel> model;
int max_length = 128;

#if defined (_WIN32) || defined (_WIN64)
#include <windows.h>
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
#else
int main() {
#endif
    std::thread llm_thread([]() {
        // Set default values
        int num_threads = 4;
        GraphOptimizationLevel optimization_level = ORT_ENABLE_BASIC;
        
        try {
            // Try to read and parse the config file
            std::ifstream config_file("deploy/llm_config.json");
            if (config_file.is_open()) {
                nlohmann::json config;
                config_file >> config;
                
                // Get number of threads, use default if not found or invalid
                if (config.contains("number_thread") && config["number_thread"].is_number()) {
                    num_threads = config["number_thread"].get<int>();
                    num_threads = std::clamp(num_threads, 1, 16);
                }

                if (config.contains("max_length") && config["max_length"].is_number()) {
                    max_length = config["max_length"].get<int>();
                    max_length = std::clamp(max_length, 128, 4096);
                }
                
                // Get optimization level, use default if not found or invalid
                if (config.contains("optimization_level") && config["optimization_level"].is_string()) {
                    std::string level = config["optimization_level"].get<std::string>();
                    if (level == "ALL") {
                        optimization_level = ORT_ENABLE_ALL;
                    } else if (level == "EXTENDED") {
                        optimization_level = ORT_ENABLE_EXTENDED;
                    } else if (level == "DISABLE") {
                        optimization_level = ORT_DISABLE_ALL;
                    } else if (level == "BASIC") {
                        optimization_level = ORT_ENABLE_BASIC;
                    } else {
                        optimization_level = ORT_DISABLE_ALL;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading config file: " << e.what() << std::endl;
            // Use default values if there's an error
        }
        
        tokenizer.emplace("./llm/source.spm", "./llm/target.spm", "./llm/vocab.txt");
#if defined (_WIN32) || defined (_WIN64)
        model.emplace(L"./llm/encoder_model.onnx", L"./llm/decoder_model.onnx", num_threads, optimization_level);
#else
        model.emplace("./llm/encoder_model.onnx", "./llm/decoder_model.onnx", num_threads, optimization_level);
#endif
    });
    
    std::thread drogon_thread([]() {
        //Set HTTP listener address and port
        drogon::app().addListener("http://localhost", 5555);
        //Load config file
        //drogon::app().loadConfigFile("../config.json");
        drogon::app().loadConfigFile("./config.yaml");
        //Run HTTP framework,the method will block in the internal event loop
        drogon::app().run();
    });

    webview::webview window(false, nullptr);
    window.set_title("PDF Translator");
    window.set_size(800, 600, WEBVIEW_HINT_NONE);
    window.navigate("http://localhost:5555");

    window.run();
    drogon::app().quit();
    llm_thread.join();
    drogon_thread.join();
    return 0;
}
