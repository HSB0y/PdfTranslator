#include "TranslateController.h"
#include <iostream>
#include "OpusModel.hpp"
#include "SentencePieceTokenizer.hpp"
#include <optional>

extern std::optional<SentencePieceTokenizer> tokenizer;
extern std::optional<OpusModel> model;
extern int max_length;

void TranslateController::asyncHandleHttpRequest(const HttpRequestPtr& req, std::function<void (const HttpResponsePtr &)> &&callback)
{
    // Parse JSON request body
    auto json = req->getJsonObject();
    
    if (json && json->isMember("text"))
    {
        std::string selectedText = (*json)["text"].asString();
        
        // Print the selected text to console
        std::string decoded_text;
        if (tokenizer.has_value() && model.has_value())
        {
            auto encoded_tokens = tokenizer->encode(selectedText);
            encoded_tokens.push_back(OpusModel::EOS_TOKEN_ID);
            auto output_ids = model->infer(encoded_tokens, max_length);
            decoded_text = tokenizer->decode(output_ids);
        }
        
        // Send success response
        auto resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k200OK);
        resp->setContentTypeCode(CT_APPLICATION_JSON);
        
        Json::Value responseJson;
        responseJson["status"] = "success";
        responseJson["translated_text"] = decoded_text;
        resp->setBody(responseJson.toStyledString());
        
        callback(resp);
    }
    else
    {
        // Send error response if no text field
        auto resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k400BadRequest);
        resp->setContentTypeCode(CT_APPLICATION_JSON);
        
        Json::Value responseJson;
        responseJson["status"] = "error";
        responseJson["translated_text"] = "";
        resp->setBody(responseJson.toStyledString());
        
        callback(resp);
    }
}
