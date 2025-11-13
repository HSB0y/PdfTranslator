#pragma once

#include <drogon/HttpSimpleController.h>

using namespace drogon;

class LoginController : public drogon::HttpSimpleController<LoginController>
{
  public:
    void asyncHandleHttpRequest(const HttpRequestPtr& req, std::function<void (const HttpResponsePtr &)> &&callback) override;
    PATH_LIST_BEGIN
    // list path definitions here;
    PATH_ADD("/login", Get);
    PATH_LIST_END
};
