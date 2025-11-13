#include "LoginController.h"
#include <drogon/HttpResponse.h>
#include <drogon/HttpTypes.h>
#include <string>

void LoginController::asyncHandleHttpRequest(const HttpRequestPtr& req, std::function<void (const HttpResponsePtr &)> &&callback)
{
    // write your application logic here
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k200OK);
    resp->setContentTypeCode(drogon::CT_TEXT_HTML);
    std::string html = "<html><body><h1>Username: ";
    html.append(req->getParameter("username"));
    html.append("</h1><h1>Password: ");
    html.append(req->getParameter("password"));
    html.append("</h1></body></html>");
    resp->setBody(html);
    callback(resp);
}
