#include "growt_http_client.h"
#include <curl/curl.h>
#include <stdexcept>

namespace growt {

static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* output) {
    output->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

HttpClient::HttpClient(const std::string& api_url) : api_url_(api_url) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

HttpClient::~HttpClient() {
    curl_global_cleanup();
}

AuditResult HttpClient::audit(const std::string& json_payload) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("Failed to init curl");

    std::string response_body;
    std::string url = api_url_ + "/api/v1/audit";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("Growt API call failed: ") + curl_easy_strerror(res));
    }

    // TODO: Parse JSON response_body into AuditResult
    // Using a JSON library like nlohmann/json
    AuditResult result{};
    result.safe_to_deploy = false;
    result.diagnosis = "PARSE_NOT_IMPLEMENTED";
    result.report = response_body;
    return result;
}

}  // namespace growt
