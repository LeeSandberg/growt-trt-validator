#pragma once
#include <string>

namespace growt {

struct AuditResult {
    bool safe_to_deploy;
    std::string diagnosis;
    float transfer_oracle;
    float coverage_pct;
    std::string report;
};

class HttpClient {
public:
    explicit HttpClient(const std::string& api_url);
    ~HttpClient();

    AuditResult audit(const std::string& json_payload);

private:
    std::string api_url_;
};

}  // namespace growt
