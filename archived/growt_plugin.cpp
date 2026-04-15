#include "growt_plugin.h"
#include <cstring>
#include <cuda_runtime.h>

namespace growt {

GrowtPlugin::GrowtPlugin(const std::string& api_url) : api_url_(api_url) {}

nvinfer1::IPluginCapability* GrowtPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

// Passthrough: same number of outputs as inputs
int32_t GrowtPlugin::getNbOutputs() const noexcept { return 1; }

int32_t GrowtPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept {
    outputTypes[0] = inputTypes[0];  // Passthrough type
    return 0;
}

int32_t GrowtPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    outputs[0] = inputs[0];  // Passthrough shape
    return 0;
}

bool GrowtPlugin::supportsFormatCombination(int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept {
    return true;  // Accept all formats (passthrough)
}

int32_t GrowtPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    return 0;
}

int32_t GrowtPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    return 0;
}

int32_t GrowtPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    try {
        // Passthrough: copy input to output
        size_t size = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
            size *= inputDesc[0].dims.d[i];
        size *= sizeof(float);  // TODO: handle other types

        cudaMemcpyAsync(outputs[0], inputs[0], size, cudaMemcpyDeviceToDevice, stream);

        // TODO: On first call, trigger async Growt audit
        // if (!audited_) { ... }

        return 0;
    } catch (...) {
        return -1;
    }
}

// Creator
GrowtPluginCreator::GrowtPluginCreator() {
    fields_.push_back({"api_url", nullptr, nvinfer1::PluginFieldType::kCHAR, 0});
    field_collection_.nbFields = static_cast<int32_t>(fields_.size());
    field_collection_.fields = fields_.data();
}

nvinfer1::IPluginV3* GrowtPluginCreator::createPlugin(char const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    std::string api_url = "https://api.transferoracle.ai";
    for (int i = 0; i < fc->nbFields; ++i) {
        if (std::string(fc->fields[i].name) == "api_url") {
            api_url = static_cast<const char*>(fc->fields[i].data);
        }
    }
    return new GrowtPlugin(api_url);
}

nvinfer1::PluginFieldCollection const* GrowtPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}
char const* GrowtPluginCreator::getPluginName() const noexcept { return "GrowPlugin"; }
char const* GrowtPluginCreator::getPluginVersion() const noexcept { return "1"; }
char const* GrowtPluginCreator::getPluginNamespace() const noexcept { return "growt"; }

}  // namespace growt

REGISTER_TENSORRT_PLUGIN(growt::GrowtPluginCreator);
