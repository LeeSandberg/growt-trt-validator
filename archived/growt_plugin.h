#pragma once

#include "NvInferPlugin.h"
#include "growt_http_client.h"
#include <string>
#include <memory>

namespace growt {

class GrowtPlugin : public nvinfer1::IPluginV3,
                    public nvinfer1::IPluginV3OneCore,
                    public nvinfer1::IPluginV3OneBuild,
                    public nvinfer1::IPluginV3OneRuntime {
public:
    GrowtPlugin(const std::string& api_url = "https://api.transferoracle.ai");
    ~GrowtPlugin() override = default;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    // IPluginV3OneRuntime
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

private:
    std::string api_url_;
    bool audited_ = false;
};

class GrowtPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    GrowtPluginCreator();
    nvinfer1::IPluginV3* createPlugin(char const* name,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection field_collection_;
    std::vector<nvinfer1::PluginField> fields_;
};

}  // namespace growt
