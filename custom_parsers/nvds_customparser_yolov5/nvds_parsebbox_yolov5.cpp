/**
 * @file nvds_parsebbox_yolov5.cpp
 *
 * @brief Custom parser and engine generator for yolov5
 *
 * @ingroup yolov5
 *
 * @author Akash James
 * Contact: akash@iwizardsolutions.com
 *
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"
#include "common.hpp"
#include "half.h"

using namespace nvinfer1;

#define kNMS_THRESH 0.45

struct NetworkInfo {
    std::string networkType;
	std::string modelFilePath;
    std::string int8Cache;
    std::string deviceType;
    std::string inputBlobName;
    std::string outputBlobName;
    int networkmode;
	int batchsize;
	int workspacesize;
    int classes;
    int channels;
    int height;
    int width;
};

NetworkInfo networkInfo;

enum MODEL_TYPE {DEFAULT = 0, P6 = 1};

std::unordered_map<std::string, float>
        mPerTensorDynamicRangeMap;
void setLayerPrecision(INetworkDefinition* network) {
    for (int i = 0; i < network->getNbLayers(); ++i) {
        auto layer = network->getLayer(i);

        // Don't set the precision on non-computation layers as they don't support
        // int8.
        if (layer->getType() != LayerType::kCONSTANT && layer->getType() != LayerType::kCONCATENATION
            && layer->getType() != LayerType::kSHAPE) {
            // set computation precision of the layer
            layer->setPrecision(nvinfer1::DataType::kINT8);
        }

        for (int j = 0; j < layer->getNbOutputs(); ++j) {
            std::string tensorName = layer->getOutput(j)->getName();
            // set output type of execution tensors and not shape tensors.
            if (layer->getOutput(j)->isExecutionTensor()) {
                layer->setOutputType(j, nvinfer1::DataType::kINT8);
            }
        }
    }
}

bool readPerTensorDynamicRangeValues() {
    std::ifstream iDynamicRangeStream(networkInfo.int8Cache);
    if (!iDynamicRangeStream) {
        std::cerr << "[Error] Could not find per tensor scales file: " << networkInfo.int8Cache << std::endl;
        return false;
    }

    std::string line;
    char delim = ':';
    while (std::getline(iDynamicRangeStream, line)) {
        std::istringstream iline(line);
        std::string token;
        std::getline(iline, token, delim);
        std::string tensorName = token;
        std::getline(iline, token, delim);
        try {
            float dynamicRange = std::stof(token);
            mPerTensorDynamicRangeMap[tensorName] = dynamicRange;
        }
        catch (...) {}
    }
    return true;
}

bool setDynamicRange(INetworkDefinition* network) {
    // populate per tensor dynamic range
    if (!readPerTensorDynamicRangeValues()) {
        return false;
    }

    // set dynamic range for network input tensors
    for (int i = 0; i < network->getNbInputs(); ++i) {
        std::string tName = network->getInput(i)->getName();
        if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end()) {
            if (!network->getInput(i)->setDynamicRange(
                    -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName))) {
                return false;
            }
        }
        else {
            std::cout << "[Warning] Missing dynamic range for tensor: " << tName << std::endl;
        }
    }

    // set dynamic range for layer output tensors
    for (int i = 0; i < network->getNbLayers(); ++i) {
        auto lyr = network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j) {
            std::string tName = lyr->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end()) {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                if (!lyr->getOutput(j)->setDynamicRange(
                        -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName))) {
                    return false;
                }
            }
            else if (lyr->getType() == LayerType::kCONSTANT) {
                IConstantLayer* cLyr = static_cast<IConstantLayer*>(lyr);
                std::cout << "[Warning] Computing missing dynamic range for tensor, " << tName << ", from weights."
                                    << std::endl;

                auto wts = cLyr->getWeights();
                double max = std::numeric_limits<double>::min();
                for (int64_t wb = 0, we = wts.count; wb < we; ++wb) {
                    double val{};
                    switch (wts.type) {
                    case DataType::kFLOAT: val = static_cast<const float*>(wts.values)[wb]; break;
                    case DataType::kBOOL: val = static_cast<const bool*>(wts.values)[wb]; break;
                    case DataType::kINT8: val = static_cast<const int8_t*>(wts.values)[wb]; break;
                    case DataType::kHALF: val = static_cast<const half_float::half*>(wts.values)[wb]; break;
                    case DataType::kINT32: val = static_cast<const int32_t*>(wts.values)[wb]; break;
                    }
                    max = std::max(max, std::abs(val));
                }

                if (!lyr->getOutput(j)->setDynamicRange(-max, max)) {
                    return false;
                }
            }
            else {
                std::cout << "[Warning] Missing dynamic range for tensor: " << tName << std::endl;
            }
        }
    }
    return true;
}

IPluginV2Layer*
ConstructModel(INetworkDefinition* network, unsigned int maxBatchSize, float& gd, float& gw) {

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(networkInfo.inputBlobName.c_str(),
                                      DataType::kFLOAT,
                                      Dims3 { networkInfo.channels, networkInfo.height, networkInfo.width });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(networkInfo.modelFilePath);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0", networkInfo.height, networkInfo.width);
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (networkInfo.classes + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (networkInfo.classes + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (networkInfo.classes + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    IPluginV2Layer* yolo = addYoLoLayer(network, weightMap,
                            "model.24",
                            std::vector<IConvolutionLayer*>{det0, det1, det2},
                            networkInfo.classes, networkInfo.height, networkInfo.width);

    return yolo;
}

IPluginV2Layer*
ConstructModelP6(INetworkDefinition* network, unsigned int maxBatchSize, float& gd, float& gw) {

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(networkInfo.inputBlobName.c_str(),
                                        DataType::kFLOAT,
                                        Dims3 { networkInfo.channels, networkInfo.height, networkInfo.width });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(networkInfo.modelFilePath);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0", networkInfo.height, networkInfo.width);
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto spp10 = SPP(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), 3, 5, 7, "model.10");
    auto c3_11 = C3(network, weightMap, *spp10->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *c3_11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (networkInfo.classes + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (networkInfo.classes + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (networkInfo.classes + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (networkInfo.classes + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

    IPluginV2Layer* yolo = addYoLoLayer(network, weightMap,
                                        "model.33",
                                        std::vector<IConvolutionLayer*>{det0, det1, det2, det3},
                                        networkInfo.classes, networkInfo.height, networkInfo.width);

    return yolo;
}

ICudaEngine*
BuildEngine(nvinfer1::IBuilder* builder, int model_type, float& gd, float& gw) {
    ICudaEngine* cudaEngine = nullptr;

    auto config = builder->createBuilderConfig();
	// const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(0U);

    IPluginV2Layer* yolo;
    if(model_type == P6) {
        yolo = ConstructModelP6(network, networkInfo.batchsize, gd, gw);
    }
    else {
        yolo = ConstructModel(network, networkInfo.batchsize, gd, gw);
    }
    yolo->getOutput(0)->setName(networkInfo.outputBlobName.c_str());
    network->markOutput(*yolo->getOutput(0));

	builder->setMaxBatchSize(networkInfo.batchsize);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setMaxWorkspaceSize(networkInfo.workspacesize << 20);

    if(networkInfo.networkmode == NvDsInferNetworkMode_FP16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if(networkInfo.networkmode == NvDsInferNetworkMode_INT8) {
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(nullptr);

        // force layer to execute with required precision
        setLayerPrecision(network);

        // set INT8 Per Tensor Dynamic range
        if (!setDynamicRange(network)) {
            std::cerr << "[Error] Unable to set per tensor dynamic range." << std::endl;
            network->destroy();
            return cudaEngine;
        }
    }

	cudaEngine = builder->buildEngineWithConfig(*network, *config);

    network->destroy();

    return cudaEngine;
}

extern "C"
bool BuildCustomYOLOv5sEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool BuildCustomYOLOv5lEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool BuildCustomYOLOv5mEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool BuildCustomYOLOv5xEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool BuildCustomYOLOv5sP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool BuildCustomYOLOv5lP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool BuildCustomYOLOv5mP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool BuildCustomYOLOv5xP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferParseCustomYOLOv5(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	NvDsInferNetworkInfo  const &networkInfo,
	NvDsInferParseDetectionParams const &detectionParams,
	std::vector<NvDsInferObjectDetectionInfo> &objectList);


extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList) {
    const float kCONF_THRESH = detectionParams.perClassThreshold[0];
    std::vector<Detection> res;

    nms(res, (float*)(outputLayersInfo[0].buffer), kCONF_THRESH, kNMS_THRESH);

    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;

	    oinfo.classId = r.class_id;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]-r.bbox[2]*0.5f);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]-r.bbox[3]*0.5f);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
	    objectList.push_back(oinfo);
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5sEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.int8Cache       = (initParams->int8CalibrationFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 0.33, gw = 0.50;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, DEFAULT, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5mEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 0.67, gw = 0.75;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, DEFAULT, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5lEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 1.0, gw = 1.0;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, DEFAULT, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5xEngine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 1.33, gw = 1.25;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, DEFAULT, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5sP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.int8Cache       = (initParams->int8CalibrationFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 0.33, gw = 0.50;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, P6, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5mP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 0.67, gw = 0.75;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, P6, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5lP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 1.0, gw = 1.0;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, P6, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool BuildCustomYOLOv5xP6Engine(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine) {

	networkInfo.networkType     = "yolov5";
	networkInfo.modelFilePath   = (initParams->modelFilePath);
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";
    networkInfo.outputBlobName  = "prob";
    networkInfo.networkmode     = (initParams->networkMode);
	networkInfo.batchsize       = (initParams->maxBatchSize);
    networkInfo.workspacesize   = (initParams->workspaceSize);
    networkInfo.classes         = (initParams->numDetectedClasses);

    // Input Dims
    NvDsInferDimsCHW inferdims  = (initParams->inferInputDims);
    networkInfo.channels        = inferdims.c;
    networkInfo.height          = inferdims.h;
    networkInfo.width           = inferdims.w;

    float gd = 1.33, gw = 1.25;

    assert(builder != nullptr);

    cudaEngine = BuildEngine(builder, P6, gd, gw);

    if (cudaEngine == nullptr) {
        std::cerr << "[Error] Failed to build cuda engine on "
                  << networkInfo.modelFilePath << std::endl;
        return false;
    }
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
