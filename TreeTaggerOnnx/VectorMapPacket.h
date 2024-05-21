#pragma once
#include "pch.h"
#include "json.hpp"
#include "VectorMap.h"


class VectorMapPacket {
    using json = nlohmann::json;
    using VectorMapParams = VectorMap::VectorMapParams;

private:

public:

    std::string assemblyPath;
    std::string projectPath;
    std::string vectorsPath;
    bool joinMaps;
    bool pointMaps;
    std::array<double, 4> extent;
    std::vector<VectorMapParams> vectorMapParams;

    VectorMapPacket(json& j) {

        j.at("projectPath").get_to(projectPath);
        j.at("vectorsPath").get_to(vectorsPath);
        j.at("extent").get_to(extent);
        j.at("joinMaps").get_to(joinMaps);
        j.at("pointMaps").get_to(pointMaps);

        vectorMapParams.reserve(j.at("gridSizes").size());

        for (int i = 0; i < (int)j.at("gridSizes").size(); i++) {
            vectorMapParams.emplace_back(VectorMapParams{ j.at("gridSizes").at(i), j.at("minTrees").at(i), j.at("averagingMethods").at(i) });
        }

    }

};