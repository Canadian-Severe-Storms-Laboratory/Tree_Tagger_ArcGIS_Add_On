#pragma once
#include "pch.h"
#include "json.hpp"
#include "VectorMap.h"


class TreeTagPacket {
    using json = nlohmann::json;
    using VectorMapParams = VectorMap::VectorMapParams;

private:
    void computeImageryExtent() {
        extent = { 1e38f, -1e38f, -1e38f, 1e38f }; //minX, maxY, maxX, minY

        for (const auto& [path, coords, size, scale] : imageFiles) {
            extent[0] = coords[0] < extent[0] ? coords[0] : extent[0];
            extent[1] = extent[1] < coords[1] ? coords[1] : extent[1];

            extent[2] = coords[0] + size[0] * scale > extent[2] ? coords[0] + size[0] * scale : extent[2];
            extent[3] = extent[3] > coords[1] - size[1] * scale ? coords[1] - size[1] * scale : extent[3];

        }
    }

public:

    struct ImageFile {
        std::string path;
        std::array<double, 2> coords;
        std::array<int, 2> size;
        double scale;

        std::array<double, 4>  getExtent() const {
            return { coords[0], coords[1], coords[0] + size[0] * scale, coords[1] - size[1] * scale };
        }
    };

    struct LineJoinParams {
        float angleThreshold;
        float directMergeAngleThreshold;
        float directMergeThreshold;
        float distThreshold;
        float minLineLength;
        float maxLineLength;
    };

    std::string assemblyPath;
    std::string projectPath;
    std::vector<std::array<double, 2>> polygonPts;
    bool fast;
    bool joinMaps;
    bool pointMaps;
    std::array<double, 4> extent;
    std::vector<ImageFile> imageFiles;
    std::vector<VectorMapParams> vectorMapParams;
    LineJoinParams lineJoinParams;

    TreeTagPacket(json& j) {

        j.at("assemblyPath").get_to(assemblyPath);
        j.at("projectPath").get_to(projectPath);
        //j.at("polygonPoints").get_to(polygonPts);
        j.at("fast").get_to(fast);
        j.at("joinMaps").get_to(joinMaps);
        j.at("pointMaps").get_to(pointMaps);

        j.at("angleThreshold").get_to(lineJoinParams.angleThreshold);
        j.at("directMergeAngleThreshold").get_to(lineJoinParams.directMergeAngleThreshold);
        j.at("directMergeThreshold").get_to(lineJoinParams.directMergeThreshold);
        j.at("distThreshold").get_to(lineJoinParams.distThreshold);
        j.at("minLineLength").get_to(lineJoinParams.minLineLength);
        j.at("maxLineLength").get_to(lineJoinParams.maxLineLength);


        polygonPts.reserve(j.at("polygonPoints").size()/2);

        for (int i = 0; i < (int)j.at("polygonPoints").size() - 1; i+=2) {
            polygonPts.push_back({ (double)j.at("polygonPoints").at(i), (double)j.at("polygonPoints").at(i + 1) });
        }

        imageFiles.reserve(j.at("imagePaths").size());

        for (int i = 0; i < (int)j.at("imagePaths").size(); i++) {
            imageFiles.emplace_back(ImageFile{ j.at("imagePaths").at(i), j.at("imageCoords").at(i), j.at("imageSizes").at(i), j.at("imageScales").at(i) });
        }

        vectorMapParams.reserve(j.at("gridSizes").size());

        for (int i = 0; i < (int)j.at("gridSizes").size(); i++) {
            vectorMapParams.emplace_back(VectorMapParams{ j.at("gridSizes").at(i), j.at("minTrees").at(i), j.at("averagingMethods").at(i) });
        }

        computeImageryExtent();
    }

};