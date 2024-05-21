#pragma once
#include "json.hpp"

class ImageParser {
	using json = nlohmann::json;

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

    std::array<double, 4> extent;
    std::vector<ImageFile> imageFiles;
    std::vector<std::array<double, 2>> polygonPts;

	ImageParser(json& j) {

        imageFiles.reserve(j.at("imagePaths").size());

        for (int i = 0; i < (int)j.at("imagePaths").size(); i++) {

            imageFiles.emplace_back(ImageFile{ j.at("imagePaths").at(i), j.at("imageCoords").at(i), 
                                               j.at("imageSizes").at(i), j.at("imageScales").at(i) });
        }

        polygonPts.reserve(j.at("polygonPoints").size() / 2);

        for (int i = 0; i < (int)j.at("polygonPoints").size() - 1; i += 2) {
            polygonPts.push_back({ (double)j.at("polygonPoints").at(i), (double)j.at("polygonPoints").at(i + 1) });
        }

        computeImageryExtent();
	}

};