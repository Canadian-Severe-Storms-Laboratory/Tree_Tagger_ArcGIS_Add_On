#pragma once
#include "json.hpp"

class LineJoinParser {
	using json = nlohmann::json;

private:


public:

    struct LineJoinParams {
        double angleThreshold;
        double directMergeAngleThreshold;
        double directMergeThreshold;
        double distThreshold;
        double minLineLength;
        double maxLineLength;
    };

    LineJoinParams lineJoinParams;

	LineJoinParser(json& j) {

        j.at("angleThreshold").get_to(lineJoinParams.angleThreshold);
        j.at("directMergeAngleThreshold").get_to(lineJoinParams.directMergeAngleThreshold);
        j.at("directMergeThreshold").get_to(lineJoinParams.directMergeThreshold);
        j.at("distThreshold").get_to(lineJoinParams.distThreshold);
        j.at("minLineLength").get_to(lineJoinParams.minLineLength);
        j.at("maxLineLength").get_to(lineJoinParams.maxLineLength);

	}


};