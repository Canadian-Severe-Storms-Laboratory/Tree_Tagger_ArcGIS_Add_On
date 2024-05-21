#pragma once
#include "json.hpp"
#include "VectorMap.h"

class OutlineParser {
	using json = nlohmann::json;
	using VectorMapParams = VectorMap::VectorMapParams;

private:


public:

	std::vector<VectorMapParams> vectorMapParams;

	OutlineParser(json& j) {

		vectorMapParams.reserve(j.at("gridSizes").size());

		for (int i = 0; i < (int)j.at("gridSizes").size(); i++) {
			vectorMapParams.emplace_back(VectorMapParams{ j.at("gridSizes").at(i), j.at("minTrees").at(i), VectorMap::AveragingMethod::None,
				                                          j.at("smoothSteps").at(i), j.at("smoothGrowths").at(i), j.at("smoothShrinks").at(i) });
		}

	}

};
