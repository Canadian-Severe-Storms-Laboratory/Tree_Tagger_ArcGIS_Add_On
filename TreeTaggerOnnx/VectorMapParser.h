#pragma once
#include "json.hpp"
#include "VectorMap.h"

class VectorMapParser {
	using json = nlohmann::json;
	using VectorMapParams = VectorMap::VectorMapParams;

private:


public:

	bool joinMaps;
	bool pointMaps;
	std::vector<VectorMapParams> vectorMapParams;

	VectorMapParser(json& j) {

		j.at("joinMaps").get_to(joinMaps);
		j.at("pointMaps").get_to(pointMaps);

		vectorMapParams.reserve(j.at("gridSizes").size());

		for (int i = 0; i < (int)j.at("gridSizes").size(); i++) {
			vectorMapParams.emplace_back(VectorMapParams{ j.at("gridSizes").at(i), j.at("minTrees").at(i), j.at("averagingMethods").at(i), 10, 0.5, 0.5 });
		}

	}

};
