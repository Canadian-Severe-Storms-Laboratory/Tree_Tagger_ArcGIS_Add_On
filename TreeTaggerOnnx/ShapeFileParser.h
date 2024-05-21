#pragma once
#include "json.hpp"

class ShapeFileParser {
	using json = nlohmann::json;

private:


public:

	std::string shapeFilePath;
	std::array<double, 4> extent;

	ShapeFileParser(json& j) {

		j.at("shapeFilePath").get_to(shapeFilePath);
		j.at("extent").get_to(extent);

	}

};

