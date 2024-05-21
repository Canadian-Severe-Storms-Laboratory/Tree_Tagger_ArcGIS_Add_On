#pragma once
#include "json.hpp"

class ProjectParser {
	using json = nlohmann::json;

private:


public:

	std::string assemblyPath;
	std::string projectPath;
	bool fast;

	ProjectParser(json& j) {
		
		j.at("projectPath").get_to(projectPath);

		try {
			j.at("assemblyPath").get_to(assemblyPath);
		}
		catch (std::exception) {
			assemblyPath = "";
		}

		try {
			j.at("fast").get_to(fast);
		}
		catch (std::exception) {
			fast = false;
		}
	}


};