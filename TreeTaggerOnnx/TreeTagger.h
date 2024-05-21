#pragma once
#include "pch.h"

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstdlib>
#include <filesystem>
#include <ctime>
#include <stacktrace>
#include <direct.h>

#include "JoinLines.h"
#include "TreeSegModel.h"
#include "TreeDirectionModel.h"
#include "ShapeFile.h"
#include "shapefil.h"
#include "JsonMarshaledPacket.h"
#include "ProjectParser.h"
#include "ImageParser.h"
#include "LineJoinParser.h"
#include "VectorMapParser.h"
#include "ShapeFileParser.h"
#include "OutlineParser.h"
#include "ProgressBar.h"
#include "Utils.h"
#include "Console.h"
#include "Coords.h"
#include "VectorMap.h"

using json = nlohmann::json;
using JMP = JsonMarshaledPacket;
using VM = VectorMap;

using namespace Utils;
using namespace Coords;
using namespace JoinLines;

#ifdef __cplusplus
extern "C" {
#endif


#define TREE_TAGGER_API __declspec(dllimport)

	int TREE_TAGGER_API analyzeEvent(const char* packet, ConsoleCallback cb, bool* cf);
	int TREE_TAGGER_API createVectorMaps(const char* packet, ConsoleCallback cb, bool* cf);
	int TREE_TAGGER_API createOutlines(const char* packet, ConsoleCallback cb, bool* cf);

#ifdef __cplusplus
}
#endif

template<typename Func>
int apiWrapper(ConsoleCallback cb, bool* cf, Func func) {

	try {
		Utils::cancelFlag = cf;
		Console::consoleCallback = cb;

		func();

		closeEnv();
	}
	catch (Ort::Exception& e) {
		//std::cerr << "onnxruntime exception: " << e.what() << "\n Error Code: " << e.GetOrtErrorCode() << "\n\n";
		Console::consoleCallback("onnxruntime exception: ");
		Console::consoleCallback(e.what());
		Console::consoleCallback("Error Code: " + e.GetOrtErrorCode());
		Console::consoleCallback("\n\n");
		printStackTrace();
		return 1;
	}
	catch (std::string str) {
		//std::cerr << str << "\n\n";
		printState(str);
		printStackTrace();
		return 1;
	}
	catch (std::exception ex) {
		//std::cerr << ex.what() << "\n\n";
		printState(ex.what());
		printStackTrace();
		return 1;
	}

	return 0;
}