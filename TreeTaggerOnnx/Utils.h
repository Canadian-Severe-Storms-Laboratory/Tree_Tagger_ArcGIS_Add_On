#pragma once
#include "pch.h"
#include <ctime>
#include <iostream>
#include <vector>
#include <stacktrace>
#include <ctime>
#include "Console.h"
#include "ImageParser.h"
#include "LineJoinParser.h"
#include "Timer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace Utils {

	using IMGP = ImageParser;
	using LJP = LineJoinParser;
	using namespace Console;

	bool* cancelFlag = nullptr;

	Timer timer;

	void printState(const std::string& state) {
		//std::cout << state << "\n\n" << std::flush;

		if (consoleCallback != nullptr) {
			consoleCallback((state + "\n\n").c_str());
		}
	}

	void printStackTrace() {
		auto trace = std::stacktrace::current();

		printState("Stack trace:");

		for (int i = 2; i < trace.size(); i++) {
			const auto& entry = trace[i];

			// Ignore functions outside of this project
			if (entry.source_file().empty() || entry.source_line() == 0) continue;

			if (!entry.source_file().contains(".cpp") && !entry.source_file().contains(".h")) continue;

			std::string traceLine = "Description: " + entry.description() + "\nfile: " + entry.source_file() + 
									"\nline: " + std::to_string(entry.source_line()) + "\n------------------------------------";

			printState(traceLine);

			/*std::cout << "Description: " << entry.description() << std::endl;
			std::cout << "file: " << entry.source_file() << std::endl;
			std::cout << "line: " << entry.source_line() << std::endl;
			std::cout << "------------------------------------" << std::endl;*/
		}
	}

	cv::Mat imread(const std::string& path, const cv::ImreadModes mode=cv::IMREAD_COLOR) {
		try {
			cv::Mat im = cv::imread(path, mode);

			if (im.empty()) throw std::runtime_error("\nFailed to read image: " + path);

			return im;
		}
		catch (const std::exception& ex) {
			//std::cout << ex.what() << "\n\n";

			Utils::consoleCallback(ex.what());

			printStackTrace();

			exit(1);
		}
	}

	void createDirectory(const std::string& path) {
		if (!std::filesystem::is_directory(path) || !std::filesystem::exists(path)) { // Check if folder exists
			std::filesystem::create_directory(path); // create folder
		}
	}

	std::string dateTime() {
		time_t curr_time;

		(void)time(&curr_time);
		const tm* curr_tm = localtime(&curr_time);

		char time_string[18];

		(void)strftime(time_string, 18, "%y_%m_%d_%H_%M_%S", curr_tm);

		return time_string;
	}

	void checkCancelled() {
		if (cancelFlag != nullptr && *cancelFlag) throw std::runtime_error("Cancelled");
	}

	std::string setupEnv(const std::string& projectPath, const bool fullSetup = true) {
		std::string resultsPath = projectPath + "/TreeTagger/Results_" + dateTime();

		createDirectory(projectPath + "/TreeTagger");
		createDirectory(resultsPath);

		const int p = omp_get_num_procs();
		omp_set_num_threads(p);

		timer.reset();

		if (fullSetup) {
			(void)_putenv("CUDA_MODULE_LOADING=LAZY");

			const auto title = "   ___ ___ ___ _        _       _         _____              _____                       \n"
				"  / __/ __/ __| |      /_\\ _  _| |_ ___  |_   _| _ ___ ___  |_   _|_ _ __ _ __ _ ___ _ _ \n"
				" | (__\\__ \\__ \\ |__   / _ \\ || |  _/ _ \\   | || '_/ -_) -_)   | |/ _` / _` / _` / -_) '_|\n"
				"  \\___|___/___/____| /_/ \\_\\_,_|\\__\\___/   |_||_| \\___\\___|   |_|\\__,_\\__, \\__, \\___|_|  \n"
				"                                                                      |___/|___/         \n"
				"-----------------------------------------------------------------------------------------";

			printState(title);
		}

		return resultsPath;
	}


	void closeEnv() {

		const auto finishedString = "-----------------------------------------------------------------------------------------\n"
			"Done. Completed in: " + timer.durationHMS() + ".\n";

		printState(finishedString);
	}
};