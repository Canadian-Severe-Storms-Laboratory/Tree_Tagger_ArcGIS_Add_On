#pragma once
#include "pch.h"
#include <chrono>
#include <ctime>
#include <string>

using namespace std::chrono;

// Simple timer class
class Timer {

private:
	steady_clock::time_point startTime;

public:
	
	Timer() {
		startTime = std::chrono::steady_clock::now();
	}

	void reset() {
		startTime = std::chrono::steady_clock::now();
	}

	std::string durationHMS(const bool reset = false) {
		return toHMS(duration(reset));	
	}

	static std::string toHMS(long long dt) {
		auto hours = std::to_string(dt / 3600);
		auto minutes = std::to_string(dt % 3600 / 60);
		auto seconds = std::to_string(dt % 60);

		if (hours.length() < 2) {
			hours.insert(0, 2 - hours.length(), '0');
		}

		minutes.insert(0, 2 - minutes.length(), '0');
		seconds.insert(0, 2 - seconds.length(), '0');

		return hours + ":" + minutes + ":" + seconds;
	}

	long long duration(const bool reset = false) {

		steady_clock::time_point currentTime = std::chrono::steady_clock::now();

		const auto d = duration_cast<seconds>(currentTime - startTime).count();

		if (reset) {
			startTime = currentTime;
		}

		return d;
	}
	
	long long durationMs(const bool reset = false) {
		steady_clock::time_point currentTime = std::chrono::steady_clock::now();

		const auto d = duration_cast<milliseconds>(currentTime - startTime).count();

		if (reset) {	
			startTime = currentTime;

		}

		return d;
	}
	
	long long durationUs(const bool reset = false) {
		steady_clock::time_point currentTime = std::chrono::steady_clock::now();

		const auto d = duration_cast<microseconds>(currentTime - startTime).count();

		if (reset) {
			startTime = currentTime;
		}

		return d;
	}
};
