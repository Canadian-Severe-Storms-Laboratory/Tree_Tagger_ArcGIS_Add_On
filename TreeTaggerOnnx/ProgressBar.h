#pragma once
#include "pch.h"
#include <iostream>
#include "Timer.h"
#include "Utils.h"

class ProgressBar {

private:
	Timer timer;
	long long timeSum;

public:

	int idx;
	int total;
	int barLength;

	// displays a progressbar for a loop
	// total = total iterations of loop
	// idx = current index of loop
	ProgressBar(const int total, const int barLength=50, const int idx = 0) {
		this->idx = idx;
		this->total = total;
		this->barLength = barLength;

		timeSum = 0;
		timer = Timer();

		drawProgressBar();
	}

	// increment progress bar
	void update(){
		update(1);
	}

	void update(const int inc) {
		idx += inc;
		redrawProgressBar();
	}

	void drawProgressBar() {

		Utils::consoleCallback(("[" + std::string(barLength, ' ') + "] " + std::to_string(idx) + "/" + std::to_string(total) + " ").c_str());
		//std::cout << "[" << std::string(barLength, ' ') << "] " << idx << "/" << total << std::flush;
	}

	void redrawProgressBar() {
		const float progress = (float)idx / (float)total;
		const int pos = (int)(progress * (float)barLength);

		const auto dt = timer.duration(true);

		timeSum += dt;

		const auto eta = (long long)(((double)timeSum / (double)idx) * (double)(total - idx));

		Utils::consoleCallback(("\r[" + std::string(pos, '#') + std::string(barLength - pos, ' ') + "] " + std::to_string(idx) + "/" + std::to_string(total) + "  ETA: " + timer.toHMS(eta) + " ").c_str());

		//std::cout << "\r[" << std::string(pos, (char)254) << std::string(barLength - pos, ' ') << "] " << idx << "/" << total;
		//std::cout << "  ETA: " << timer.toHMS(eta) << std::flush;
	}

	~ProgressBar() {
		Utils::consoleCallback(("\r" + std::string(barLength + 20 + std::to_string(total).size() * 2, ' ') + "\r").c_str());
		//std::cout << "\r" << std::string(barLength + 20 + std::to_string(total).size() * 2, ' ') << "\r" << std::flush;
	}
};
