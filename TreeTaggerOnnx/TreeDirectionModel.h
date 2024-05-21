#pragma once
#include "pch.h"
#include "OnnxModel.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

class TreeDirectionModel : public OnnxModel {
	using Line = std::array<double, 4>;
	using Point = std::array<double, 2>;

private:

	static constexpr int inputWidth = 256;
	static constexpr int inputHeight = 128;
	static constexpr int inputChannels = 3;
	static constexpr int inputSize = inputWidth * inputHeight * inputChannels;

	static constexpr int predicitionSize = 3;

	static void inline normalize(cv::Mat& image) {
		image.convertTo(image, CV_32FC3, 1.0 / 255.0);
	}

	static std::array<int, 4> getMinExtendedLineBox(const Line& line) {

		constexpr double halfLineLength = 384.0;
		constexpr double heightFactor = 1.5 * inputHeight;

		//line midpoint
		const Point mid = { 0.5 * (line[0] + line[2]), 0.5 * (line[1] + line[3]) };

		//line difference vector from midpoint to endpoint
		Point d = { line[0] - mid[0], line[1] - mid[1] };

		const double mag = hypot(d[0], d[1]);

		d = { d[0] / mag, d[1] / mag };

		//perpendicular vector to difference vector
		const Point p = { d[1], -d[0] };

		//extended endpoints
		const Point e1 = { mid[0] + halfLineLength * d[0], mid[1] + halfLineLength * d[1] };
		const Point e2 = { mid[0] - halfLineLength * d[0], mid[1] - halfLineLength * d[1] };

		std::array<Point, 4> box;

		//extended box around extended line
		box[0] = { e1[0] - heightFactor * p[0], e1[1] - heightFactor * p[1] };
		box[1] = { e1[0] + heightFactor * p[0], e1[1] + heightFactor * p[1] };
		box[2] = { e2[0] - heightFactor * p[0], e2[1] - heightFactor * p[1] };
		box[3] = { e2[0] + heightFactor * p[0], e2[1] + heightFactor * p[1] };

		return { (int)floor(min(min(min(box[0][1], box[1][1]), box[2][1]), box[3][1])),
				 (int)ceil(max(max(max(box[0][1], box[1][1]), box[2][1]), box[3][1])) + 1,
				 (int)floor(min(min(min(box[0][0], box[1][0]), box[2][0]), box[3][0])),
				 (int)ceil(max(max(max(box[0][0], box[1][0]), box[2][0]), box[3][0])) + 1 };
	}

	static cv::Mat cropTreeBox(const cv::Mat& image, const Line& line) {

		//find minimum bounding box around extended line
		const auto box = getMinExtendedLineBox(line);

		cv::Mat minBox = image(cv::Range(box[0], box[1]), cv::Range(box[2], box[3])).clone();

		//draw line on image
		cv::line(minBox, cv::Point((int)line[0] - box[2], (int)line[1] - box[0]), cv::Point((int)line[2] - box[2], (int)line[3] - box[0]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		//rotate min box to make the line horizontally aligned
		const float angle = cv::fastAtan2((float)(line[3] - line[1]), (float)(line[2] - line[0]));

		auto rotMat = cv::getRotationMatrix2D(cv::Point2d(0.5 * (minBox.size().width - 1) , 0.5 * (minBox.size().height - 1)), angle, 1.0);

		const cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), minBox.size(), angle).boundingRect2f();

		rotMat.at<double>(0, 2) += bbox.width / 2.0 - minBox.cols / 2.0;
		rotMat.at<double>(1, 2) += bbox.height / 2.0 - minBox.rows / 2.0;

		cv::warpAffine(minBox, minBox, rotMat, bbox.size());

		//crop out box around line
		const int midRow = (minBox.size().height - 1) / 2;
		const int midCol = (minBox.size().width - 1) / 2;

		minBox = minBox(cv::Range(midRow - 3 * (inputHeight / 2), midRow + 3 * (inputHeight / 2)), cv::Range(midCol - 3 * (inputWidth / 2), midCol + 3 * (inputWidth / 2)));

		//resize to required size
		cv::resize(minBox, minBox, cv::Size(inputWidth, inputHeight), 0.0, 0.0, cv::INTER_NEAREST);

		normalize(minBox);

		return minBox;
	}

	void loadTreeBoxBatch(const cv::Mat& image, const std::vector<Line>& lines, const int offset, const int currentBatchSize) {
		
		#pragma omp parallel for
		for (int i = 0; i < currentBatchSize; i++) {

			const auto treeImage = cropTreeBox(image, lines[offset + i]);

			memcpy(input.data() + i * inputSize, treeImage.data, inputSize * sizeof(float));
		}
	}

	static void printPredicition(const float* p) {
		const int idx = (p[0] > p[1]) ? ((p[0] > p[2]) ? 0 : 2) : ((p[1] > p[2]) ? 1 : 2);

		std::cout << "[ " << p[0] << ", " << p[1] << ", " << p[2] << " ] -> ";

		switch (idx) {
			case 0:
				std::cout << "LEFT\n";
				break;
			case 1:
				std::cout << "INC\n";
				break;
			case 2:
				std::cout << "RIGHT\n";
				break;
		}
	}

	static int pArgMax(const float* p) {
		return (p[0] > p[1]) ? ((p[0] > p[2]) ? 0 : 2) : ((p[1] > p[2]) ? 1 : 2);
	}

	void interpretPredictions(std::vector<Line>& lines, const int offset, const int currentBatchSize) const {
		
		for (int i = 0; i < currentBatchSize; i++) {

			const int idx = pArgMax(&output[predicitionSize * i]);

			Line& line = lines[offset + i];

			if (idx == 0) {
				//swap line points
				line = { line[2], line[3], line[0], line[1] };
			}
			else if (idx == 1) {
				//mark line as inconclusive
				line[0] = std::nanf("0");
			}
		}
	}


public:
	TreeDirectionModel(const wchar_t* modelPath, const int batchSize = 16) :
		OnnxModel(modelPath, "Tree Directions", batchSize) {

		inputShape = { batchSize, inputHeight, inputWidth, inputChannels };
		outputShape = { batchSize, predicitionSize };

		initializeTensors(inputSize, predicitionSize );
	}

	void predictAndInterpret(const cv::Mat& image, std::vector<Line>& lines) {

		const int numLines = (int)lines.size();

		for (int i = 0; i < numLines; i += batchSize) {

			const int currentBatchSize = numLines - i < batchSize ? numLines - i : batchSize;

			loadTreeBoxBatch(image, lines, i, currentBatchSize);

			predict();

			interpretPredictions(lines, i, currentBatchSize);

		}
	}
};

