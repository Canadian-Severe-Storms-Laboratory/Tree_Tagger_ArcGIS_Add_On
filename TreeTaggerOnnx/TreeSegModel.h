#pragma once
#include "pch.h"
#include "OnnxModel.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "ximgproc.h"


class TreeSegModel : public OnnxModel{

private:

	static constexpr int inputWidth = 256;
	static constexpr int inputHeight = 256;
	static constexpr int inputChannels = 3;
	static constexpr int inputSize = inputWidth * inputHeight * inputChannels;
	static constexpr double inputScale = 0.05;

	static constexpr int predictionWidth = 256;
	static constexpr int predictionHeight = 256;
	static constexpr int predictionChannels = 2;
	static constexpr int predicitionSize = predictionWidth * predictionHeight * predictionChannels;


	static std::array<float, 2> resizeAndNormalize(cv::Mat& image, const double scale) {
		const cv::Size s = image.size();

		const std::array<float, 2> sf = { (float)(s.width * scale / inputScale), (float)(s.height * scale / inputScale) };

		cv::resize(image, image, cv::Size(((int)sf[0] / inputWidth) * inputWidth, ((int)sf[1] / inputHeight) * inputHeight));

		image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

		return sf;
	}

	std::vector<std::vector<cv::Mat>> splitIntoTileBatches(const cv::Mat& image, const int shift) const {

		const int tilesPerCol = image.rows / inputHeight - (shift > 0 ? 1 : 0);
		const int tilesPerRow = image.cols / inputWidth - (shift > 0 ? 1 : 0);
		const int numOfTiles = tilesPerCol * tilesPerRow;
		const int numOfBatches = numOfTiles / batchSize + 1;

		std::vector<std::vector<cv::Mat>> tileBatches;

		tileBatches.resize(numOfBatches);

		int tilesInBatch = numOfTiles < batchSize ? numOfTiles : batchSize;

		tileBatches[0].resize(tilesInBatch);

		int batchIdx = 0;
		int tileIdx = 0;
		int tilesComplete = 0;

		//split image into tiles
		for (int i = 0; i < tilesPerCol; i++) {
			for (int j = 0; j < tilesPerRow; j++) {
				tileBatches[batchIdx][tileIdx++] = image(cv::Range(i * inputHeight + shift, i * inputHeight + inputHeight + shift), cv::Range(j * inputWidth + shift, j * inputWidth + inputWidth + shift));

				if (tileIdx == tilesInBatch && batchIdx + 1 < numOfBatches) {
					tilesComplete += tileIdx;
					tileIdx = 0;
					batchIdx++;

					tilesInBatch = numOfTiles - tilesComplete < batchSize ? numOfTiles - tilesComplete : batchSize;

					tileBatches[batchIdx].resize(tilesInBatch);
				}
			}
		}

		return tileBatches;
	}

	void serializeTileBatch(const std::vector<cv::Mat>& tiles) {
		#pragma omp parallel for
		for (int j = 0; j < (int)tiles.size(); j++) {

			const auto& tile = tiles[j];

			//serialize tile
			for (int y = 0; y < inputHeight; y++) {
				for (int x = 0; x < inputWidth; x++) {
					const cv::Vec3f& pixel = tile.at<cv::Vec3f>(y, x);

					const int idx = j * inputSize + inputChannels * (y * inputWidth + x);

					input[idx] = pixel[0];
					input[idx + 1] = pixel[1];
					input[idx + 2] = pixel[2];
				}
			}
		}
	}

	void unserializeAndInterpretBatch(std::vector<cv::Mat>& outTiles) const {
		#pragma omp parallel for
		for (int j = 0; j < outTiles.size(); j++) {

			//unserialize tile
			for (int y = 0; y < predictionHeight; y++) {
				for (int x = 0; x < predictionWidth; x++) {

					const int idx = j * predicitionSize + predictionChannels * (y * predictionWidth + x);

					outTiles[j].at<uint8_t>(y, x) = outTiles[j].at<uint8_t>(y, x) | (output[idx] > output[idx + 1] ? (uint8_t)0 : (uint8_t)255);
				}
			}
		}
	}

	static void postProcessOutput(cv::Mat& outImage, const std::array<float, 2> s) {
		//thinning algorithm
		ximgproc::guo_hall_thinning(outImage);
		cv::dilate(outImage, outImage, cv::Mat());
		cv::blur(outImage, outImage, cv::Size(3, 3));
		cv::resize(outImage, outImage, cv::Size((int)round(s[0]), (int)round(s[1])));
	}

	static void rotateAllTiles90(std::vector<std::vector<cv::Mat>>& tileBatches) {
		for (auto& tileBatch : tileBatches){
			for (auto& tile : tileBatch){
				cv::rotate(tile, tile, cv::ROTATE_90_CLOCKWISE);
			}
		}
	}


public:

	TreeSegModel(const wchar_t* modelPath, const int batchSize = 16) :
	OnnxModel(modelPath, "Tree Segmentation", batchSize) {

		inputShape = { batchSize, inputHeight, inputWidth, inputChannels };
		outputShape = { batchSize, predictionHeight, predictionWidth, predictionChannels };

		initializeTensors(inputSize, predicitionSize);
	}

	std::vector<std::array<double, 4>> predictAndInterpret(cv::Mat image, const double scale, const bool fast = true) {

		const auto sf = resizeAndNormalize(image, scale);

		cv::Mat outImage = cv::Mat(image.size(), CV_8UC1);

		//for each shift (0, 128)
		for (int s = 0; s == 0 || (!fast && s < 2); s++) {

			const int shift = s * 128;

			std::vector<std::vector<cv::Mat>> inputTileBatches = splitIntoTileBatches(image, shift);
			std::vector<std::vector<cv::Mat>> outputTileBatches = splitIntoTileBatches(outImage, shift);

			//for each rotation (0, 90, 180, 270)
			for (int r = 0; r == 0 || (!fast && r < 4); r++) {

				//for each batch
				for (int i = 0; i < (int)inputTileBatches.size(); i++) {

					serializeTileBatch(inputTileBatches[i]);

					//run model on batch
					predict();

					unserializeAndInterpretBatch(outputTileBatches[i]);

				}

				if (!fast) {
					rotateAllTiles90(inputTileBatches);
					rotateAllTiles90(outputTileBatches);
				}
			}
		}
		
		postProcessOutput(outImage, sf);

		auto lines = ximgproc::fastLineDetector(outImage, false);

		return lines;
	}
};

