#pragma once
#include "pch.h"
#include <iostream>
#include <vector>
#include "ShapeFile.h"
#include "Utils.h"
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

namespace Coords {
	
	using Point = std::array<double, 2>;
	using Line = std::array<double, 4>;
	using Extent = std::array<double, 4>;
	using namespace Utils;

	static constexpr double inputScale = 0.05; // 5cm imagery

	void referenceAndAppend(std::vector<Line>& allLines, const std::vector<Line>& lines, const IMGP::ImageFile& imageFile, const Extent& extent) {

		const double offsetX = (imageFile.coords[0] - extent[0]) / inputScale;
		const double offsetY = (extent[1] - imageFile.coords[1]) / inputScale;

		for (const auto& l : lines) {
			allLines.push_back({ offsetX + l[0], offsetY + l[1], offsetX + l[2], offsetY + l[3] });
		}
	}

	void dereferenceLines(std::vector<Line>& lines, const IMGP::ImageFile& imageFile, const Extent& extent) {

		const double offsetX = (imageFile.coords[0] - extent[0]) / inputScale;
		const double offsetY = (extent[1] - imageFile.coords[1]) / inputScale;

		for (auto& l : lines) {
			l = { l[0] - offsetX, l[1] - offsetY, l[2] - offsetX, l[3] - offsetY };
		}
	}

	//referenced (to all images) pixels to coords
	std::vector<Line> pixelsToCoords(const std::vector<Line>& lines, const Extent& extent) {
		std::vector<Line> clines;
		clines.reserve(lines.size());

		for (const auto& l : lines) {
			Line cl = { l[0] * inputScale + extent[0], extent[1] - l[1] * inputScale, l[2] * inputScale + extent[0], extent[1] - l[3] * inputScale };
			clines.push_back(cl);
		}

		return clines;
	}

	//relative (to a single image) pixels to coords
	void pixelsToCoords(std::vector<Line>& lines, const IMGP::ImageFile& imgf, Extent& extent) {
		const double offsetX = imgf.coords[0];
		const double offsetY = imgf.coords[1];

		for (auto& l : lines) {
			l = { offsetX + inputScale * l[0], offsetY - inputScale * l[1], offsetX + inputScale * l[2], offsetY - inputScale * l[3] };
		}

	}

	std::vector<Line> coordsToPixels(const std::vector<Line>& lines, const Extent& extent) {
		std::vector<Line> plines;
		plines.reserve(lines.size());

		for (const auto& l : lines) {
			Line pl = { (l[0] - extent[0]) / inputScale, (extent[1] - l[1]) / inputScale, (l[2] - extent[0]) / inputScale, (extent[1] - l[3]) / inputScale };
			plines.push_back(pl);
		}

		return plines;
	}

	std::vector<Point> coordsToPixels(const std::vector<Point>& pts, const Extent& extent) {
		std::vector<Point> pPts;
		pPts.reserve(pts.size());

		for (const auto& p : pts) {
			Point pp = { (p[0] - extent[0]) / inputScale, (extent[1] - p[1]) / inputScale };
			pPts.push_back(pp);
		}

		return pPts;
	}

	std::vector<Line> shapesToLines(const std::vector<std::vector<double>>& shapes) {
		std::vector<Line> lines;
		lines.reserve(shapes.size()/2);

		for (const auto& shape : shapes) {
			if (shape.size() < 4) throw std::runtime_error("Invalid polyline shapefile, too few points");

			lines.push_back({ shape[0], shape[1], shape[2], shape[3] });
		}
	
		return lines;
	}

	std::vector<cv::Point> pointsToCvPoints(const std::vector<Point>& pts) {
		std::vector<cv::Point> cvPoints;
		cvPoints.reserve(pts.size());
		
		for (const auto& p : pts) {
			cvPoints.emplace_back((int)round(p[0]), (int)round(p[1]));
		}

		return cvPoints;
	}

	void exportPolylines(const std::string& resultsPath, const std::string& type, const std::vector<Line>& lines, const Extent& extent) {
		printState("Exporting " + type);

		auto clines = pixelsToCoords(lines, extent);

		ShapeFile::writeFile<Line>(resultsPath + "/" + type, SHPT_POLYLINE, clines);
	}

	void exportPolylines(const std::string& resultsPath, const std::string& type, std::vector<Line>& lines) {
		printState("Exporting " + type);

		ShapeFile::writeFile<Line>(resultsPath + "/" + type, SHPT_POLYLINE, lines);
	}

	inline bool pointInBox(const Point& pt, const Extent& box) {
		if (pt[0] > box[0] && pt[0] <= box[0] + box[2] && pt[1] <= box[1] && pt[1] > box[1] - box[3]) {
			return true;
		}
		return false;
	}

	void exportDirections(const std::string& resultsPath, std::vector<VectorMap>& vmaps, const bool pointMaps, const bool joinMaps) {

		printState("Exporting Vector Maps");

		if (joinMaps) {
			VectorMap::exportMaps(vmaps, resultsPath, pointMaps);
		}
		else {
			for (auto& map : vmaps) {
				map.exportMap(resultsPath, pointMaps);
			}
		}

	}

	void exportOutlines(const std::string& resultsPath, VectorMap& vm) {

		printState("Exporting Outlines");

		vm.exportMapContours(resultsPath);
	}

	void exportOutlines(const std::string& resultsPath, std::vector<VectorMap>& vmaps) {

		printState("Exporting Outlines");

		for (auto& vm : vmaps) {
			vm.exportMapContours(resultsPath);
		}
	}


	Extent findExtentOverlap(const Extent& extent1, const Extent& extent2) {

		const double leftX = max(extent1[0], extent2[0]);
		const double rightX = min(extent1[2], extent2[2]);
		const double topY = min(extent1[1], extent2[1]);
		const double bottomY = max(extent1[3], extent2[3]);

		if (leftX < rightX && bottomY < topY) {
			return { leftX, topY, rightX, bottomY };
		}

		return { std::nan("0"), 0.0, 0.0, 0.0 };
	}

	//might need to handle rounding better
	std::array<int, 4> findPixelRegion(const cv::Mat& img, const Extent& extent, const Extent& overlap) {

		const int x1 = max(0, (int)round((overlap[0] - extent[0]) / inputScale));
		const int x2 = min(img.cols, (int)round((overlap[2] - extent[0]) / inputScale));

		const int y1 = max(0, (int)round((extent[1] - overlap[1]) / inputScale));
		const int y2 = min(img.rows, (int)round((extent[1] - overlap[3]) / inputScale));

		return { y1, y2, x1, x2 };
	}


	void resizeToInputScale(cv::Mat& image, const double s) {

		if (abs(s - inputScale) > 0.0001) {
			cv::resize(image, image, cv::Size(), s / inputScale, s / inputScale);
		}

	}


	std::pair<cv::Mat, IMGP::ImageFile> extendImage(const std::vector<IMGP::ImageFile>& imageFiles, const int s, const int idx) {

		auto& imgf = imageFiles[idx];
		cv::Mat image = imread(imgf.path);
		resizeToInputScale(image, imgf.scale);

		//get extended coords / image size
		const Point coords = { imgf.coords[0] - s * inputScale, imgf.coords[1] + s * inputScale };
		const std::array<int, 2> size = { image.cols + 2 * s, image.rows + 2 * s };

		IMGP::ImageFile eimgf = { imgf.path, coords, size, inputScale };

		cv::Mat eImage = cv::Mat::zeros(cv::Size(size[0], size[1]), CV_8UC3);

		image.copyTo(eImage(cv::Range(s, size[1] - s), cv::Range(s, size[0] - s)));

		const Extent extent = eimgf.getExtent();

		for (int i = 0; i < (int)imageFiles.size(); i++) {

			if (i == idx) continue;

			Extent extentI = imageFiles[i].getExtent();

			Extent overlap = findExtentOverlap(extent, extentI);

			if (isnan(overlap[0])) continue;

			auto ri = findPixelRegion(eImage, extent, overlap);

			cv::Mat imgI = imread(imageFiles[i].path);
			resizeToInputScale(imgI, imageFiles[i].scale);

			auto rj = findPixelRegion(imgI, extentI, overlap);

			imgI(cv::Range(rj[0], rj[1]), cv::Range(rj[2], rj[3])).copyTo(eImage(cv::Range(ri[0], ri[1]), cv::Range(ri[2], ri[3])));
		}

		return { eImage, eimgf };
	}

	std::vector<std::vector<Line>> sortLinesToImages(const std::vector<Line>& lines, const std::vector<IMGP::ImageFile>& imageFiles, const Extent& extent) {

		std::vector<std::vector<Line>> lineGroups;
		lineGroups.resize(imageFiles.size());

		for (const auto& l : lines) {
			Point midPoint = { 0.5 * (l[0] + l[2]), 0.5 * (l[1] + l[3]) };

			midPoint = { midPoint[0] * inputScale + extent[0], extent[1] - midPoint[1] * inputScale };

			for (int i = 0; i < imageFiles.size(); i++) {

				const auto& imgf = imageFiles[i];

				if (pointInBox(midPoint, { imgf.coords[0], imgf.coords[1], imgf.size[0] * imgf.scale, imgf.size[1] * imgf.scale })) {
					lineGroups[i].push_back(l);
					break;
				}
			}
		}

		return lineGroups;
	}


	void maskPolygon(cv::Mat& image, const std::vector<cv::Point>& poly) {
	
		cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

		cv::fillPoly(mask, poly, cv::Scalar(255));

		//cv::bitwise_and(image, image, image, mask); inplace doesn't work...

		//inplace AND
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				auto& p = image.at<cv::Vec3b>(i, j);
				auto m = mask.at<uchar>(i, j);

				p[0] &= m;
				p[1] &= m;
				p[2] &= m;
			}
		}

	}
}