#pragma once
#include "pch.h"
#include <algorithm>
#include "KMedoids.h"
#include "ShapeFile.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


class VectorMap {
	using Point = std::array<double, 2>;
	using Line = std::array<double, 4>;
	using Extent = std::array<double, 4>;
	using VMap = std::vector<std::vector<std::vector<Point>>>;

private:

	Point medianUnitVector(std::vector<Point>& uvecs) {
		Point medVec = { median(uvecs, uvecs.begin(), uvecs.end(), 0), median(uvecs, uvecs.begin(), uvecs.end(), 1) };
		const double mag = hypot(medVec[0], medVec[1]);

		return { { medVec[0] / mag, medVec[1] / mag } };
	}

	std::vector<Point> medianVectorClustering(std::vector<Point>& v) {

		// get x and y medians
		const double medX = median(v, v.begin(), v.end(), 0);
		const double medY = median(v, v.begin(), v.end(), 1);

		//get median orientation [0 - 180]
		double medAngle = cv::fastAtan2((float)medY, (float)medX);
		medAngle = medAngle <= 180.0 ? medAngle : medAngle - 180.0;

		//partition list into two clusters based on being on either side of the median orientation's dividing line
		const auto it = std::partition(v.begin(), v.end(), [medAngle](const Point u) { const double a = cv::fastAtan2((float)u[1], (float)u[0]);
																					   return a > medAngle && a < medAngle + 180.0; });

		//get medians of both clusters
		Point med1 = { median(v, v.begin(), it, 0), median(v, v.begin(), it, 1) };
		Point med2 = { median(v, it, v.end(), 0), median(v, it, v.end(), 1) };

		//convert cluster medians to unit vectors
		const double mag1 = hypot(med1[0], med1[1]);
		const double mag2 = hypot(med2[0], med2[1]);

		med1 = { med1[0] / mag1, med1[1] / mag1 };
		med2 = { med2[0] / mag2, med2[1] / mag2 };

		const double sqrt1_2 = 1.0 / sqrt(2.0);

		//if angle difference is less than 45 degrees by way of dot product
		if ((med1[0] * med2[0] + med1[1] * med2[1]) > sqrt1_2) {

			const double mag = hypot(medX, medY);

			//return single overall median
			return { { medX / mag, medY / mag } };
		}

		//return cluster medians
		return { med1, med2 };

	}

	std::vector<Point> medoidVectorClustering(std::vector<Point>& v) {
		const auto [medVecs, clusters] = KMedoids::kMedoids(v, 2, 100);

		const double ratio = (double)clusters[0].size() / (double)(clusters[0].size() + clusters[1].size());
		const double dp = medVecs[0][0] * medVecs[1][0] + medVecs[0][1] * medVecs[1][1];

		if (dp > (1.0 / sqrt(2.0)) || ratio > 0.6 || ratio < 0.4) {
			return { medianUnitVector(v) };
		}

		return medVecs;
	
	}

	std::vector<Point> averageUVecs(std::vector<Point>& uvecs) {

		switch (vmp.averagingMethod) {

			case Median:
				return { medianUnitVector(uvecs) };

			case DualMedian:
				return medianVectorClustering(uvecs);

			case DualMedoid:
				return medoidVectorClustering(uvecs);

			default:
				return { {nan("0"), nan("0")} };
		}
	}

	int clamp(int x, int _min, int _max) {
		return min(max(x, _min), _max);
	}

	std::vector<std::vector<std::vector<Line>>> sortLinesToGrid(const std::vector<Line>& lines, const double gridSize) {

		const int rows = (int)ceil((extent[1] - extent[3]) / gridSize); 
		const int cols = (int)ceil((extent[2] - extent[0]) / gridSize);

		std::vector<std::vector<std::vector<Line>>> grid;
		grid.resize(rows);

		for (auto& row : grid) {
			row.resize(cols);
		}

		for (const auto& l : lines) {
			Point midPoint = { 0.5 * (l[0] + l[2]), 0.5 * (l[1] + l[3]) };

			const int x = clamp((int)floor((midPoint[0] - extent[0]) / gridSize), 0, cols - 1);
			const int y = clamp((int)floor((extent[1] - midPoint[1]) / gridSize), 0, rows - 1);

			grid[y][x].push_back(l);
		}

		return grid;
	}

	std::vector<Point> linesToUnitVectors(const std::vector<Line>& lines) {

		std::vector<Point> uvecs;
		uvecs.reserve(lines.size());

		for (const auto& l : lines) {
			const double dx = l[2] - l[0];
			const double dy = l[3] - l[1];
			const double mag = hypot(dx, dy);

			uvecs.push_back({ dx / mag, dy / mag });
		}

		return uvecs;
	}

	//Taubin smoothing algorithm from pypi.org/project/shapelysmooth/#taubin
	void smoothPolygon(std::vector<Point>& pts, const double k = 0.5, const double u = 0.5, const int steps = 10) {

		int N = (int)pts.size();

		if (N < 3) {
			return;
		}

		std::vector<Point> L(N);

		for (int s = 0; s < steps; s++) {
			
			L[0][0] = (pts[N - 1][0] + pts[1][0]) / 2.0 - pts[0][0];
			L[0][1] = (pts[N - 1][1] + pts[1][1]) / 2.0 - pts[0][1];

			L[N - 1][0] = (pts[N - 2][0] + pts[0][0]) / 2.0 - pts[N - 1][0];
			L[N - 1][1] = (pts[N - 2][1] + pts[0][1]) / 2.0 - pts[N - 1][1];

			for (int i = 1; i < N - 1; i++) {
				L[i][0] = (pts[i - 1][0] + pts[i + 1][0]) / 2.0 - pts[i][0];
				L[i][1] = (pts[i - 1][1] + pts[i + 1][1]) / 2.0 - pts[i][1];
			}
			
			const double c = ((s & 1) == 0) ? k : -u;

			for (int i = 0; i < N; i++) {
				pts[i][0] += c * L[i][0];
				pts[i][1] += c * L[i][1];
			}
		}
	}

	Point centroid(const std::vector<Point>& pts) {
		const int N = (int)pts.size();
		
		Point c = { 0.0, 0.0 };
		double A = 0.0;

		for (int i = 0; i < N; i++) {
			const double d = pts[i][0] * pts[(i + 1) % N][1] - pts[(i + 1) % N][0] * pts[i][1];

			c[0] += (pts[i][0] + pts[(i + 1) % N][0]) * d;
			c[1] += (pts[i][1] + pts[(i + 1) % N][1]) * d;
			A += d;
		}

		A *= 3.0;

		c[0] /= A;
		c[1] /= A;
		
		return c;
	}

	void expandPolygon(std::vector<Point>& pts, const double factor) {
		
		const Point c = centroid(pts);

		for (auto& p : pts) {
			Point v = { p[0] - c[0], p[1] - c[1] };

			const double s = factor / hypot(v[0], v[1]);

			p = { p[0] + s * v[0], p[1] + s * v[1] };
		}

	}

	cv::Mat removeThinSections(const cv::Mat& image) {
		// Define a kernel to check 4 neighbors
		cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0,
			                                       1, 0, 1,
			                                       0, 1, 0);

		// Iterate over each pixel and count its neighbors
		cv::Mat resultImage(image.rows, image.cols, CV_8U, cv::Scalar(0));
		for (int i = 1; i < image.rows - 1; i++) {
			for (int j = 1; j < image.cols - 1; j++) {

				if (image.at<uchar>(i, j) > 0) {

					cv::Mat roi = image(cv::Range(i - 1, i + 2), cv::Range(j - 1, j + 2));
					double neighborsSum = cv::sum(roi.mul(kernel) / 255.0)[0];

					if (neighborsSum > 2) { // If the pixel has 3 or more neighbors, keep it
						resultImage.at<uchar>(i - 1, j - 1) = 255;
					}
				}
			}
		}

		return resultImage;
	}

	bool isClockwise(const std::vector<Point>& poly) {
	
		double area = 0.0;

		const int N = (int)poly.size();

		for (int i = 0; i < N; i++) {
			area += (poly[(i + 1) % N][0] - poly[i][0]) * (poly[(i + 1) % N][1] + poly[i][1]);
		}
		
		return area >= 0.0;
	}


public:

	VMap vmap;

	enum AveragingMethod { Median, DualMedian, DualMedoid, None };

	struct VectorMapParams {
		int gridSize;
		int minTrees;
		AveragingMethod averagingMethod;
		int steps;
		double growthFactor;
		double shrinkFactor;
	};

	VectorMapParams vmp;
	Extent extent;

	VectorMap(const std::vector<Line>& lines, const VectorMapParams vmp, const Extent extent) {

		this->vmp = vmp;
		this->extent = extent;

		auto grid = sortLinesToGrid(lines, vmp.gridSize);

		vmap.resize(grid.size());

		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < (int)grid.size(); i++) {

			const auto& row = grid[i];
			vmap[i].resize(row.size());

			for (int j = 0; j < (int)row.size(); j++) {

				if ((int)row[j].size() < vmp.minTrees) continue;

				std::vector<Point> uvecs = linesToUnitVectors(row[j]);

				vmap[i][j] = averageUVecs(uvecs);

			}
		}
	}

	static void exportMaps(std::vector<VectorMap>& vmaps, const std::string& resultsPath, const bool pointMap) {

		if (pointMap) {
			exportMapsAsPoints(vmaps, resultsPath);
		}
		else {
			exportMapsAsLines(vmaps, resultsPath);
		}
	
	}

	static void exportMapsAsLines(std::vector<VectorMap>& vmaps, const std::string& resultsPath) {
		std::string fname = resultsPath + "/Directions";

		std::vector<Line> allLines;
		std::vector<double> layerIdxs;

		for(int i = 0; i < (int)vmaps.size(); i++) {
			auto& map = vmaps[i];

			auto lines = map.asLines();

			allLines.insert(allLines.end(), lines.begin(), lines.end());
			layerIdxs.insert(layerIdxs.end(), lines.size(), (double)(i+1));
		}

		ShapeFile::writeFile<Line>(fname, SHPT_POLYLINE, allLines, "Layer", layerIdxs);
	}


	static void exportMapsAsPoints(std::vector<VectorMap>& vmaps, const std::string& resultsPath) {
		std::string fname = resultsPath + "/Point_Directions";

		std::vector<Point> allPoints;

		std::vector<std::string> fieldNames = { "Angle", "Layer" };
		std::vector<std::vector<double>> fieldValues;

		for (int i = 0; i < (int)vmaps.size(); i++) {
			auto& map = vmaps[i];

			auto [points, angles] = map.asPoints();

			allPoints.insert(allPoints.end(), points.begin(), points.end());
			
			for (const auto& a : angles) {
				fieldValues.push_back({ a, (double)(i+1) });
			}
		}

		ShapeFile::writeFile<Point>(fname, SHPT_POINT, allPoints, fieldNames, fieldValues);
	
	}


	void exportMapContours(const std::string& resultsPath) {

		cv::Mat gridImage = cv::Mat::zeros(cv::Size((int)vmap[0].size(), (int)vmap.size()), CV_8UC1);

		for (int i = 0; i < (int)vmap.size(); i++) {
			for (int j = 0; j < (int)vmap[0].size(); j++) {
				
				gridImage.at<uchar>(i, j) = vmap[i][j].size() > 0 ? 255 : 0;
			}
		}

		//cv::imwrite(resultsPath + "/gridImage.png", gridImage);
	
		cv::dilate(gridImage, gridImage, cv::Mat::ones(cv::Size(5, 5), CV_8UC1));
		cv::erode(gridImage, gridImage, cv::Mat::ones(cv::Size(3, 3), CV_8UC1));

		gridImage = removeThinSections(gridImage);

		cv::imwrite(resultsPath + "/gridImage.png", gridImage);
		/*cv::imshow("gridImage", gridImage);
		cv::waitKey(0);*/

		std::vector<std::vector<cv::Point>> contoursI;
		std::vector<std::vector<Point>> contours;
		cv::findContours(gridImage, contoursI, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		contours.resize(contoursI.size());

		for (int i = 0; i < (int)contoursI.size(); i++) {
			contours[i].reserve(contoursI[i].size());

			for (auto& p : contoursI[i]) {
				contours[i].push_back({(double)p.x, (double)p.y});
			}

			smoothPolygon(contours[i], vmp.growthFactor, vmp.shrinkFactor, vmp.steps);
			expandPolygon(contours[i], 1);
		}

		std::vector<std::vector<double>> contourPolygons;

		contourPolygons.resize(contours.size());

		for (int i = 0; i < (int)contours.size(); i++) {
			auto& c = contours[i];
			contourPolygons[i].reserve(c.size());

			// polygons must be clockwise
			if (isClockwise(c)) std::reverse(c.begin(), c.end());

			for (const auto& p : c) {

				contourPolygons[i].push_back(extent[0] + p[0] * vmp.gridSize + vmp.gridSize / 2.0);
				contourPolygons[i].push_back(extent[1] - p[1] * vmp.gridSize - vmp.gridSize / 2.0);
			}

			// polygon must be closed onto istelf ("closed ring")
			contourPolygons[i].push_back(contourPolygons[i][0]);
			contourPolygons[i].push_back(contourPolygons[i][1]);
		}

		std::string fname = resultsPath + "/Outlines_" + std::to_string(vmp.gridSize) + "m_min_" + std::to_string(vmp.minTrees) + "_trees";

		ShapeFile::writeFile<std::vector<double>>(fname, SHPT_POLYGON, contourPolygons);

	}


	void exportMap(const std::string& resultsPath, const bool pointMap) {

		if (pointMap) {
			exportAsPoints(resultsPath);
		}
		else {
			exportAsLines(resultsPath);
		}

	}

	void exportAsLines(const std::string& resultsPath) {
		std::string fname = resultsPath + "/Directions_" + std::to_string(vmp.gridSize) + "m_min_" + std::to_string(vmp.minTrees) + "_trees";

		auto lines = asLines();

		ShapeFile::writeFile<Line>(fname, SHPT_POLYLINE, lines);

	}

	void exportAsPoints(const std::string& resultsPath) {
		std::string fname = resultsPath + "/Point_Directions_" + std::to_string(vmp.gridSize) + "m_min_" + std::to_string(vmp.minTrees) + "_trees";

		auto [points, angles] = asPoints();

		ShapeFile::writeFile<Point>(fname, SHPT_POINT, points, "Angle", angles);
	}


	std::vector<Line> asLines() {
	
		std::vector<Line> lines;
		lines.reserve(2 * vmap.size() * vmap[0].size());

		const double s = vmp.gridSize / 2.0;

		for (int i = 0; i < (int)vmap.size(); i++) {
			for (int j = 0; j < (int)vmap[i].size(); j++) {
				for (const auto& vec : vmap[i][j]) {

					const double mag = hypot(vec[0], vec[1]);
					const Point uvec = { vec[0] / mag, vec[1] / mag };

					const Point center = { extent[0] + j * vmp.gridSize + s, extent[1] - i * vmp.gridSize - s };

					lines.push_back({ center[0] - uvec[0] * s * 0.5, center[1] - uvec[1] * s * 0.5, center[0] + uvec[0] * s * 0.5, center[1] + uvec[1] * s * 0.5 });
				}
			}
		}

		return lines;
	
	}

	std::pair<std::vector<Point>, std::vector<double>> asPoints() {
	
		std::vector<double> angles;
		angles.reserve(2 * vmap.size() * vmap[0].size());

		std::vector<Point> points;
		points.reserve(2 * vmap.size() * vmap[0].size());

		const double s = vmp.gridSize / 2.0;

		for (int i = 0; i < (int)vmap.size(); i++) {
			for (int j = 0; j < (int)vmap[i].size(); j++) {
				for (const auto& vec : vmap[i][j]) {

					//~0.3 degrees off should be more than good enough...
					angles.push_back((double)cv::fastAtan2((float)vec[1], (float)vec[0]));

					points.push_back({ extent[0] + j * vmp.gridSize + s, extent[1] - i * vmp.gridSize - s });
				}
			}
		}

		return { points, angles };
	
	}

	double median(const std::vector<Point>& vec, const std::vector<Point>::iterator begin, const std::vector<Point>::iterator end, const int idx) {

		const unsigned int n = (int)(end - begin);

		const unsigned int n_2 = n / 2;

		const auto compFunc = [idx](const Point a, const Point b) { return a[idx] < b[idx]; };

		//from an unsorted list, find the nth_element of the list as though it were a sorted list,
		//but without sorting causes that's a waste of time O(n) vs O(nlog(n))...
		nth_element(begin, begin + n_2, end, compFunc);

		//if n is even
		if ((n & 1) == 0) {

			const unsigned int n1_2 = n_2 - 1;

			nth_element(begin, begin + n1_2, end, compFunc);

			return (vec[n1_2][idx] + vec[n_2][idx] * 0.5);
		}

		return vec[n_2][idx];

	}

};