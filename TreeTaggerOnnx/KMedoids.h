#pragma once
#include "pch.h"
#include <vector>
#include <array>
#include <queue>
#include <algorithm>
#include <random>


class KMedoids{
	using Point = std::array<double, 2>;

private:
	static std::vector<int> randBuild(std::vector<int>& ptsIndicies, const int k) {
		std::vector<int> initMedoids;
		initMedoids.reserve(k);

		std::sample(ptsIndicies.begin(), ptsIndicies.end(), std::back_inserter(initMedoids), k, std::mt19937{ std::random_device{}() });

		return initMedoids;
	}

	static std::vector<int> splitBuild(const int numOfPts, const int k) {
		std::vector<int> initMedoid;
		initMedoid.reserve(k);

		for (int i = 0; i < numOfPts; i += (numOfPts - 1) / (k - 1)) {
			initMedoid.push_back(i);
		}

		return initMedoid;
	}

	static std::vector<int> deterministicBuild(const std::vector<std::vector<double>>& distMat, const int k){

		std::priority_queue<std::pair<double, int>> q;

		for (int i = 0; i < distMat.size(); i++) {
			double sum = 0.0;

			for (int j = 0; j < distMat.size(); j++){
				sum -= distMat[i][j];
			}

			q.emplace(sum, i);
		}
		
		std::vector<int> initMedoids;
		initMedoids.reserve(k);

		for (int i = 0; i < k; i++) {
			initMedoids.push_back(q.top().second);
			q.pop();
		}

		return initMedoids;
	}

	static inline double dotDist(Point a, Point b) {
		return 1.0 - (a[0] * b[0] + a[1] * b[1]);
	}

	static inline double l2Dist(Point a, Point b) {
		return hypot(a[0] - b[0], a[1] - b[1]);
	}

	static std::vector<std::vector<double>> computeDistMatrix(const std::vector<Point>& pts) {
		const int size = (int)pts.size();

		std::vector<std::vector<double>> distMat;
		distMat.resize(size);

		for (int i = 0; i < size; i++) {
			distMat[i].resize(size, 0.0);
		}

		for (int i = 0; i < size; i++) {
			for (int j = i + 1; j < size; j++) {
				const double dist = dotDist(pts[i], pts[j]);

				distMat[i][j] = dist;
				distMat[j][i] = dist;
			}
		}

		return distMat;
	}

	static void groupClosest(std::vector<std::vector<int>>& clusters, std::vector<double>& clusterDistSums, const std::vector<int>& medoids, const std::vector<std::vector<double>>& distMat, const int numOfPts) {

		for (int i = 0; i < (int)clusters.size(); i++) {
			clusters[i].clear();
			clusterDistSums[i] = 0.0;
		}

		for (int i = 0; i < numOfPts; i++) {
			double minDist = distMat[medoids[0]][i]; //dotDist(pts[medoids[0]], pts[i]);
			int minIdx = 0;

			for (int j = 1; j < (int)medoids.size(); j++) {
				const double dist = distMat[medoids[j]][i]; //dotDist(pts[medoids[j]], pts[i]);

				if (dist < minDist) {
					minDist = dist;
					minIdx = j;
				}
			}

			clusters[minIdx].push_back(i);
			clusterDistSums[minIdx] += minDist;
		}
	}
	 
	static double computeClusterDistSum(const std::vector<int>& cluster, const int medoid, const std::vector<std::vector<double>>& distMat) {
		double clusterDistSum = 0.0;

		for (const int i : cluster){
			clusterDistSum += distMat[i][medoid]; //dotDist(pts[cluster[i]], pts[medoid]);
		}

		return clusterDistSum;
	}


public:
	
	static std::pair<std::vector<Point>, std::vector<std::vector<int>>> kMedoids(const std::vector<Point>& pts, const int k, const int maxIters) {
		
		//initialization
		int iters = 0;
		bool changed = true;

		const std::vector<std::vector<double>> distMat = computeDistMatrix(pts);

		std::vector<int> medoids = deterministicBuild(distMat, k); //splitBuild(pts.size(), k);

		std::vector<std::vector<int>> clusters;
		clusters.resize(k);

		std::vector<double> clusterDistSums;
		clusterDistSums.resize(k);


		//algorithm
		while (iters < maxIters && changed) {
			iters++;
			changed = false;

			groupClosest(clusters, clusterDistSums, medoids, distMat, (int)pts.size());

			for (int i = 0; i < k; i++) {
				for (int j = 0; j < (int)clusters[i].size(); j++) {
					const double distSum = computeClusterDistSum(clusters[i], clusters[i][j], distMat);

					if (distSum < clusterDistSums[i]) {
						changed = true;
						clusterDistSums[i] = distSum;
						medoids[i] = clusters[i][j];
					}
				}
			}
		}


		//return result
		std::vector<Point> finalMedoids;
		finalMedoids.reserve(k);

		for (int i = 0; i < k; i++) {
			finalMedoids.push_back(pts[medoids[i]]);
		}

		return { finalMedoids, clusters };
	}


};


