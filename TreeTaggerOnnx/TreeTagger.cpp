#include "pch.h"
#include "TreeTagger.h"


std::vector<std::array<double, 4>> detectTrees(const std::vector<IMGP::ImageFile>& imageFiles, const LJP::LineJoinParams& ljp, const Extent& extent, std::string assemblyPath, std::vector<Point>& polygonPts, const bool fast) {

	const auto treeSegModelPath = std::wstring(assemblyPath.begin(), assemblyPath.end()) + L"/models/TreeSegModel_tensorRT.onnx";
	auto treeSegModel = TreeSegModel(treeSegModelPath.c_str(), 18);

	std::vector<Line> allLines;

	printState("Detecting Trees in Imagery");

	ProgressBar bar((int)imageFiles.size());

	for (const auto& imageFile : imageFiles)
	{
		cv::Mat im = imread(imageFile.path, cv::IMREAD_COLOR);

		auto pPolygonPts = pointsToCvPoints(coordsToPixels(polygonPts, imageFile.getExtent()));

		if (pPolygonPts.size() > 2) maskPolygon(im, pPolygonPts);

		auto lines = treeSegModel.predictAndInterpret(im, imageFile.scale, fast);

		referenceAndAppend(allLines, lines, imageFile, extent);

		bar.update();
	}

	bar.~ProgressBar();

	printState("Joining Detected Tree Lines");

	const int x = (int)ceil((extent[2] - extent[0]) / inputScale / 256.0);
	const int y = (int)ceil((extent[1] - extent[3]) / inputScale / 256.0);

	//0.15708f, 0.03142f, 1.55f, 4.5, 50, 600
	auto joinedLines = JoinLines::joinLines(allLines, x, y, ljp.angleThreshold, ljp.directMergeAngleThreshold, ljp.directMergeThreshold, ljp.distThreshold, ljp.minLineLength, ljp.maxLineLength);

	return joinedLines;
}


std::vector<std::array<double, 4>> refineLines(const std::vector<Line>& lines, const LJP::LineJoinParams& ljp, const Extent& extent) {
	printState("Refining Tree Lines");

	auto clines = coordsToPixels(lines, extent);
	constexpr double minLength = 5.0 / inputScale;

	for (auto& l : clines) {
		const double dx = l[2] - l[0];
		const double dy = l[3] - l[1];
		const double mag = hypot(dx, dy);

		if (mag > minLength) continue;

		//extent small line in vector direction
		Point uVec = { dx / mag, dy / mag };

		l = { l[0], l[1], l[0] + uVec[0] * minLength, l[1] + uVec[1] * minLength };
	}

	const int x = (int)ceil((extent[2] - extent[0]) / inputScale / 256.0);
	const int y = (int)ceil((extent[1] - extent[3]) / inputScale / 256.0);

	//0.15708f, 0.03142f, 1.55f, 4.5, 50, 600
	const auto joinedLines = JoinLines::joinLines(clines, x, y, ljp.angleThreshold, ljp.directMergeAngleThreshold, ljp.directMergeThreshold, ljp.distThreshold, minLength, ljp.maxLineLength);

	return pixelsToCoords(joinedLines, extent);
}


std::vector<Line> detectDirections(std::vector<Line>& lines, const std::vector<IMGP::ImageFile>& imageFiles, Extent& extent, std::string assemblyPath) {
	const auto treeDirectionModelPath = std::wstring(assemblyPath.begin(), assemblyPath.end()) + L"/models/TreeDirectionModel_tensorRT.onnx";
	auto treeDirectionModel = TreeDirectionModel(treeDirectionModelPath.c_str(), 18);

	printState("Determining Treefall Vectors");

	auto lineGroups = sortLinesToImages(lines, imageFiles, extent);

	ProgressBar bar((int)imageFiles.size());

	for (int i = 0; i < (int)imageFiles.size(); i++) {

		//get extended image
		auto [extendedImg, eimgf] = extendImage(imageFiles, 512, i);

		//relative to extended img
		dereferenceLines(lineGroups[i], eimgf, extent);

		treeDirectionModel.predictAndInterpret(extendedImg, lineGroups[i]);

		//relative to extended img
		pixelsToCoords(lineGroups[i], eimgf, extent);

		bar.update();

	}

	//gather tree vectors into a single array
	std::vector<Line> treeVectors;

	for (const auto& group : lineGroups) {
		for (const auto& line : group) {
			if (!isnan(line[0])) treeVectors.push_back(line);
		}
	}

	return treeVectors;
}


std::vector <VectorMap> computeVectorMaps(const std::vector<Line>& lines, const std::vector<VM::VectorMapParams>& vmp, const Extent& extent) {

	printState("Computing Vector Maps");

	std::vector <VectorMap> vectorMaps;
	vectorMaps.reserve(vmp.size());

	ProgressBar bar((int)vmp.size());

	for (const auto& params : vmp) {

		vectorMaps.emplace_back(lines, params, extent);

		bar.update();

		checkCancelled();
	}

	std::sort(vectorMaps.begin(), vectorMaps.end(), [](const VectorMap& a, const VectorMap& b) { return a.vmp.gridSize < b.vmp.gridSize; });

	return vectorMaps;
}


std::vector<VectorMap> vectorMapsFromShape(const std::string& shapePath, const std::vector<VM::VectorMapParams>& vmp, const Extent& extent) {

	std::vector<std::vector<double>> vectorShapes;

	if (ShapeFile::readFile(shapePath, vectorShapes) != SHPT_POLYLINE) {
		throw std::runtime_error("Shapefile Type is not Polyline");
	}

	auto vectorLines = shapesToLines(vectorShapes);

	//compute area averaged directions
	return computeVectorMaps(vectorLines, vmp, extent);
}


int analyzeEvent(const char* packet, ConsoleCallback cb, bool* cf) { return apiWrapper(cb, cf, [&packet]() {

	auto ttp = JMP::parsePacket<ProjectParser, ImageParser, LineJoinParser, VectorMapParser>(packet);

	const auto resultsPath = setupEnv(ttp.projectPath);

	// polygon reference
	auto joinedLines = detectTrees(ttp.imageFiles, ttp.lineJoinParams, ttp.extent, ttp.assemblyPath, ttp.polygonPts, ttp.fast);

	exportPolylines(resultsPath, "Raw_Tree_Lines", joinedLines, ttp.extent);

	auto vectorLines = detectDirections(joinedLines, ttp.imageFiles, ttp.extent, ttp.assemblyPath);

	exportPolylines(resultsPath, "Tree_Vectors", vectorLines);

	auto refinedLines = refineLines(vectorLines, ttp.lineJoinParams, ttp.extent);

	exportPolylines(resultsPath, "Refined_Tree_Lines", refinedLines);

	//compute area averaged directions
	auto directionMaps = computeVectorMaps(vectorLines, ttp.vectorMapParams, ttp.extent);

	//export directions
	exportDirections(resultsPath, directionMaps, ttp.pointMaps, ttp.joinMaps);

	exportOutlines(resultsPath, directionMaps[0]);

});}

int createVectorMaps(const char* packet, ConsoleCallback cb, bool* cf) { return apiWrapper(cb, cf, [&packet]() {

	auto mapPacket = JMP::parsePacket<ProjectParser, ShapeFileParser, VectorMapParser>(packet);

	const auto resultsPath = setupEnv(mapPacket.projectPath, false);

	//compute area averaged directions from vectors shapefile
	auto directionMaps = vectorMapsFromShape(mapPacket.shapeFilePath, mapPacket.vectorMapParams, mapPacket.extent);

	//export directions
	exportDirections(resultsPath, directionMaps, mapPacket.pointMaps, mapPacket.joinMaps);

});}

int createOutlines(const char* packet, ConsoleCallback cb, bool* cf) { return apiWrapper(cb, cf, [&packet]() {

	std::cout << packet << std::endl;
	
	auto outlinePacket = JMP::parsePacket<ProjectParser, ShapeFileParser, OutlineParser>(packet);

	const auto resultsPath = setupEnv(outlinePacket.projectPath, false);

	//compute area averaged directions from vectors shapefile
	auto directionMaps = vectorMapsFromShape(outlinePacket.shapeFilePath, outlinePacket.vectorMapParams, outlinePacket.extent);

	exportOutlines(resultsPath, directionMaps);

});}




