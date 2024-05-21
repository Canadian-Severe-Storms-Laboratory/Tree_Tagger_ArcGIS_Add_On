#pragma once
#include "shapefil.h"
#include "Console.h"
#include <vector>
#include <string>

class ShapeFile {
	

private:
	struct SHPFILE
	{
		SHPHandle sf;
		DBFHandle df;
		int type;
	};

	static SHPFILE createFile(const std::string& fname, const int type) {
		SHPFILE shp;

		const auto sName = fname + ".shp";
		const auto dName = fname + ".dbf";

		shp.sf = SHPCreate(sName.c_str(), type);
		shp.df = DBFCreate(dName.c_str());

		if (shp.sf == nullptr || shp.df == nullptr) {
			throw "Could not create shapefile: " + sName;
		}

		shp.type = type;

		return shp;
	}

	static SHPFILE openFile(const std::string& fname) {
		SHPFILE shp;

		const auto sName = fname + ".shp";
		const auto dName = fname + ".dbf";

		shp.sf = SHPOpen(sName.c_str(), "rb");
		shp.df = DBFOpen(dName.c_str(), "rb");

		if (shp.sf == nullptr || shp.df == nullptr) {
			throw "Could not open shapefile: " + sName;
		}

		shp.type = shp.sf->nShapeType;

		return shp;
	}

	static void closeFile(const SHPFILE& shp) {

		if (shp.sf != nullptr) {
			SHPClose(shp.sf);
		}
		if (shp.df != nullptr) {
			DBFClose(shp.df);
		}
	}

	static std::vector<int> addFields(const SHPFILE& shp, const std::vector<std::string>& fields) {

		std::vector<int> fieldIds;
		fieldIds.reserve(fields.size());

		for (const auto& field : fields){
			fieldIds.push_back(DBFAddField(shp.df, field.c_str(), FTDouble, 10, 6));
		}

		return fieldIds;
	}

	static void readFieldNames(const SHPFILE& shp, std::vector<std::string>& fields, const int numFields) {
		fields.reserve(numFields);

		//read dbf field names
		for (int i = 0; i < numFields; i++) {
			char fieldNameBuffer[20] = {};
			int fieldNameLength = 0;
			auto fieldInfo = DBFGetFieldInfo(shp.df, i, fieldNameBuffer, &fieldNameLength, nullptr);

			std::cout << fieldNameLength << '\n';

			fields.emplace_back(fieldNameBuffer, fieldNameLength);
		}
	}

	template <typename container>
	static void writeShape(SHPFILE& shp, const int shapeNum, std::vector<int>& fIds, container& shape, std::vector<double>& fieldValues) {

		if (shape.size() < 2) return;

		const int numVerts = (int)shape.size() / 2;
		
		std::vector<double> xs;
		xs.reserve(numVerts);

		std::vector<double> ys;
		ys.reserve(numVerts);

		//split points into x and y
		for (int i = 0; i < shape.size() - 1; i += 2) {
			xs.push_back((double)shape.at(i));
			ys.push_back((double)shape.at(i + 1));
		}

		//create and write shape
		const auto shapeObject = SHPCreateSimpleObject(shp.type, numVerts, xs.data(), ys.data(), NULL);
		SHPWriteObject(shp.sf, shapeNum, shapeObject);
		SHPDestroyObject(shapeObject);

		//write associated fields to database file
		for (int i = 0; i < fIds.size(); i++) {
			DBFWriteDoubleAttribute(shp.df, shapeNum, fIds.at(i), fieldValues.at(i));
		}
	}

	static void readShape(const SHPFILE& shp, const int idx, std::vector<double>& shape) {
		const auto shapeObject = SHPReadObject(shp.sf, idx);

		const int numVerts = shapeObject->nVertices;
		shape.reserve(2 * numVerts);

		for (int j = 0; j < numVerts; j++) {
			shape.push_back(shapeObject->padfX[j]);
			shape.push_back(shapeObject->padfY[j]);
		}
	}

	static void readFieldValues(const SHPFILE& shp, const int idx, std::vector<double>& fieldValues, const int numFields) {
		fieldValues.reserve(numFields);

		for (int j = 0; j < numFields; j++) {
			fieldValues.push_back(DBFReadDoubleAttribute(shp.df, idx, j));
		}
	}


public:

	template <typename container>
	static void writeFile(const std::string fname, const int type, std::vector<container>& shapes, std::vector<std::string>& fields, std::vector<std::vector<double>>& fieldValues) {

		auto shp = createFile(fname, type);

		auto fIds = addFields(shp, fields);

		for (int i = 0; i < shapes.size(); i++) {
			writeShape(shp, i, fIds, shapes[i], fieldValues[i]);
		}

		closeFile(shp);
	}

	template <typename container>
	static void writeFile(const std::string fname, const int type, std::vector<container>& shapes, const std::string field, const std::vector<double> fieldValues) {
		auto shp = createFile(fname, type);

		const std::vector<std::string> fields = { field };

		auto fIds = addFields(shp, fields);

		for (int i = 0; i < shapes.size(); i++) {

			std::vector<double> value = { fieldValues[i]};

			writeShape(shp, i, fIds, shapes[i], value);
		}

		closeFile(shp);

	}

	template <typename container>
	static void writeFile(const std::string fname, const int type, std::vector<container>& shapes) {
		auto shp = createFile(fname, type);

		const std::vector<std::string> fields = { "Number" };

		auto fIds = addFields(shp, fields);

		for (int i = 0; i < shapes.size(); i++) {

			std::vector<double> idx = { (double)i };

			writeShape(shp, i, fIds, shapes[i], idx);
		}

		closeFile(shp);
	}

	static int readFile(const std::string& fname, std::vector<std::vector<double>>& shapes, std::vector<std::string>& fields, std::vector<std::vector<double>>& fieldValues) {
		const auto shp = openFile(fname);

		//file parameters
		const int numShapes = shp.sf->nRecords;
		const int numFields = shp.df->nFields;
		const int numRecords = shp.df->nRecords;

		if (numShapes != numRecords || numFields < 1) {
			return -1; //invalid file
		}

		//initialize containers
		shapes.resize(numShapes);
		fieldValues.resize(numShapes);
		
		readFieldNames(shp, fields, numFields);

		//read shapes and field values
		for (int i = 0; i < numShapes; i++) {

			readShape(shp, i, shapes[i]);

			readFieldValues(shp, i, fieldValues[i], numFields);
		}

		closeFile(shp);

		return shp.type;
	}

	static int readFile(const std::string& fname, std::vector<std::vector<double>>& shapes) {
		const auto shp = openFile(fname);

		//file parameters
		const int numShapes = shp.sf->nRecords;

		//initialize containers
		shapes.resize(numShapes);

		//read shapes
		for (int i = 0; i < numShapes; i++) {
			readShape(shp, i, shapes[i]);
		}

		closeFile(shp);

		return shp.type;
	}

};