#pragma once
#include "pch.h"
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <iostream>
#include <string>
#include <array>
#include <vector>

#include <io.h>
#include "Utils.h"
#define streamDup(fd1) _dup(fd1)
#define streamDup2(fd1,fd2) _dup2(fd1,fd2)
//#define _CRT_SECURE_NO_WARNINGS

using namespace Utils;

class OnnxModel{
	
protected:
	//Onnx session boiler plate
	Ort::Env env;
	Ort::RunOptions runOptions;
	Ort::Session session = Ort::Session(nullptr);
	std::array<const char*, 1> inputNames;
	std::array<const char*, 1> outputNames;
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	int batchSize;
	bool* cancelFlag;
	std::string modelName;

	//input / output tensors of model, must be initialized in subclass
	//These are intermediates that reference underlying containers which have the actual data
	Ort::Value inputTensor = Ort::Value(nullptr);
	Ort::Value outputTensor = Ort::Value(nullptr);

	std::vector<int64_t> inputShape;
	std::vector<int64_t> outputShape;

	//Actual input / output containers
	std::vector<float> input;
	std::vector<float> output;

private:

	int m_old_stdout;
	int m_old_stderr;
	void redirectStandardOutputs(char const* stdout_name, char const* stderr_name) {
		(void)fflush(stdout);
		(void)fflush(stderr);
		m_old_stdout = _dup(_fileno(stdout));
		m_old_stderr = _dup(_fileno(stderr));
		(void)freopen(stdout_name, "wb", stdout); //cast to void means discard return value
		(void)freopen(stderr_name, "wb", stderr);
	}

	void restoreStandardOutputs() const {
		(void)fflush(stdout);
		(void)fflush(stderr);
		(void)_dup2(m_old_stdout, _fileno(stdout));
		(void)_dup2(m_old_stderr, _fileno(stderr));
	}

	static bool attemptToUseCuda(OrtApi const& ortApi, const Ort::SessionOptions& sessionOptions)
	{
		try {
			//setup CUDA options
			OrtCUDAProviderOptions options;
			options.device_id = 0;
			options.arena_extend_strategy = 0;
			options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
			options.do_copy_in_default_stream = 1;

			OrtTensorRTProviderOptions trt_options{};
			trt_options.device_id = 0;
			trt_options.trt_max_workspace_size = 8589934592; //8GB
			trt_options.trt_max_partition_iterations = 10;
			trt_options.trt_min_subgraph_size = 5;
			trt_options.trt_engine_cache_enable = 1;
			trt_options.trt_engine_cache_path = "models/cache";
			//trt_options.trt_dump_subgraphs = 1;

			//attempt to use CUDA
			Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_TensorRT(sessionOptions, &trt_options));
			Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_CUDA(sessionOptions, &options));
		}
		catch (Ort::Exception& _) {
			(void)_;
			return false;
		}

		return true;
	}

	static bool attemptToUseROCm(OrtApi const& ortApi, Ort::SessionOptions& sessionOptions) {
		try {
			OrtROCMProviderOptions rocmOptions;
			rocmOptions.device_id = 0;

			sessionOptions.AppendExecutionProvider_ROCM(rocmOptions);
			//Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_ROCM(sessionOptions, &rocmOptions));
		}
		catch (Ort::Exception& e) {
			std::cout << e.what() << '\n';
			return false;
		}

		return true;
	}

	static bool attemptToUseDML(OrtApi const& ortApi, Ort::SessionOptions &sessionOptions) {
		try {
			//setup DirectML (DirectX12) options
			OrtDmlApi const* ortDmlApi = nullptr;
			ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<void const**>(&ortDmlApi));

			sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // For DML EP
			sessionOptions.DisableMemPattern(); // For DML EP

			Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, /*device index*/ 0));
		}
		catch (Ort::Exception& e) {
			
			printState(e.what());

			return false;
		}
		
		return true;
	}

	void attemptToUseGPU(OrtApi const& ortApi, Ort::SessionOptions& sessionOptions) const {
		std::string provider;

		if (attemptToUseCuda(ortApi, sessionOptions)) {
			provider = "Running on GPU, TensorRT(Cuda) ENABLED ";
		}
		else if (attemptToUseDML(ortApi, sessionOptions)) {
			provider = "Running on GPU, DirectML ENABLED";
		}
		else {
			provider = "Running on CPU, No GPU found/supported";
		}

		//std::cout << "Initializing " << modelName << " Model - " << provider << '\n' << '\n' << std::flush;
		printState("Initializing " + modelName + " Model - " + provider);
	}


public:

	OnnxModel(const wchar_t* modelPath, const std::string& modelName, const int batchSize = 16) {

		this->batchSize = batchSize;
		this->cancelFlag = cancelFlag;
		this->modelName = modelName;

		const Ort::Env env = Ort::Env{ ORT_LOGGING_LEVEL_FATAL, "test" };

		Ort::SessionOptions sessionOptions = Ort::SessionOptions();
		sessionOptions.SetLogSeverityLevel(4);
		runOptions.SetRunLogSeverityLevel(4);
		runOptions.SetRunLogVerbosityLevel(4);


		OrtApi const& ortApi = Ort::GetApi();

		attemptToUseGPU(ortApi, sessionOptions);

		//setting a constant batch size can improve performance
		ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", batchSize);

		//to get onnx to shut up with the unless warnings...
		//redirectStandardOutputs("NUL", "NUL");

		session = Ort::Session(env, modelPath, sessionOptions);

		//restoreStandardOutputs();

		//default boiler plate
		const Ort::AllocatorWithDefaultOptions ort_alloc;
		Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
		Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
		inputNames = { inputName.get() };
		outputNames = { outputName.get() };
		(void)inputName.release();
		(void)outputName.release();

	}

	void initializeTensors(const int inputSize, const int predicitionSize) {
		input.resize(batchSize * inputSize);
		output.resize(batchSize * predicitionSize);

		inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
		outputTensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), outputShape.data(), outputShape.size());
	}

	void predict() {

		Utils::checkCancelled();

		session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
	}
};

