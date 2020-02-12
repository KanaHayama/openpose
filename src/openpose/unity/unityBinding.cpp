#ifdef USE_UNITY_SUPPORT
// ------------------------- OpenPose Unity Binding -------------------------

// OpenPose dependencies
#include <openpose/headers.hpp>
#include "opencv2/core/mat.hpp"

namespace op
{
    #ifdef _WIN32
        // Output callback register in Unity
        typedef void(__stdcall * OutputCallback) (void * ptrs, int ptrSize, int * sizes, int sizeSize,
                                                  unsigned char outputType);
        // Global output callback
        OutputCallback sUnityOutputCallback;
    #endif

    // Other global parameters
    bool sMultiThreadDisabled = false;
    bool sUnityOutputEnabled = true;
    bool sImageOutput = false;

    enum class OutputType : unsigned char
    {
        None,
        DatumsInfo,
        Name,
        PoseKeypoints,
        PoseIds,
        PoseScores,
        PoseHeatMaps,
        PoseCandidates,
        FaceRectangles,
        FaceKeypoints,
        FaceHeatMaps,
        HandRectangles,
        HandKeypoints,
        HandHeightMaps,
        PoseKeypoints3D,
        FaceKeypoints3D,
        HandKeypoints3D,
        CameraMatrix,
        CameraExtrinsics,
        CameraIntrinsics,
        Image
    };

	// next frame
	std::mutex FrameLock;
	unsigned long long FrameNumber;
	std::string FrameName;
	Matrix Frame;
	// frame buffer
	std::mutex BufferLock;
	Matrix Buffer;
	// input worker producer
	class UnityPluginUserInput : public WorkerProducer<std::shared_ptr<std::vector<std::shared_ptr<Datum>>>> {
	private:
	protected:
		void initializationOnThread() override {}

		std::shared_ptr<std::vector<std::shared_ptr<Datum>>> workProducer() override {// see datumProducer.hpp & webcamReader.cpp & videoCaptureReader.cpp
			auto datums = std::make_shared<std::vector<std::shared_ptr<Datum>>>();
			datums->emplace_back(std::make_shared<Datum>());// TODO: what will happen if we do not add a default Datum if the image is not ready
			std::unique_lock<std::mutex> lock(FrameLock);
			if (!Frame.empty()) {
				Datum& datum = *datums->at(0);
				std::swap(datum.name, FrameName);
				datum.frameNumber = FrameNumber;
				auto& matrix = datum.cvInputData;
				std::swap(matrix, Frame);
				datum.cvOutputData = datum.cvInputData;
			} else {
				lock.unlock();
				std::this_thread::sleep_for(std::chrono::microseconds{ 5 });
			}
			return datums;
		}
	public:
		// step 1 
		unsigned char* allocateNewFrameBuffer(const int width, const int height) {
			try {
				const std::lock_guard<std::mutex> lock(BufferLock);
				// each pixel 3 channels byte (BGR)
				// default stride (number of bytes) is calculated as width * elemSize() = 3 * width
				// allocate new memory
				auto frame = cv::Mat(height, width, CV_8UC3);
				Buffer = OP_CV2OPMAT(frame);
				return static_cast<cv::Mat*>(Buffer.getCvMat())->data;
			} catch (const std::exception& e) {
				errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}

		// step 2
		void postNewFrame() {
			try {
				const std::lock_guard<std::mutex> lock(BufferLock);
				if (!Buffer.empty()) {
					const std::lock_guard<std::mutex> lock2(FrameLock);
					std::swap(Frame, Buffer);
				}
			} catch (const std::exception& e) {
				errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}
	};

    // This worker will just read and return all the jpg files in a directory
    class UnityPluginUserOutput : public WorkerConsumer<std::shared_ptr<std::vector<std::shared_ptr<Datum>>>>
    {
    public:
        void initializationOnThread()
        {
        }

        void workConsumer(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                if (datumsPtr != nullptr && !datumsPtr->empty())
                {
                    if (sUnityOutputEnabled)
                    {
                        sendDatumsInfoAndName(datumsPtr);
                        sendPoseKeypoints(datumsPtr);
                        sendPoseIds(datumsPtr);
                        sendPoseScores(datumsPtr);
                        sendPoseHeatMaps(datumsPtr);
                        sendPoseCandidates(datumsPtr);
                        sendFaceRectangles(datumsPtr);
                        sendFaceKeypoints(datumsPtr);
                        sendFaceHeatMaps(datumsPtr);
                        sendHandRectangles(datumsPtr);
                        sendHandKeypoints(datumsPtr);
                        sendHandHeatMaps(datumsPtr);
                        if (sImageOutput)
                            sendImage(datumsPtr);
                        sendEndOfFrame();
                    }
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

    private:
        template<class T>
        void outputValue(T** ptrs, const int ptrSize, int* sizes, const int sizeSize, const OutputType outputType)
        {
            try
            {
                #ifdef _WIN32
                    if (sUnityOutputCallback)
                        sUnityOutputCallback(
                            static_cast<void*>(ptrs), ptrSize, sizes, sizeSize, (unsigned char)outputType);
                #else
                    UNUSED(ptrs);
                    UNUSED(ptrSize);
                    UNUSED(sizes);
                    UNUSED(sizeSize);
                    UNUSED(outputType);
                    error("Function only available on Windows.", __LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendDatumsInfoAndName(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& datum = datumsPtr->at(0);
                int sizes[] = { 1 };
                const int sizeSize = 1;
                unsigned long long* val[] = {&(datum->id), &(datum->subId), &(datum->subIdMax), &(datum->frameNumber)};
                const int ptrSize = 4;
                outputValue(&val[0], ptrSize, &sizes[0], sizeSize, OutputType::DatumsInfo);

                const char* a[] = { datum->name.c_str() };
                outputValue(&a[0], 1, &sizes[0], sizeSize, OutputType::Name);
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendPoseKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->poseKeypoints; // Array<float>
                if (!data.empty())
                {
                    auto sizeVector = data.getSize();
                    const int sizeSize = (int)sizeVector.size();
                    int* sizes = &sizeVector[0];
                    float* val = data.getPtr();
                    outputValue(&val, 1, sizes, sizeSize, OutputType::PoseKeypoints);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendPoseIds(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->poseIds; // Array<long long>
                if (!data.empty())
                {
                    auto sizeVector = data.getSize();
                    const int sizeSize = (int)sizeVector.size();
                    int* sizes = &sizeVector[0];
                    long long* val = data.getPtr();
                    outputValue(&val, 1, sizes, sizeSize, OutputType::PoseIds);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendPoseScores(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->poseScores; // Array<float>
                if (!data.empty())
                {
                    auto sizeVector = data.getSize();
                    const int sizeSize = (int)sizeVector.size();
                    int* sizes = &sizeVector[0];
                    float* val = data.getPtr();
                    outputValue(&val, 1, sizes, sizeSize, OutputType::PoseScores);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendPoseHeatMaps(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->poseHeatMaps; // Array<float>
                if (!data.empty())
                {
                    auto sizeVector = data.getSize();
                    const int sizeSize = (int)sizeVector.size();
                    int* sizes = &sizeVector[0];
                    float* val = data.getPtr();
                    outputValue(&val, 1, sizes, sizeSize, OutputType::PoseHeatMaps);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendPoseCandidates(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->poseCandidates; // std::vector<std::vector<std::array<float, 3>>>
                if (!data.empty())
                {
                    // TODO
                    /*auto a = data[0][0].data();
                    auto sizeVector = data.getSize();
                    int sizeSize = sizeVector.size();
                    int * sizes = &sizeVector[0];
                    long long * val = data.getPtr();
                    outputValue(&val, 1, sizes, sizeSize, OutputType::PoseIds);*/
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendFaceRectangles(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->faceRectangles; // std::vector<Rectangle<float>>
                if (data.size() > 0)
                {
                    int sizes[] = { (int)data.size(), 4 };
                    std::vector<float> vals(data.size() * 4);
                    for (auto i = 0u; i < data.size(); i++)
                    {
                        vals[4 * i + 0] = data[i].x;
                        vals[4 * i + 1] = data[i].y;
                        vals[4 * i + 2] = data[i].width;
                        vals[4 * i + 3] = data[i].height;
                    }
                    float * val = &vals[0];
                    outputValue(&val, 1, sizes, 2, OutputType::FaceRectangles);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendFaceKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->faceKeypoints; // Array<float>
                if (!data.empty())
                {
                    auto sizeVector = data.getSize();
                    int sizeSize = (int)sizeVector.size();
                    int * sizes = &sizeVector[0];
                    float * val = data.getPtr();
                    outputValue(&val, 1, sizes, sizeSize, OutputType::FaceKeypoints);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendFaceHeatMaps(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->faceHeatMaps; // Array<float>
                if (!data.empty())
                {
                    auto sizeVector = data.getSize();
                    int sizeSize = (int)sizeVector.size();
                    int * sizes = &sizeVector[0];
                    float * val = data.getPtr();
                    outputValue(&val, 1, sizes, sizeSize, OutputType::FaceHeatMaps);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendHandRectangles(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->handRectangles; // std::vector<std::array<Rectangle<float>, 2>>
                if (!data.empty())
                {
                    std::vector<float*> valPtrs;
                    for (auto i = 0u; i < data.size(); i++)
                    {
                        float vals[8];
                        for (auto j = 0; j < 2; j++)
                        {
                            vals[4 * j + 0] = data[i][j].x;
                            vals[4 * j + 1] = data[i][j].y;
                            vals[4 * j + 2] = data[i][j].width;
                            vals[4 * j + 3] = data[i][j].height;
                        }
                        valPtrs.push_back(vals);
                    }
                    int sizes[] = {2, 4};
                    const int sizeSize = 2;
                    outputValue(valPtrs.data(), (int)valPtrs.size(), sizes, sizeSize, OutputType::HandRectangles);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendHandKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->handKeypoints; // std::array<Array<float>, 2>
                if (data.size() == 2 && !data[0].empty())
                {
                    auto sizeVector = data[0].getSize();
                    const int sizeSize = (int)sizeVector.size();
                    int* sizes = &sizeVector[0];
                    float* ptrs[] = { data[0].getPtr(), data[1].getPtr() };
                    outputValue(ptrs, 2, sizes, sizeSize, OutputType::HandKeypoints);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendHandHeatMaps(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->handHeatMaps; // std::array<Array<float>, 2>
                if (data.size() == 2 && !data[0].empty())
                {
                    auto sizeVector = data[0].getSize();
                    const int sizeSize = (int)sizeVector.size();
                    int* sizes = &sizeVector[0];
                    float* ptrs[] = { data[0].getPtr(), data[1].getPtr() };
                    outputValue(ptrs, 2, sizes, sizeSize, OutputType::HandHeightMaps);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendImage(const std::shared_ptr<std::vector<std::shared_ptr<Datum>>>& datumsPtr)
        {
            try
            {
                auto& data = datumsPtr->at(0)->cvInputData; // cv::Mat
                if (!data.empty())
                {
                    int sizeVector[] = { data.rows(), data.cols(), 3 };
                    const int sizeSize = 3;
                    auto valPtr = data.data();
                    outputValue(&valPtr, 1, sizeVector, sizeSize, OutputType::Image);
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void sendEndOfFrame()
        {
            try
            {
                outputValue((void**)nullptr, 0, nullptr, 0, OutputType::None);
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    };

	// Global user input
	UnityPluginUserInput*  ptrUserInput = nullptr;

    // Global user output
    UnityPluginUserOutput* ptrUserOutput = nullptr;

    // Global setting structs
    std::shared_ptr<WrapperStructPose> spWrapperStructPose;
    std::shared_ptr<WrapperStructHand> spWrapperStructHand;
    std::shared_ptr<WrapperStructFace> spWrapperStructFace;
    std::shared_ptr<WrapperStructExtra> spWrapperStructExtra;
    std::shared_ptr<WrapperStructInput> spWrapperStructInput;
    std::shared_ptr<WrapperStructOutput> spWrapperStructOutput;
    std::shared_ptr<WrapperStructGui> spWrapperStructGui;

    // Main
    void openpose_main()
    {
        try
        {
            // Starting
            opLog("Starting OpenPose...");

            // OpenPose wrapper
            auto spWrapper = std::make_shared<Wrapper>();

            // Initializing the user custom classes
			auto spUserInput = std::make_shared<UnityPluginUserInput>();
			ptrUserInput = spUserInput.get();
            auto spUserOutput = std::make_shared<UnityPluginUserOutput>();
            ptrUserOutput = spUserOutput.get();

            // Add custom processing
			if (spWrapperStructInput->producerType == ProducerType::None) {
				const auto workerInputOnNewThread = true; // TODO: should it be false to increase performance?
				spWrapper->setWorker(WorkerType::Input, spUserInput, workerInputOnNewThread);
			}
            const auto workerOutputOnNewThread = true;
            spWrapper->setWorker(WorkerType::Output, spUserOutput, workerOutputOnNewThread);

            // Apply configurations
            spWrapper->configure(*spWrapperStructPose);
            spWrapper->configure(*spWrapperStructHand);
            spWrapper->configure(*spWrapperStructFace);
            spWrapper->configure(*spWrapperStructExtra);
            spWrapper->configure(*spWrapperStructInput);
            spWrapper->configure(*spWrapperStructOutput);

            // Multi-threading
            if (sMultiThreadDisabled)
                spWrapper->disableMultiThreading();

            // Processing...
            spWrapper->exec();

            // Ending
            opLog("OpenPose finished");
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    // Functions called from Unity
    extern "C" {
#ifdef _WIN32
		OP_API unsigned char* _OPAllocateNewFrameBuffer(int width, int height) {
			try {
				return ptrUserInput->allocateNewFrameBuffer(width, height);
			} catch (const std::exception& e) {
				errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}
		OP_API void _OPPostNewFrame() {
			try {
				ptrUserInput->postNewFrame();
			} catch (const std::exception& e) {
				errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}
#endif

        // Start openpose safely
        OP_API void _OPRun()
        {
            try
            {
                openpose_main();
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        // Stop openpose safely
        OP_API void _OPShutdown()
        {
            try
            {
                if (ptrUserOutput != nullptr)
                {
                    opLog("Stopping...");
                    ptrUserOutput->stop();
                }
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        // Register Unity output callback function
        #ifdef _WIN32
            OP_API void _OPRegisterOutputCallback(OutputCallback callback)
            {
                try
                {
                    sUnityOutputCallback = callback;
                }
                catch (const std::exception& e)
                {
                    errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }
        #endif

        // Enable/disable output callback
        OP_API void _OPSetOutputEnable(bool enable)
        {
            try
            {
                sUnityOutputEnabled = enable;
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        // Enable/disable image output
        OP_API void _OPSetImageOutputEnable(bool enable)
        {
            try
            {
                sImageOutput = enable;
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        // Configs
        OP_API void _OPConfigurePose(
            unsigned char poseMode,
            int netInputSizeX, int netInputSizeY, // Point
            int outputSizeX, int outputSizeY, // Point
            unsigned char keypointScaleMode, // ScaleMode
            int gpuNumber, int gpuNumberStart, int scalesNumber, float scaleGap,
            unsigned char renderMode, // RenderMode
            unsigned char poseModel, // PoseModel
            bool blendOriginalFrame, float alphaKeypoint, float alphaHeatMap, int defaultPartToRender,
            char* modelFolder, bool heatMapAddParts, bool heatMapAddBkg,
            bool heatMapAddPAFs, // HeatMapType // unsigned char heatmap_type,
            unsigned char heatMapScaleMode, // ScaleMode
            bool addPartCandidates, float renderThreshold, int numberPeopleMax,
            bool maximizePositives, double fpsMax, char* protoTxtPath, char* caffeModelPath, float upsamplingRatio)
        {
            try
            {
                spWrapperStructPose = std::make_shared<WrapperStructPose>(
                    (PoseMode)poseMode, Point<int>{netInputSizeX, netInputSizeY}, Point<int>{outputSizeX, outputSizeY},
                    (ScaleMode) keypointScaleMode, gpuNumber, gpuNumberStart, scalesNumber, scaleGap,
                    (RenderMode) renderMode, (PoseModel) poseModel, blendOriginalFrame, alphaKeypoint, alphaHeatMap,
                    defaultPartToRender, modelFolder,
                    // HeatMapType // (HeatMapType) heatmap_type,
                    flagsToHeatMaps(heatMapAddParts, heatMapAddBkg, heatMapAddPAFs),
                    (ScaleMode) heatMapScaleMode, addPartCandidates, renderThreshold, numberPeopleMax,
                    maximizePositives, fpsMax, protoTxtPath, caffeModelPath, upsamplingRatio, true
                );
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        OP_API void _OPConfigureHand(
            bool enable, unsigned char detector, int netInputSizeX, int netInputSizeY, // Point
            int scalesNumber, float scaleRange,
            unsigned char renderMode, // RenderMode
            float alphaKeypoint, float alphaHeatMap, float renderThreshold)
        {
            try
            {
                spWrapperStructHand = std::make_shared<WrapperStructHand>(
                    enable, (Detector) detector, Point<int>{ netInputSizeX, netInputSizeY }, scalesNumber, scaleRange,
                    (RenderMode) renderMode, alphaKeypoint, alphaHeatMap, renderThreshold);
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        OP_API void _OPConfigureFace(
            bool enable, unsigned char detector, int netInputSizeX, int netInputSizeY, // Point
            unsigned char renderMode, // RenderMode
            float alphaKeypoint, float alphaHeatMap, float renderThreshold)
        {
            try
            {
                spWrapperStructFace = std::make_shared<WrapperStructFace>(
                    enable, (Detector) detector, Point<int>{ netInputSizeX, netInputSizeY }, (RenderMode) renderMode,
                    alphaKeypoint, alphaHeatMap, renderThreshold
                );
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        OP_API void _OPConfigureExtra(
            bool reconstruct3d, int minViews3d, bool identification, int tracking, int ikThreads)
        {
            try
            {
                spWrapperStructExtra = std::make_shared<WrapperStructExtra>(
                    reconstruct3d, minViews3d, identification, tracking, ikThreads
                );
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        OP_API void _OPConfigureInput(
            unsigned char producerType, char* producerString, // ProducerType
            unsigned long long frameFirst, unsigned long long frameStep, unsigned long long frameLast,
            bool realTimeProcessing, bool frameFlip, int frameRotate, bool framesRepeat,
            int cameraResolutionX, int cameraResolutionY, // Point
            char* cameraParameterPath, bool undistortImage, int numberViews)
        {
            try
            {
                spWrapperStructInput = std::make_shared<WrapperStructInput>(
                    (ProducerType) producerType, producerString, frameFirst, frameStep, frameLast, realTimeProcessing,
                    frameFlip, frameRotate, framesRepeat, Point<int>{ cameraResolutionX, cameraResolutionY },
                    cameraParameterPath, undistortImage, numberViews
                );
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        OP_API void _OPConfigureOutput(
            double verbose, char* writeKeypoint, unsigned char writeKeypointFormat, // DataFormat
            char* writeJson, char* writeCocoJson, int writeCocoJsonVariants, int writeCocoJsonVariant, char* writeImages,
            char* writeImagesFormat, char* writeVideo, double writeVideoFps, bool writeVideoWithAudio,
            char* writeHeatMaps, char* writeHeatMapsFormat, char* writeVideo3D, char* writeVideoAdam, char* writeBvh,
            char* udpHost, char* udpPort)
        {
            try
            {
                spWrapperStructOutput = std::make_shared<WrapperStructOutput>(
                    verbose, writeKeypoint, (DataFormat) writeKeypointFormat, writeJson, writeCocoJson,
                    writeCocoJsonVariants, writeCocoJsonVariant, writeImages, writeImagesFormat, writeVideo, writeVideoFps,
                    writeVideoWithAudio, writeHeatMaps, writeHeatMapsFormat, writeVideo3D, writeVideoAdam, writeBvh,
                    udpHost, udpPort);
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        OP_API void _OPConfigureGui(
            unsigned short displayMode, // DisplayMode
            bool guiVerbose, bool fullScreen)
        {
            try
            {
                spWrapperStructGui = std::make_shared<WrapperStructGui>(
                    (DisplayMode) displayMode, guiVerbose, fullScreen);
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        OP_API void _OPConfigureDebugging(
            unsigned char loggingLevel, // Priority
            bool disableMultiThread,
            unsigned long long profileSpeed)
        {
            try
            {
                ConfigureLog::setPriorityThreshold((Priority)loggingLevel);
                sMultiThreadDisabled = disableMultiThread;
                Profiler::setDefaultX(profileSpeed);
            }
            catch (const std::exception& e)
            {
                errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    }
}
#endif
