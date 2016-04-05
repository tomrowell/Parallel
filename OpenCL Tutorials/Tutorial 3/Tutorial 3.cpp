#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef string myType;
		typedef int intType;
		typedef float floatType;
		string loc, yy, mm, dd, tm, tp;
		/*std::vector<myType> entryLoc;
		std::vector<myType> entryYear;
		std::vector<myType> entryMonth;
		std::vector<myType> entryDay;
		std::vector<myType> entryTime;*/
		std::vector<intType> entryTemp;

		//uses file path to open the .txt data file
		string filePath = "C:\\Users\\Computing\\Documents\\GitHub\\Parallel\\OpenCL Tutorials\\x64\\Debug\\Files\\temp_lincolnshire.txt";
		int entryLength = 0; //variable to track the length of the data retroactively
		try {
			ifstream fileLoc(filePath);
			string fileLine;
			while (fileLoc >> loc >> yy >> mm >> dd >> tm >> tp) {
				/*entryLoc.push_back(loc);
				entryYear.push_back(yy);
				entryMonth.push_back(mm);
				entryDay.push_back(dd);
				entryTime.push_back(tm);*/
				entryTemp.push_back((stof(tp)) * 10); //reads the 6th element on each row of data as the temperature

				entryLength++;
				if (entryLength % 1000 == 0) {
					std::cout << "\rLine Count:" << (entryLength) << flush; //shows the user how many lines have currently been processed
				}
			}
			std::cout << "\rLine Count:" << (entryLength) << flush;
			std::cout << endl;
		}
		catch (const cl::Error& err) {
			std::cout << "File input error";
		}

		size_t entryElements = entryTemp.size(); //amount of elements in the temperature array
		size_t entryInputSize = entryTemp.size()*sizeof(intType); //size in bytes, used for buffer

		size_t entryGroupSize;
		size_t entryGroups;

		//finds the maximum amount of threads that can be used for the given set of data
		for (int i = 1024; i >= 1; i--)
		{
			if (entryLength % i == 0) {
				entryGroupSize = i;
				entryGroups = entryLength / entryGroupSize; //uses the amount of threads to determin the amount of work groups
				break;
			}
		}

		//creates the output vectors with a size 1 higher than there are work groups
		//the 0th element will be left empty for future use
		std::vector<intType> outTempMin(entryGroups + 1);
		std::vector<intType> outTempMax(entryGroups + 1);
		std::vector<floatType> outTempAvg(entryGroups + 1);
		size_t entryOutputSize = outTempMin.size()*sizeof(intType);

		//creates buffers for input & output vectors
		cl::Buffer bufferInTemp(context, CL_MEM_READ_ONLY, entryInputSize);
		cl::Buffer bufferOutTempMin(context, CL_MEM_READ_WRITE, entryOutputSize);
		cl::Buffer bufferOutTempMax(context, CL_MEM_READ_WRITE, entryOutputSize);
		cl::Buffer bufferOutTempAvg(context, CL_MEM_READ_WRITE, entryOutputSize);

		//writes and fills buffers
		queue.enqueueWriteBuffer(bufferInTemp, CL_TRUE, 0, entryInputSize, &entryTemp[0]);
		queue.enqueueFillBuffer(bufferOutTempMin, 0, 0, entryOutputSize);
		queue.enqueueFillBuffer(bufferOutTempMax, 0, 0, entryOutputSize);
		queue.enqueueFillBuffer(bufferOutTempAvg, 0, 0, entryOutputSize);

		//kernel for finding the minumum value
		cl::Kernel kernel_1 = cl::Kernel(program, "value_min");
		kernel_1.setArg(0, bufferInTemp);
		kernel_1.setArg(1, bufferOutTempMin);
		kernel_1.setArg(2, cl::Local(entryGroupSize*sizeof(myType)));

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(entryElements), cl::NDRange(entryGroupSize)); //call all kernels in a sequence
		queue.enqueueReadBuffer(bufferOutTempMin, CL_TRUE, 0, entryOutputSize, &outTempMin[0]); //Copy the results from device to host

		//kernel for finding the maximum value
		cl::Kernel kernel_2 = cl::Kernel(program, "value_max");
		kernel_2.setArg(0, bufferInTemp);
		kernel_2.setArg(1, bufferOutTempMax);
		kernel_2.setArg(2, cl::Local(entryGroupSize*sizeof(myType)));

		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(entryElements), cl::NDRange(entryGroupSize));
		queue.enqueueReadBuffer(bufferOutTempMax, CL_TRUE, 0, entryOutputSize, &outTempMax[0]);

		//kernel for summing all values, used later to find the average
		cl::Kernel kernel_3 = cl::Kernel(program, "value_avg");
		kernel_3.setArg(0, bufferInTemp);
		kernel_3.setArg(1, bufferOutTempAvg);
		kernel_3.setArg(2, cl::Local(entryGroupSize*sizeof(myType)));

		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(entryElements), cl::NDRange(entryGroupSize));
		queue.enqueueReadBuffer(bufferOutTempAvg, CL_TRUE, 0, entryOutputSize, &outTempAvg[0]);

		//sets the 0th element of the output vectors to the same as the 1st element
		outTempMin[0] = outTempMin[1];
		outTempMax[0] = outTempMax[1];
		outTempAvg[0] = outTempAvg[1];

		//starts from the 2nd element of each vector
		for (int i = 2; i <= entryGroups; i++) {
			//compares the 0th element of the minimum vector to each other element 
			if (outTempMin[0] > outTempMin[i]) //if the 0th element is larger than the current element
				outTempMin[0] = outTempMin[i]; //sets the 0th element to the same value as the current element

			//compares the 0th element of the maximum vector to each other element 
			if (outTempMax[0] < outTempMax[i]) //if the 0th element is lower than the current element
				outTempMax[0] = outTempMax[i]; //sets the 0th element to the same value as the current element

			outTempAvg[0] += outTempAvg[i]; //adds the current element to the 0th element to create the summed total
		}

		outTempAvg[0] /= entryElements; //divides the summed total of all the elements by the amount of elements to find the average

		//outputs the minimum, maximum and average values to the console
		std::cout << std::endl;
		std::cout << "Min = " << (float)outTempMin[0] / 10 << std::endl;
		std::cout << "Max = " << (float)outTempMax[0] / 10 << std::endl;
		std::cout << "Avg = " << outTempAvg[0] / 10 << std::endl;
		std::cout << endl;

		int histRange = (ceil((float)outTempMax[0] / 10)) - (floor((float)outTempMin[0] / 10)) + 1; //finds the range of values using the minimum and maximum values
		int histOffset = floor((float)outTempMin[0] / 10); //uses the minimum value as an offset so that all values can be above 0
		if (histOffset > 0) //checks to see if the offset is above 0
			histOffset = 0;	//sets the offset to 0, as it is unnecessary in this scenario

		std::vector<intType> hist(histRange); //creates a new vector for the histogram
		size_t histSize = hist.size()*sizeof(intType); //size in bytes

		//buffer for histogram
		cl::Buffer bufferHist(context, CL_MEM_READ_WRITE, histSize);
		queue.enqueueFillBuffer(bufferHist, 0, 0, histSize);

		//kernel for histogram
		cl::Kernel kernel_4 = cl::Kernel(program, "value_hist");
		kernel_4.setArg(0, bufferInTemp);
		kernel_4.setArg(1, bufferHist);
		kernel_4.setArg(2, histRange);
		kernel_4.setArg(3, histOffset);

		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(entryElements), cl::NDRange(entryGroupSize));
		queue.enqueueReadBuffer(bufferHist, CL_TRUE, 0, histSize, &hist[0]);

		int histStart = 0;
		int histEnd = histRange;

		//finds the lowest percentile for the histogram
		int histCount = 0;
		for (int i = 0; i < histRange; i++) {
			histCount += hist[i]; //sums all values that represent the first ~1% of data
			if (histCount > (entryElements / 100)) {
				hist[i] = histCount; //adds all summed values into the lowest relevent bin
				histStart = i; //records which bin the lowest relevent bin
				break;
			}
		}

		//finds the highest percentile for the histogram
		histCount = 0;
		for (int i = histRange - 1; i >= 0; i--) {
			histCount += hist[i]; //sums all values that represent the last ~1% of data
			if (histCount > (entryElements / 100)) {
				hist[i] = histCount; //adds all summed values into the highest relevent bin
				histEnd = i; //records which bin the highest relevent bin
				break;
			}
		}

		//outputs the histogram
		std::cout << "Temperature Histogram" << endl;
		for (int i = histStart; i <= histEnd; i++) { //uses the lowest and highest relevant bins to condense all outlying data into the first and last bins
			if (i == histStart)
				std::cout << "<="; //adds the "<=" symbol to the first bin of the histogram to show that it also represents outlying values 
			std::cout << (i + histOffset);
			if (i == histEnd)
				std::cout << "+"; //adds the "+" symbol to the last bin of the histogram to show that it also represents outlying values
			std::cout << ": ";
			std::cout << hist[i] << endl;
		}
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
