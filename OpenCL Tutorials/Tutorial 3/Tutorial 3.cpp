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

	for (int i = 1; i < argc; i++)	{
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
		typedef float floatType;
		string loc, yy, mm, dd, tm, tp;
		std::vector<myType> entryLoc;
		std::vector<myType> entryYear;
		std::vector<myType> entryMonth;
		std::vector<myType> entryDay;
		std::vector<myType> entryTime;
		std::vector<floatType> entryTemp;

		int entryLength = 0;
		try {
			ifstream fileLoc("C:\\Users\\Computing\\Documents\\GitHub\\Parallel\\OpenCL Tutorials\\x64\\Debug\\Files\\temp_lincolnshire_short.txt");
			string fileLine;
			while (fileLoc >> loc >> yy >> mm >> dd >> tm >> tp) {
				entryLoc.push_back(loc);
				entryYear.push_back(yy);
				entryMonth.push_back(mm);
				entryDay.push_back(dd);
				entryTime.push_back(tm);
				entryTemp.push_back(stof(tp));

				entryLength++;
				if (entryLength % 1000 == 0) {
					std::cout << entryLength << endl;
				}
			}
		}
		catch (const cl::Error& err) {
			std::cout << "File input error";
		}

		size_t entryElements = entryTemp.size(); 
		size_t entryInputSize = entryTemp.size()*sizeof(floatType);//size in bytes
		//size_t entryGroups = entryElements / entryLocalSize;

		size_t entryGroupSize;
		size_t entryGroups;

		//finds the maximum amount of threads that can be used for the given set of data
		for (int i = 1024; i >= 1; i--)
		{
			if (entryLength % i == 0) {
				entryGroupSize = i;
				entryGroups = entryLength / entryGroupSize;
				break;
			}
		}

		std::vector<floatType> outTemp(entryGroupSize);
		size_t entryOutputSize = outTemp.size()*sizeof(floatType);//size in bytes

		cl::Buffer bufferTempIn(context, CL_MEM_READ_ONLY, entryInputSize);
		cl::Buffer bufferTempOut(context, CL_MEM_READ_WRITE, entryOutputSize);

		queue.enqueueWriteBuffer(bufferTempIn, CL_TRUE, 0, entryInputSize, &entryTemp[0]);
		queue.enqueueFillBuffer(bufferTempOut, 0, 0, entryOutputSize);//zero B buffer on device memory

		cl::Kernel kernel_1 = cl::Kernel(program, "value_min");
		kernel_1.setArg(0, bufferTempIn);
		kernel_1.setArg(1, bufferTempOut);
		kernel_1.setArg(2, cl::Local(entryGroupSize*sizeof(myType)));//local memory size

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(entryElements), cl::NDRange(entryGroupSize));//call kernels for the first work group in a sequence

		/*if (entryGroups > 1) {
			for (int i = 1; i < entryGroups; i++) {
				queue.enqueueNDRangeKernel(kernel_1, cl::NDRange(i*entryGroupSize), cl::NDRange(entryElements), cl::NDRange(entryGroupSize));
			}
		}*/

		queue.enqueueReadBuffer(bufferTempOut, CL_TRUE, 0, entryOutputSize, &outTemp[0]);//Copy the result from device to host

		//std::cout << "A = " << entryTemp << std::endl;
		std::cout << "B = " << outTemp << std::endl;

		/*typedef int mytype;

		//Part 4 - memory allocation
		//host - input
		std::vector<mytype> A(10, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 10;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		std::vector<mytype> B(input_elements);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_1");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
//		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;*/
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
