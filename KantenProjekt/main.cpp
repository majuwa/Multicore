#include <iostream>
#include <CL/cl.hpp>
#include "opencl-helper.hpp"
#include "images.h"
#include <random>

#include <string>

int main(int argc, char* argv[]) {
	std::cout << "Kanten Projekt" << std::endl;
	//cv::Mat cv_image = cv::imread("test.jpg", CV_LOAD_IMAGE_COLOR);

	auto default_device = initialize_gpu();
	auto list { default_device };
	cl::Context context(list);

	cl::Program::Sources sources;
	std::string kernel_code = get_file_contents("kernel.txt");
	sources.push_back( { kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);

	build_kernel(kernel_code, default_device, program);
	image_t t;
	loadImage("eingabe.bmp", &t);

	cl::ImageFormat format;
	format.image_channel_data_type = CL_UNSIGNED_INT8;
	format.image_channel_order = CL_RGBA;
	cl_int er;
	cl::Image2D image { cl::Image2D(context,
	CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, t.width, t.height, 0,
			t.pixels, &er) };
	if (er == CL_SUCCESS)
		std::cout << "Success loading image" << std::endl;
	cl::Image2D image_2 { cl::Image2D(context, CL_MEM_WRITE_ONLY, format,
			t.width, t.height, 0, nullptr, &er) };
	if (er == CL_SUCCESS)
		std::cout << "Success loading result image" << std::endl;
	cl::CommandQueue queue(context, default_device);
	cl::Kernel kernel_add = cl::Kernel(program, "find_edges", &er);
	if (er == CL_SUCCESS)
		std::cout << "kernel successfull loaded" << std::endl;
	er = kernel_add.setArg(0, image);
	if (er == CL_SUCCESS)
		std::cout << "created param 1" << std::endl;
	er = kernel_add.setArg(1, image_2);
	if (er == CL_SUCCESS)
		std::cout << "created param 2" << std::endl;

	er = queue.enqueueNDRangeKernel(kernel_add, cl::NullRange,
			cl::NDRange(t.width, t.height), cl::NullRange);
	if (er == CL_SUCCESS)
		std::cout << "created kernel enviremont successfully" << std::endl;
	queue.finish();
	if (er == CL_SUCCESS)
		std::cout << "executed successfully" << std::endl;
	unsigned char* test_array = new unsigned char[t.height * t.width * t.bpp];
	cl::size_t<int(3)> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	cl::size_t<int(3)> region;
	region[0] = t.width;
	region[1] = t.height;
	region[2] = 1;
	er = queue.enqueueReadImage(image_2, CL_TRUE, origin, region,
			sizeof(unsigned char) * t.width * t.bpp, 0, (void*) test_array,
			nullptr, nullptr);
	if (er == CL_SUCCESS)
		std::cout << "copying in output image" << std::endl;
	image_t i2;
	i2.height = t.height;
	i2.width = t.width;
	i2.bpp = t.bpp;
	i2.pixels = test_array;
	saveImage("test2.bmp", i2);
	delete t.pixels;
	delete test_array;
	//opencv_imshow("test",output_image,queue);

	std::cout << "test" << std::endl;
	return 0;
}
