#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <boost/compute/system.hpp>
#include <boost/compute/interop/opencv/core.hpp>
#include <boost/compute/interop/opencv/highgui.hpp>
#include <boost/compute/utility/source.hpp>
using namespace boost::compute;
 char opencl_kernel_code[] = BOOST_COMPUTE_STRINGIZE_SOURCE (
		 __constant sampler_t sampler =	CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

 	 	 __kernel convert(__read_only image2d_t input, __write_only image2d_t output){

 	 	 }
		 );
int main(int argc, char* argv[]) {
	std::cout << "Kanten Projekt" << std::endl;
	cv::Mat cv_image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

	device gpu = boost::compute::system::default_device();
	context context(gpu);
	command_queue queue(context, gpu);

	cv::cvtColor(cv_image, cv_image, CV_BGR2BGRA);

	image2d input_image = opencv_create_image2d_with_mat(cv_image,
			image2d::read_only, queue);
	image2d output_image(context, input_image.width(), input_image.height(),
			input_image.format(), image2d::write_only);
	auto program_code = program::create_with_source(opencl_kernel_code,context);
	try{
		program_code.build();
	}
	catch(boost::compute::opencl_error e){
		std::cout << "Programm error" << program_code.build_log();
	}
	kernel lap_gau {program_code,"convert"};
	lap_gau.set_arg(0,input_image);
	lap_gau.set_arg(1,output_image);

    size_t origin[2] = { input_image.width(), input_image.height() };
	queue.enqueue_nd_range_kernel(lap_gau, 0,origin,nullptr, nullptr);

	opencv_imshow("test",output_image,queue);

	std::cout << "test" << std::endl;
	return 0;
}
