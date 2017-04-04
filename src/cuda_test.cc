/*
* @File: cuda_test
* @Author: kmrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-10-15 10:11:19
* @Last Modified by:   kmrocki
* @Last Modified time: 2015-10-15 10:11:19
*/

#include <iostream>
#include <tensor.h>
#include <rbm.h>
#include <sp.h>
#include <matrix.h>
#include <vector.h>
#include <rand.h>
#include <timer.h>
#include <thread>
#include <font.h>
#include <imageimporter.h>
#include <glWindow.h>
#include <gpu_rbm.h>

void test(void) {

	Matrix<float>* main_display;
	Timer t;

	std::vector<Matrix<float>*> inputs = ImageImporter::importFromFile("data/mnist/train-images-idx3-ubyte", 28, 28, 16);
	Matrix<float> graph(1, 1);
	Matrix<float> graph2(1, 1);
	graph.display_mode = DISPLAY_MODE::GRAPH;

	size_t patch_size = 28;
	float err_sum = .0f;
	float hact_sum = .0f;
	float h2_sum = .0f;
	float time_per_iteration;
	size_t record_interval = 1000;
	size_t num_h0 = 1024;
	size_t batchsize = 16;

	if (inputs.size() > 0) {

		GPU_RBM h0(patch_size * patch_size, num_h0, NTYPE::BINARY, NTYPE::BINARY, LTYPE::CD, DISPLAY_MODE::UNFLATTEN, batchsize, 0.05f, 0.00001f, 0.00001f);

		Matrix<float> patch(patch_size, patch_size);

		main_display = new Matrix<float>(patch_size, patch_size);
		GLWindowRenderer::set(1, h0.v);
		GLWindowRenderer::set(2, h0.vn);
		GLWindowRenderer::set(3, h0.pc);
		GLWindowRenderer::set(4, h0.W);
		GLWindowRenderer::set(5, h0.H);
		GLWindowRenderer::set(6, &graph);
		//GLWindowRenderer::set(7, &graph2);

		t.start();

		for (size_t iteration = 0;; iteration++) {

			if (!_pause || _step) {

				_step = false;

				display_mutex.lock();

				for (size_t b = 0; b < batchsize; b++) {

					int num = MT19937::randint() % inputs.size();
					Matrix<float>::randsubmatrix(*(inputs[num]), patch);
					//MATRIX_MEMCPY(main_display, &patch);
					MATRIX_MEMCPY_ROW(h0.v, &patch, b);
					
				}

				h0.up();
				h0.down();
				h0.adapt();
				//h0.compute_statistics();

				err_sum += h0.err;
				hact_sum += h0.hsum;
				h2_sum += h0.I;

				if ((iteration % record_interval) == 0) {

					v_err.push_back(err_sum/(float)record_interval);
					v_hsum.push_back(hact_sum/(float)record_interval);
					v_h2_sum.push_back(h2_sum/(float)record_interval);
					err_sum = .0f;
					hact_sum = .0f;
					h2_sum = .0f;
					time_per_iteration = t.end()/(float)record_interval;
					printf("err = %g, hsum = %.3f, entropy = %.3f, t = %.6f s/iter (> %.3f GF/s)\n", v_err.back(), 
						v_hsum.back(), v_h2_sum.back(), time_per_iteration, 
						((h0.mflops/time_per_iteration)/ 1024.0f));

					t.start();
				
				}

				display_mutex.unlock();
			}

			usleep(100);

		}

		delete(main_display);

		for (size_t i = 0; i < inputs.size(); i++)
		 	delete(inputs[i]);
	}

}

int main(int argc, char* argv[]) {

	GLWindowRenderer::init();
	//GLWindowRenderer::set_frame_rate(30);
	GLWindowRenderer::add("v");
	GLWindowRenderer::add("vn");
	GLWindowRenderer::add("pc");
	GLWindowRenderer::add("W");
	GLWindowRenderer::add("H");
	// GLWindowRenderer::add("h");
	// GLWindowRenderer::add("hn");
	// GLWindowRenderer::add("H1.H");
	// GLWindowRenderer::add("H1.W");
	// GLWindowRenderer::add("W projection");
	// GLWindowRenderer::add("vn1");
	// GLWindowRenderer::add("vn2");
	// GLWindowRenderer::add("W3 projection");
	// GLWindowRenderer::add("h2.H");
	GLWindowRenderer::add("ve");
	//GLWindowRenderer::add("h_act");

	std::thread main_thread(test);

	sleep(1);

	GLWindowRenderer::main_loop();

	main_thread.join();

    return 0;

}
