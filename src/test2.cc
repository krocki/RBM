/*
* @File: test0.cc
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-05-13 12:48:59
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-05-13 15:43:19
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

int test(void) {

	Timer t;
	Matrix<float>* main_display;

	size_t patch_size = 16;
	size_t offset = 32;
	size_t characters = 128 - offset;
	float err_sum = .0f;
	size_t record_interval = 1000;
	size_t num_h0 = 256;
	size_t batchsize = 4;

	RBM h0(characters, num_h0, NTYPE::BINARY, NTYPE::BINARY, LTYPE::CD, DISPLAY_MODE::DEFAULT_MODE, batchsize);
	Matrix<float> patch(1, characters);
	Matrix<float> graph(1, 1);
	graph.display_mode = DISPLAY_MODE::GRAPH;

	main_display = new Matrix<float>(1, characters);
	GLWindowRenderer::set(1, h0.v);
	GLWindowRenderer::set(2, h0.H);
	GLWindowRenderer::set(3, h0.vn);
	GLWindowRenderer::set(4, &graph);

	h0.v->reset();

	for (size_t iteration = 0;; iteration++) {

		display_mutex.lock();

		if (!_pause || _step) {

			_step = false;
			
			for (size_t b = 0; b < batchsize; b++) {

				int n = MT19937::randint() % (characters);
				char c = (char)(n+offset);
				//char2matrix_16x16(c, *(main_display));
				char2matrix_one_hot(c, *(main_display));
				MATRIX_MEMCPY_ROW(h0.v, main_display, b);
			}

			h0.up();
			h0.down(h0.H);
			h0.adapt(0.1f);

			h0.compute_statistics();

			err_sum += h0.err;

			if ((iteration % record_interval) == 0) {

				v_err.push_back(err_sum/(float)record_interval);
				err_sum = .0f;
				printf("%.6f\n", v_err.back());
			
			}

		}

		display_mutex.unlock();
	}

	delete(main_display);

	return 0;

}

int main(int argc, char* argv[]) {

	GLWindowRenderer::init();
	GLWindowRenderer::add("v");
	GLWindowRenderer::add("H");
	GLWindowRenderer::add("vn");
	GLWindowRenderer::add("ve");

	std::thread main_thread(test);

	GLWindowRenderer::main_loop();

	main_thread.join();

    return 0;

}
