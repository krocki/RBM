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

	RBM h0(patch_size * patch_size, 64, NTYPE::BINARY, NTYPE::BINARY, LTYPE::CD);
	Matrix<float> patch(patch_size, patch_size);

	main_display = new Matrix<float>(patch_size, patch_size);
	GLWindowRenderer::set(1, main_display);
	GLWindowRenderer::set(2, h0.H);
	GLWindowRenderer::set(3, h0.vn);

	for (size_t iteration = 0;; iteration++) {

		if (!_pause) {

			display_mutex.lock();
			int n = MT19937::randint() % 94;
			char c = (char)(n+32);
			char2matrix_16x16(c, *(main_display));

			MATRIX_MEMCPY(h0.v, main_display);

			h0.up(main_display);
			h0.down(h0.H);
			h0.adapt(0.05f, 0.0005f);
			
			display_mutex.unlock();
		
		}
	}

	delete(main_display);

	return 0;

}

int main(int argc, char* argv[]) {

	GLWindowRenderer::init();
	GLWindowRenderer::add("v");
	GLWindowRenderer::add("H");
	GLWindowRenderer::add("vn");

	std::thread main_thread(test);

	GLWindowRenderer::main_loop();

	main_thread.join();

    return 0;

}
