/*
* @File: tests.cc
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-04-26 12:28:42
* @Last Modified by:   kamilrocki
* @Last Modified time: 2017-04-04 09:18:06
* @Last Modified time: 2015-05-18 15:51:36
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
#ifdef USE_OPENGL
#include <glWindow.h>
#else
std::vector<float> v_err;
std::vector<float> v_hsum;
std::vector<float> v_h2_sum;
#endif

void test( void ) {

	Matrix<float>* main_display;
	Timer t;

	std::vector<Matrix<float>*> inputs =
	    ImageImporter::importFromFile( "data/mnist/train-images-idx3-ubyte", 28, 28, 16 );
	Matrix<float> graph( 1, 1 );
	Matrix<float> graph2( 1, 1 );
	graph.display_mode = DISPLAY_MODE::GRAPH;

	size_t patch_size = 28;
	float err_sum = .0f;
	float hact_sum = .0f;
	float h2_sum = .0f;
	float time_per_iteration;
	size_t record_interval = 1000;
	size_t num_h0 = 100;
	size_t batchsize = 16;

	if ( inputs.size() > 0 ) {

		RBM h0( patch_size * patch_size, num_h0, NTYPE::BINARY, NTYPE::BINARY, LTYPE::CD,
		        DISPLAY_MODE::UNFLATTEN, batchsize );

		Matrix<float> patch( patch_size, patch_size );

		main_display = new Matrix<float>( patch_size, patch_size );
#ifdef USE_OPENGL
		GLWindowRenderer::set( 1, h0.v );
		GLWindowRenderer::set( 2, h0.vn );
		GLWindowRenderer::set( 3, h0.pc );
		GLWindowRenderer::set( 4, h0.W );
		GLWindowRenderer::set( 5, h0.H );
		GLWindowRenderer::set( 6, &graph );
#endif
		t.start();

		for ( size_t iteration = 0;; iteration++ ) {

#ifdef USE_OPENGL

			if ( !_pause || _step ) {

				_step = false;

				display_mutex.lock();
#endif

				for ( size_t b = 0; b < batchsize; b++ ) {

					int num = MT19937::randint() % inputs.size();
					Matrix<float>::randsubmatrix( *( inputs[num] ), patch );
					//MATRIX_MEMCPY(main_display, &patch);
					MATRIX_MEMCPY_ROW( h0.v, &patch, b );

				}

				h0.up( h0.v );
				h0.down( h0.H );
				h0.adapt( 0.01f, 0.00000f, 0.00000f );
				h0.compute_statistics();

				err_sum += h0.err;
				hact_sum += h0.hsum;
				h2_sum += h0.I;

				if ( ( iteration % record_interval ) == 0 ) {


					v_err.push_back( err_sum / ( float )record_interval );
					v_hsum.push_back( hact_sum / ( float )record_interval );
					v_h2_sum.push_back( h2_sum / ( float )record_interval );

					err_sum = .0f;
					hact_sum = .0f;
					h2_sum = .0f;
					time_per_iteration = t.end() / ( float )record_interval;
					printf( "err = %g, hsum = %.3f, entropy = %.3f, t = %.6f s/iter (> %.3f GF/s)\n", v_err.back(),
					        v_hsum.back(), v_h2_sum.back(), time_per_iteration,
					        ( ( h0.mflops / time_per_iteration ) / 1024.0f ) );

					t.start();

				}

#ifdef USE_OPENGL
				display_mutex.unlock();
			}

			usleep( 100 );
#endif
		}

		delete( main_display );

		for ( size_t i = 0; i < inputs.size(); i++ )
		{ delete( inputs[i] ); }
	}

}

int main( int argc, char* argv[] ) {

#ifdef USE_OPENGL
	GLWindowRenderer::init();
	GLWindowRenderer::add( "v" );
	GLWindowRenderer::add( "vn" );
	GLWindowRenderer::add( "pc" );
	GLWindowRenderer::add( "W" );
	GLWindowRenderer::add( "H" );
	GLWindowRenderer::add( "ve" );

	std::thread main_thread( test );

	GLWindowRenderer::main_loop();

	main_thread.join();

#else
	test();
#endif

	return 0;

}
