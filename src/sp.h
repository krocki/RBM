/*
* @File: sp.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-04-28 11:39:24
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-05-15 16:46:24
*/

#ifndef __SPATIAL_POOLER_H__
#define __SPATIAL_POOLER_H__

#include <encoder.h>
#include <matrix.h>

// https://cireneikual.wordpress.com

class SpatialPooler : public Encoder {

	public:

		SpatialPooler() : Encoder() {}
		SpatialPooler(size_t n_input, size_t n_output, size_t b_size = 1) : Encoder(n_input, n_output, b_size = 1) {

			W = new Matrix<float>(num_output, num_input);
			c = new Matrix<float>(num_output, batch_size);
			h = new Matrix<float>(num_output, batch_size);
			H = new Matrix<float>(num_output, batch_size);
			i = new Matrix<float>(num_output, num_output);
			v = new Matrix<float>(batch_size, num_input);
			vn = new Matrix<float>(batch_size, num_input);
			e = new Matrix<float>(batch_size, num_input);
			W_delta = new Matrix<float>(num_output, num_input);
			c_delta = new Matrix<float>(num_output, batch_size);
			i_delta = new Matrix<float>(num_output, num_output);

			W->elementwise_function(&_randn);
			W->elementwise_function_scalar(&_mult, 0.001f);
			c->elementwise_function(&_randn);
			c->elementwise_function_scalar(&_mult, 0.001f);
			i->elementwise_function(&_rand);

			W->display_mode = DISPLAY_MODE::ROWS_ARE_WEIGHTS;
			H->display_mode = DISPLAY_MODE::ROWS_ARE_IMAGES;
			h->display_mode = DISPLAY_MODE::ROWS_ARE_IMAGES;

		}

		SpatialPooler(const SpatialPooler& sp) : Encoder(sp) {}
		~SpatialPooler() {

			if (W != nullptr) delete(W);
			if (v != nullptr) delete(v);
			if (vn != nullptr) delete(vn);
			if (c != nullptr) delete(c);
			if (h != nullptr) delete(h);
			if (H != nullptr) delete(H);
			if (i != nullptr) delete(i);
			if (i_delta != nullptr) delete(i_delta);
			if (e != nullptr) delete(e);
			if (W_delta != nullptr) delete(W_delta);
			if (c_delta != nullptr) delete(c_delta);
		}

		void up(void) {

			v->flatten();
			Matrix<float>::sgemm(*h, *W, *v);
			h->elementwise_function_matrix(&_add, *c);
			h->elementwise_function(&_ReLU);

			Matrix<float>::sgemm(*H, *i, *h);
			H->elementwise_function_matrix(&_sub, *h);

			H->elementwise_function(&_sgn);
		}

		void down(void) {

			H->transpose();
			Matrix<float>::sgemm(*vn, *H, *W);
			H->transpose();

			MATRIX_MEMCPY(e, v);
			e->elementwise_function_matrix(&_sub, *vn);

		}

		void adapt(float alpha = 0.001f, float decay = 0.0000f, float sparsity = 0.1f) {

			float lscf = sparsity - (float)H->sum()/(float)num_output;

			// reset
			W_delta->elementwise_function(&_zero);
			c_delta->elementwise_function(&_zero);
			i_delta->elementwise_function(&_zero);

			// compute
			//e->transpose();
			Matrix<float>::sgemm(*W_delta, *H, *e);
			//e->transpose();

			W_delta->elementwise_function_scalar(&_mult, alpha);

			// MATRIX_MEMCPY(c_delta, h);
			// c_delta->elementwise_function_matrix(&_sub, *hn);
			// c_delta->elementwise_function_scalar(&_mult, alpha);

			// MATRIX_MEMCPY(h, H);
			// h->transpose();
			// Matrix<float>::sgemm(*i_delta, *H, *h);
			// h->transpose();
			// i_delta->elementwise_function_scalar(&_mult, alpha);
 
			// // apply
			W->elementwise_function_matrix(&_add, *W_delta);
			// i->elementwise_function_matrix(&_add, *i_delta);
			// i_delta->elementwise_function(&_ReLU);
			// i->reset_diagonal();
			W->elementwise_function_scalar(&_mult, 1.0f-decay);
			c->elementwise_function_matrix(&_add, *c);
		}

		virtual void encode(void* data, void* code, size_t size) { std::cout << "SP::encode" << std::endl; };
		virtual void decode(void* code, void* data, size_t size) { std::cout << "SP::decode" << std::endl; };

		Matrix<float>* W;
		Matrix<float>* c;
		Matrix<float>* v;
		Matrix<float>* vn;
		Matrix<float>* h;
		Matrix<float>* H;
		Matrix<float>* i;
		Matrix<float>* i_delta;
		Matrix<float>* e;
		Matrix<float>* W_delta;
		Matrix<float>* c_delta;

};

#endif /*__SPATIAL_POOLER_H__*/