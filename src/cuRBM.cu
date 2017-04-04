#ifndef __CXX_CUDA_RBM_H__
#define __CXX_CUDA_RBM_H__

#include <encoder.h>
#include <matrix.h>
#include <vector.h>
#include <rbm.h>

class CUDA_RBM : public Encoder {

  public:

	CUDA_RBM() = default;

	CUDA_RBM( size_t n_visible, size_t n_hidden, size_t b_size, DISPLAY_MODE dm ) : Encoder( n_visible,
		        n_hidden, b_size ) {

	}

	~CUDA_RBM() { }
	virtual void encode( void* data, void* code, size_t size ) { std::cout << "RBM::encode" << std::endl; };
	virtual void decode( void* code, void* data, size_t size ) { std::cout << "RBM::decode" << std::endl; };


	void up( Matrix<float>* in ) { }
	void down( Matrix<float>* in ) { };
	void compute_statistics() {}
	void adapt( float alpha = 0.001f, float decay = 0.0000f, float sparsecost = 0.0f,
	            float sparsetarget = 0.1f, float sparsedamping = 0.0f ) {};

	virtual void serialize( std::ostream& os ) const {}

	RBM* cpu_copy;

};

#endif /*__CXX_CUDA_RBM_H__*/
