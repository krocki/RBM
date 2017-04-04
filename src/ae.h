/*
* @File: ae.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-05-02 21:11:04
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-05-08 17:22:51
*/

#ifndef __AUTOENCODER_H__
#define __AUTOENCODER_H__

#include <encoder.h>

class AutoEncoder : public Encoder {

	public:

		AutoEncoder() : Encoder() {}
		AutoEncoder(size_t n_input, size_t n_output, size_t b_size = 1) : Encoder(n_input, n_output, b_size = 1) {}
		AutoEncoder(const AutoEncoder& ae) : Encoder(ae) { }
		~AutoEncoder() = default;

		virtual void encode(void* data, void* code, size_t size) { std::cout << "AE::encode" << std::endl; };
		virtual void decode(void* code, void* data, size_t size) { std::cout << "AE::decode" << std::endl; };

	protected:

};

#endif /*__AUTOENCODER_H__*/