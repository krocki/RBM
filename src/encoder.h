/*
* @File: encoder.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-04-26 10:11:13
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-05-18 14:33:19
*/

#ifndef __ENCODER_H__
#define __ENCODER_H__

class Encoder {

	public:

		Encoder() = default;
		Encoder(const Encoder&) = default;
		
		Encoder(size_t n_input, size_t n_output, size_t b_size = 1) { 

			num_input = n_input;
			num_output = n_output;
			batch_size = b_size;

		};

		virtual ~Encoder() = default;

		virtual void serialize(std::ostream& os) const {

			os << "[" << num_input << " x " << num_output << " Encoder]" << std::endl;

		}

		friend std::ostream& operator<<(std::ostream& os, const Encoder& e) {

			e.serialize(os);

			return os;

		}

		virtual void encode(void* data, void* code, size_t size) = 0;
		virtual void decode(void* code, void* data, size_t size) = 0;

		size_t num_input;
		size_t num_output;
		size_t batch_size;

};


#endif /*__ENCODER_H__*/
