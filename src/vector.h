/*
* @File: vector.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-04-26 13:05:16
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-05-02 15:09:40
*/

#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <tensor.h>

// Vector is a 1D Matrix

template <typename T>
class Vector : public Matrix<T> {

	public:

		Vector() = default;
		Vector(const Vector& v) : Matrix<T>(v) { } //copy everything from Matrix

		Vector(size_t N) : Matrix<T>({N, 1}) {

		}

		~Vector() = default;

        virtual void serialize(std::ostream& os) const {

            Matrix<T>::serialize(os);
            
        }

};

#endif /*__VECTOR_H__*/
