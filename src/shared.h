/*
* @File: shared.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-04-26 12:54:24
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-05-04 11:47:24
*/

#include <stddef.h> // size_t
#include <iostream> // couts, etc...
#include <cstring> // memset, etc...

constexpr size_t operator "" _k(unsigned long long size) {
   return static_cast<size_t>(size * 1024);
}