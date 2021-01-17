#include <stdlib.h>
#include <math.h>

#if 0 //defined(__x86_64__)
unsigned long rdrand() {

  unsigned long v;
  char c;
  do {
    __asm__ volatile(
        "rdrand %0; setc %1"
        : "=r" (v), "=qm" (c)
        );
  } while (c != 1);

  return v;
}
float randf() {

  unsigned long v;
  char c;
  do {
    __asm__ volatile(
        "rdrand %0; setc %1"
        : "=r" (v), "=qm" (c)
        );
  } while (c != 1);

  unsigned long ui_max = ~0;
  float f = (float)v / ((float)ui_max + 1.0f);
  return f;
}
#else
unsigned long rdrand() { return random(); }
float randf() { return rdrand() / (RAND_MAX + 1.0f); }
#endif
float randn(const float mean, const float std) {
  float  x = randf(),
         y = randf(),
         z = sqrtf(-2 * logf(x)) * cos(2 * M_PI * y);
  return std*z + mean;
}
