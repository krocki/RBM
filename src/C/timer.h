#include <sys/time.h>

double get_time_diff(const struct timeval *s, const struct timeval *e) {

  struct timeval diff;

  diff.tv_usec = e->tv_usec - s->tv_usec;
  diff.tv_sec  = e->tv_sec  - s->tv_sec;

  if (s->tv_usec > e->tv_usec) {

    diff.tv_usec += 1e6;
    diff.tv_sec--;

  }

  return (double) diff.tv_sec + ((double) diff.tv_usec / 1e6f);
}
