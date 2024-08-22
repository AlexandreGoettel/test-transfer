#ifndef __misc_h
#define __misc_h

void *xmalloc(size_t size);
void xfree(void *p);
static inline double dMax(double x, double y) {
    return x > y ? x : y;
}

#endif
