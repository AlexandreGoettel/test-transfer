#ifndef __lpsdexec_h
#define __lpsdexec_h

typedef struct {
    double **doublePointers;
    size_t numDoublePointers;
    int **intPointers;
    long int **liPointers;
    size_t numIntPointers, numLiPointers;
} DataPointers;

#endif 
