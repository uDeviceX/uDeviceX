#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "math/tform/type.h"
#include "math/tform/dev.h"

#include "main.h"

static __global__ void convert(Tform t, float *a, /**/ float *b) {
    
}

void convert_dev(Tform *t, float a[3], /**/ float b[3]) {
    enum {dim = 3};
    float *a_dev, *b_dev;
    Dalloc(&a_dev, dim);
    Dalloc(&b_dev, dim);

    cH2D(a_dev, a, dim);
    cH2D(b_dev, b, dim);

    Dfree(a_dev);
    Dfree(b_dev);
}
