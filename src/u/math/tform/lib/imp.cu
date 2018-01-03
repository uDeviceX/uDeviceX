#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "utils/kl.h"

#include "math/tform/type.h"
#include "math/tform/dev.h"

#include "imp.h"

static __global__ void convert(Tform t, float *a, /**/ float *b) {
    enum {X, Y, Z};
    b[X] = b[Y] = b[Z] = 42.0;
    tform_convert_dev(&t, a, /**/ b);
}

void convert_dev(Tform *t, float a_hst[3], /**/ float b_hst[3]) {
    enum {dim = 3};
    float *a_dev, *b_dev;
    Dalloc(&a_dev, dim);
    Dalloc(&b_dev, dim);

    cH2D(a_dev, a_hst, dim);
    KL(convert, (1, 1), (*t, a_dev, /**/ b_dev));
    cD2H(b_hst, b_dev, dim);

    Dfree(a_dev);
    Dfree(b_dev);
}
