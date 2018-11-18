#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "utils/kl.h"

#include "math/tform/type.h"
#include "math/tform/imp.h"
#include "math/tform/dev.h"

#include "imp.h"

static __global__ void convert(Tform_v t, float *a, /**/ float *b) {
    enum {X, Y, Z};
    tform_convert_dev(&t, a, /**/ b);
}

void convert_dev(Tform *t, float a_hst[3], /**/ float b_hst[3]) {
    enum {dim = 3};
    float *a_dev, *b_dev;
    Tform_v v;

    Dalloc(&a_dev, dim);
    Dalloc(&b_dev, dim);

    cH2D(a_dev, a_hst, dim);
    tform_to_view(t, &v);
    KL(convert, (1, 1), (v, a_dev, /**/ b_dev));
    cD2H(b_hst, b_dev, dim);

    Dfree(a_dev);
    Dfree(b_dev);
}
