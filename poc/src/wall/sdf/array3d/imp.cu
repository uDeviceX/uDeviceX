#include <stdio.h>

#include "conf.h"
#include "inc/conf.h"

#include "d/api.h"

#include "utils/imp.h"
#include "utils/error.h"
#include "utils/cc.h"
#include "utils/msg.h"

#include "type.h"
#include "imp.h"

void array3d_ini(Array3d **pq, size_t x, size_t y, size_t z) {
    Array3d *q;
    cudaChannelFormatDesc fmt;
    EMALLOC(1, &q);

    fmt = cudaCreateChannelDesc<float>();
    msg_print("alloc cuda 3D Array: %ld %ld %ld", x, y, z);
    CC(cudaMalloc3DArray(&q->a, &fmt, make_cudaExtent(x, y, z)));
    q->x = x; q->y = y; q->z = z;

    *pq = q;
}

void array3d_fin(Array3d *q) {
    CC(cudaFreeArray(q->a));
    EFREE(q);
}

static int good(size_t x, size_t y, size_t z, Array3d *q) {
    return x == q->x && y == q->y && z == q->z;
}

void array3d_copy(size_t x, size_t y, size_t z, float *D, /**/ Array3d *q) {
    cudaMemcpy3DParms copyParams;
    if (!good(x, y, z, q))
        ERR("wrong size: %ld, %ld, %ld   !=   %ld, %ld, %ld",
            x, y, z, q->x, q->y, q->z);
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.srcPtr = make_cudaPitchedPtr((void*)D, x*sizeof(float), x, y);
    copyParams.dstArray = q->a;
    copyParams.extent = make_cudaExtent(x, y, z);
    copyParams.kind = cudaMemcpyHostToDevice;
    CC(cudaMemcpy3D(&copyParams));
}
