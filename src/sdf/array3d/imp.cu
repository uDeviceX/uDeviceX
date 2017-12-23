#include <stdio.h>

#include "conf.h"
#include "inc/conf.h"

#include "utils/imp.h"
#include "utils/error.h"
#include "utils/cc.h"

#include "type.h"
#include "imp.h"

void array3d_ini(Array3d **pq, int x, int y, int z) {
    Array3d *q;
    cudaChannelFormatDesc fmt;

    UC(emalloc(sizeof(Array3d), /**/ (void**)&q));

    fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray(&q->a, &fmt, make_cudaExtent(x, y, z)));

    *pq = q;
}

void array3d_fin(Array3d *q) {
    UC(efree(q));
}

void array3d_copy(int x, int y, int z, float *D, /**/ Array3d *q) {
    cudaMemcpy3DParms copyParams;
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.srcPtr = make_cudaPitchedPtr((void*)D, x*sizeof(float), x, y);
    copyParams.dstArray = q->a;
    copyParams.extent = make_cudaExtent(x, y, z);
    copyParams.kind = cudaMemcpyHostToDevice;
    CC(cudaMemcpy3D(&copyParams));
}
