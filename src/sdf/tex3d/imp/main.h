#include <stdio.h>

#include "conf.h"
#include "inc/conf.h"

#include "d/api.h"

#include "utils/imp.h"
#include "utils/error.h"
#include "utils/cc.h"

#include "sdf/array3d/type.h"

#include "type.h"
#include "imp.h"

void tex3d_ini(Tex3d **pq) {
    Tex3d *q;
    UC(emalloc(sizeof(Tex3d), /**/ (void**)&q));
    *pq = q;
}

void tex3d_fin(Tex3d *q) {
    CC(cudaDestroyTextureObject(q->t));
    UC(efree(q));
}

void tex3d_copy(Array3d *a, /**/ Tex3d *t) {
    cudaResourceDesc resD;
    cudaTextureDesc  texD;

    memset(&resD, 0, sizeof(resD));
    resD.resType = cudaResourceTypeArray;
    resD.res.array.array = a->a;

    memset(&texD, 0, sizeof(texD));
    texD.normalizedCoords = 0;
    texD.filterMode = cudaFilterModePoint;
    texD.mipmapFilterMode = cudaFilterModePoint;
    texD.addressMode[0] = cudaAddressModeWrap;
    texD.addressMode[1] = cudaAddressModeWrap;
    texD.addressMode[2] = cudaAddressModeWrap;

    CC(cudaCreateTextureObject(&t->t, &resD, &texD, NULL));
}
