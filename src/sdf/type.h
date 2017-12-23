#include "sdf/tex3d/imp.h"

struct tex3Dca {
    Tex3d *t;
    cudaTextureObject_t to;

    void setup(cudaArray *ca) {
        tex3d_ini(&t);

        cudaResourceDesc resD;
        cudaTextureDesc  texD;

        memset(&resD, 0, sizeof(resD));
        resD.resType = cudaResourceTypeArray;
        resD.res.array.array = ca;

        memset(&texD, 0, sizeof(texD));
        texD.normalizedCoords = 0;
        texD.filterMode = cudaFilterModePoint;
        texD.mipmapFilterMode = cudaFilterModePoint;
        texD.addressMode[0] = cudaAddressModeWrap;
        texD.addressMode[1] = cudaAddressModeWrap;
        texD.addressMode[2] = cudaAddressModeWrap;

        CC(cudaCreateTextureObject(&to, &resD, &texD, NULL));
    }

    void destroy() {
        tex3d_fin(t);
        CC(cudaDestroyTextureObject(to));
    }
};
