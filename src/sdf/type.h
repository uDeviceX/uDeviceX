#include "sdf/tex3d/imp.h"

struct tex3Dca {
    Tex3d *t;
    cudaTextureObject_t to;

    void setup(cudaArray *ca) {
        tex3d_ini(&t);
    }

    void destroy() {
        tex3d_fin(t);
    }
};
