void tex3d_ini(Tex3d **pq) {
    Tex3d *q;
    UC(emalloc(sizeof(Tex3d), /**/ (void**)&q));
    *pq = q;
}

void tex3d_fin(Tex3d *q) {
    CC(cudaDestroyTextureObject(q->t));
    UC(efree(q));
}

void tex3d_copy(const Array3d *a, /**/ Tex3d *t) {
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

void tex3d_to_view(const Tex3d *t, /**/ Tex3d_v *v) { v->t = t->t; }
