/* 3D texture object binded to  cuda array */
typedef cudaTextureObject_t tex3Dca;
static void ini(tex3Dca to, cudaArray *ca) {
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

static void fin(tex3Dca *to) {
    CC(cudaDestroyTextureObject(*to));
}


