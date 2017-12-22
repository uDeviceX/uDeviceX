/* 3D texture object binded to  cuda array */
struct tex3Dca {
    cudaTextureObject_t to;
    
    void ini(cudaArray *ca) {
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

    void fin() {CC(cudaDestroyTextureObject(to));}
};

