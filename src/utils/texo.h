/* 1D texture object template */
static void sz_check(int n) {
    if (n <            0 ) ERR("too small texo: %d", n);
    if (n > MAX_TEXO_SIZE) ERR("too big texo: %d > %d", n, MAX_TEXO_SIZE);
}

template<typename T>
struct Texo {
    cudaTextureObject_t d;
};

template<typename T>
void texo_setup(int n, T *data, /**/ Texo<T> *to) {
    sz_check(n);
    
    cudaResourceDesc resD;
    cudaTextureDesc  texD;
    
    memset(&resD, 0, sizeof(resD));
    resD.resType = cudaResourceTypeLinear;
    resD.res.linear.devPtr = data;
    resD.res.linear.sizeInBytes = n * sizeof(T);
    resD.res.linear.desc = cudaCreateChannelDesc<T>();
    
    memset(&texD, 0, sizeof(texD));
    texD.normalizedCoords = 0;
    texD.readMode = cudaReadModeElementType;
    
    CC(cudaCreateTextureObject(&to->d, &resD, &texD, NULL));
}

template<typename T>
void texo_destroy(/**/ Texo<T> *to) {
    CC(cudaDestroyTextureObject(to->d));
}
