/* 1D texture object template */
template<typename T>
struct Texo {
    cudaTextureObject_t to;

    __device__ __forceinline__
    const T fetch(const int i) const {return tex1Dfetch<T>(to, i);}

    void setup(T *data, int n) {
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

        CC(cudaCreateTextureObject(&to, &resD, &texD, NULL));
    }

    void destroy() {CC(cudaDestroyTextureObject(to));}
};
