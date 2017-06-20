/* a common textrue setup */
#define setup_texture(T, TYPE) do {                         \
        (T).channelDesc = cudaCreateChannelDesc<TYPE>();    \
        (T).filterMode = cudaFilterModePoint;               \
        (T).mipmapFilterMode = cudaFilterModePoint;         \
        (T).normalized = 0;                                 \
    } while (false)


/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cudaAssert((ans), __FILE__, __LINE__); } while (0)
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        abort();
    }
}

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

#define D2D cudaMemcpyDeviceToDevice
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define H2H cudaMemcpyHostToHost
#define A2A cudaMemcpyDefault /* "[a]ll to [a]ll" */

#define cD2D(t, f, n) CC(cudaMemcpy((t), (f), (n) * sizeof((f)[0]), D2D))
#define cH2H(t, f, n) CC(cudaMemcpy((t), (f), (n) * sizeof((f)[0]), H2H))  /* [t]to, [f]rom */
#define cA2A(t, f, n) CC(cudaMemcpy((t), (f), (n) * sizeof((f)[0]), A2A))

#define cD2H(h, d, n) CC(cudaMemcpy((h), (d), (n) * sizeof((h)[0]), D2H))
#define cH2D(d, h, n) CC(cudaMemcpy((d), (h), (n) * sizeof((h)[0]), H2D))

#define dSync() CC(cudaDeviceSynchronize())
