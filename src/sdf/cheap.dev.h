/* 3D texture object binded to  cuda array */
template <typename T>
struct tex3Dca {
    cudaTextureObject_t to;

    __device__ __forceinline__
    const T fetch(const float i, const float j, const float k) const {
        return tex3D<T>(to, i, j, k);
    }
    
    void setup(cudaArray *ca) {
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

    void destroy() {CC(cudaDestroyTextureObject(to));}
};

__device__ static int iround(float x) {
    return (x > 0.5) ? (x + 0.5) : (x - 0.5);
}

/* within the rescaled texel width error */
__device__ float cheap_sdf(const tex3Dca<float> texsdf, float x, float y, float z)  {
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    int tc[3];
    float r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
        tc[c] = iround(T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]));

#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    return tex0(0, 0, 0);
#undef  tex0
}
