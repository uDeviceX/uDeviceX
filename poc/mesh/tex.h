namespace tex
{
    static __global__ void zip4_k(const int *tt, const int nt, int4 *zipped)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        const int t1 = tt[3*i+0];
        const int t2 = tt[3*i+1];
        const int t3 = tt[3*i+2];
        zipped[i] = make_int4(t1, t2, t3, 0);
    }

    void zip4(const int *tt, const int nt, int4 *zipped)
    {
        if (nt > 0)
        zip4_k <<<(nt + 127)/128, 128>>> (tt, nt, zipped);
    }

    static __global__ void zip4_k(const float *vv, const int nv, float4 *zipped)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        const float x = vv[3*i+0];
        const float y = vv[3*i+1];
        const float z = vv[3*i+2];
        zipped[i] = make_float4(x, y, z, 0);
    }
    
    void zip4(const float *vv, const int nv, float4 *zipped)
    {
        if (nv > 0)
        zip4_k <<<(nv + 127)/128, 128>>> (vv, nv, zipped);
    }

    void maketexzip(int4 *tt, const int nt, /**/ cudaTextureObject_t *ttto)
    {
        cudaResourceDesc resD;
        cudaTextureDesc  texD;

        memset(&resD, 0, sizeof(resD));
        resD.resType = cudaResourceTypeLinear;
        resD.res.linear.devPtr = tt;
        resD.res.linear.sizeInBytes = sizeof(int4) * nt;
        resD.res.linear.desc = cudaCreateChannelDesc<int4>();

        memset(&texD, 0, sizeof(texD));
        texD.normalizedCoords = 0;
        texD.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(ttto, &resD, &texD, NULL);
    }
        
    void maketexzip(float4 *vv, const int nv, /**/ cudaTextureObject_t *vvto)
    {
        cudaResourceDesc resD;
        cudaTextureDesc  texD;

        memset(&resD, 0, sizeof(resD));
        resD.resType = cudaResourceTypeLinear;
        resD.res.linear.devPtr = vv;
        resD.res.linear.sizeInBytes = sizeof(float4) * nv;
        resD.res.linear.desc = cudaCreateChannelDesc<float4>();

        memset(&texD, 0, sizeof(texD));
        texD.normalizedCoords = 0;
        texD.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(vvto, &resD, &texD, NULL);
    }

    
}
