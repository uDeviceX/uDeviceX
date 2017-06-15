# hdr

    texture<float2, 1, cudaReadModeElementType> texVertices;
    texture<int,    1, cudaReadModeElementType> texAdjVert;
    texture<int,    1, cudaReadModeElementType> texAdjVert2;
    texture<int4,            cudaTextureType1D> texTriangles4;

# imp

    void setup(int* faces)
    void forces(int nc, Particle *pp, Force *ff, float* host_av)
    int setup(Particle* pp, int nv, /*w*/ Particle *pp_hst)
    rbc_dump(int nc, Particle *p, int* triplets, int nv, int nt, int id) {
