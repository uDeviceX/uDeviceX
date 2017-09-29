enum {
    XCELLS = XS,
    YCELLS = YS,
    ZCELLS = ZS,
    XOFFSET = XCELLS / 2,
    YOFFSET = YCELLS / 2,
    ZOFFSET = ZCELLS / 2
};

namespace c {
texture<int, cudaTextureType1D> start, id;
__constant__ const float2 *PP[MAX_OBJ_TYPES];
__constant__ float *FF[MAX_OBJ_TYPES];
}

namespace h {
__constant__ int starts[27];
__constant__ Particle *pp[26];
__constant__ Force *ff[26];
}
