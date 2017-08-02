namespace k_cnt {
enum {
    XCELLS = XS,
    YCELLS = YS,
    ZCELLS = ZS,
    XOFFSET = XCELLS / 2,
    YOFFSET = YCELLS / 2,
    ZOFFSET = ZCELLS / 2
};
texture<int, cudaTextureType1D> texCellsStart, texCellEntries;
__constant__ int cnsolutes[MAX_OBJ_TYPES];
__constant__ const float2 *csolutes[MAX_OBJ_TYPES];
__constant__ float *csolutesacc[MAX_OBJ_TYPES];

__constant__ int packstarts_padded[27], packcount[26];
__constant__ Particle *packstates[26];
__constant__ Force *packresults[26];
}
