namespace k_cnt {
static const int maxsolutes = 32;
enum {
    XCELLS = XS,
    YCELLS = YS,
    ZCELLS = ZS,
    XOFFSET = XCELLS / 2,
    YOFFSET = YCELLS / 2,
    ZOFFSET = ZCELLS / 2
};
static const int NCELLS = XS * YS * ZS;
texture<int, cudaTextureType1D> texCellsStart, texCellEntries;
__constant__ int cnsolutes[maxsolutes];
__constant__ const float2 *csolutes[maxsolutes];
__constant__ float *csolutesacc[maxsolutes];

__constant__ int packstarts_padded[27], packcount[26];
__constant__ Particle *packstates[26];
__constant__ Force *packresults[26];
}
