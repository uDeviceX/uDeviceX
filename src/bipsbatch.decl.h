namespace BipsBatch {
__constant__ unsigned int start[27];

enum HaloType { HALO_BULK = 0, HALO_FACE = 1, HALO_EDGE = 2, HALO_CORNER = 3 };

struct BatchInfo {
    float *xdst;
    float2 *xsrc;
    float seed;
    int ndst, nsrc, mask, *cellstarts, *scattered_entries, dx, dy, dz, xcells,
        ycells, zcells;
    HaloType halotype;
};

__constant__ BatchInfo batchinfos[26];


bool firstcall = true;

cudaEvent_t evhalodone;
}
