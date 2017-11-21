/* return origin and extents of the cells of fragment i */
static __device__ void get_frag_box(int i, /**/ int org[3], int ext[3]) {
    int d[3] = frag_i2d3(i);
    int c, L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c) {
        org[c] = (d[c] == 1) ? L[c] - 1 : 0;
        ext[c] = (d[c] == 0) ? L[c]     : 1;
    }
}

/* convert cellid from frag coordinates to bulk coordinates */
static __device__ int frag2bulk(int hci, const int org[3], const int ext[3]) {
    enum {X, Y, Z};
    int c;
    int src[3];
    int dst[3] = {hci % ext[X], (hci / ext[X]) % ext[Y], hci / (ext[X] * ext[Y])};
    for (c = 0; c < 3; ++c) src[c] = org[c] + dst[c];
    return src[X] + XS * (src[Y] + YS * src[Z]);
}

__global__ void count_cells(const int27 cellpackstarts, const int *start, const int *count,
                            /**/ intp26 fragss, intp26 fragcc) {
    enum {X, Y, Z};
    int gid, fid; /* fragment id */
    int nhc; /* number of halo cells */
    int cid, hci; /* bulk and halo cell ids */
    int org[3], ext[3]; /* fragment origin and extend */
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= cellpackstarts.d[26]) return;

    fid = k_common::fid(cellpackstarts.d, gid);
    hci = gid - cellpackstarts.d[fid];

    get_frag_box(fid, /**/ org, ext);
    nhc = ext[X] * ext[Y] * ext[Z];

    if (hci < nhc) {
        cid = frag2bulk(hci, org, ext);
        fragss.d[fid][hci] = start[cid];
        fragcc.d[fid][hci] = count[cid];
    } else if (hci == nhc) {
        fragss.d[fid][hci] = 0;
        fragcc.d[fid][hci] = 0;
    }
}
