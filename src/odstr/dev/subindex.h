namespace odstr { namespace sub { namespace dev {

struct Pa { /* local particle */
    float2 d0, d1, d2;
};

__device__ void readPa(float2 *pp, int ws, int dwe, /**/ Pa *p) {
    k_read::AOS6f(pp + 3*ws, dwe, /**/ p->d0, p->d1, p->d2);
}

__device__ void writePa(Pa *p, int ws, int dwe, /**/ float2 *pp) {
    k_write::AOS6f(/**/ pp + 3*ws, dwe, /*i*/ p->d0, p->d1, p->d2);
}

__device__ void shiftPa(Pa *p, float r[3]) {
    enum {X, Y, Z};
    p->d0.x += r[X];   p->d0.y += r[Y];   p->d1.x += r[Z];
}

__device__ void Pa2r(Pa *p, /**/ float r[3]) {
    enum {X, Y, Z};
    r[X] = p->d0.x;   r[Y] = p->d0.y;   r[Z] = p->d1.x;
}

__device__ void Pa2v(Pa *p, /**/ float v[3]) {
    enum {X, Y, Z};
    v[X] = p->d1.y;   v[Y] = p->d2.x;   v[Z] = p->d2.y;
}

__global__ void subindex(const int n, const int strt[], /*io*/ float2 *pp, int *counts, /**/ uchar4 *subids) {
    enum {X, Y, Z};
    int warp, slot, fid;
    float2 d0, d1, d2;
    int ws;  /* warp start in global coordinates    */
    int dw;  /* shift relative to `ws' (lane)       */
    int dwe; /* warp or buffer end relative to `ws' */

    float shift[3];
    Pa p;

    warp = threadIdx.x / warpSize;
    dw   = threadIdx.x % warpSize;
    ws   = warpSize * warp + blockDim.x * blockIdx.x;

    if (ws >= n) return;
    
    dwe  = min(warpSize, n - ws);
    slot = ws + dw;
    fid  = k_common::fid(strt, slot);
    
    k_read::AOS6f(pp + 3*ws, dwe, d0, d1, d2);
    
    if (dw < dwe) {
        int xi, yi, zi, cid, subindex;
        fid2shift(fid, /**/ shift);

        d0.x += shift[X];
        d0.y += shift[Y];
        d1.x += shift[Z];

        xi = x2c(d0.x, XS);
        yi = x2c(d0.y, YS);
        zi = x2c(d1.x, ZS);

        check_vel(d1.y, XS);
        check_vel(d2.x, YS);
        check_vel(d2.y, ZS);

        cid = xi + XS * (yi + YS * zi);
        subindex = atomicAdd(counts + cid, 1);

        subids[slot] = make_uchar4(xi, yi, zi, subindex);
    }

    k_write::AOS6f(pp + 3*ws, dwe, d0, d1, d2);
}

}}} /* namespace */
