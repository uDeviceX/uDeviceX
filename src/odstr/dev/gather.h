namespace odstr { namespace sub { namespace dev {
/** gather_id is in dev/common.h */
__device__ void xchg(int dw, /**/ float3 *s0, float3 *s1) { /* collective */
    int src0, src1;
    if (dw % 2  == 0) {
        src0 =      dw / 2; src1 = 16 + dw / 2;
    } else {
        src0 = 16 + dw / 2; src1 = dw / 2;
    }
    xchg_aos4f(src0, src1, dw % 2, /**/ s0, s1); /* collective */
}

/* [F]rom [lo]cation in memory */
struct FLo { const float2 *lo, *re; };

/* [T]o [lo]cation */
struct TLo { float2 *pp; float4  *zip0; ushort4 *zip1; };

/* Data */
struct Da { float2 d0, d1, d2; };

__device__ void ini_FLo(const float2 *lo, const float2 *re, /**/ FLo *l) {
    l->lo = lo; l->re = re;
}

__device__ void FLo2D(FLo *l, int i, /**/ Da *d) {
    bool remote;
    remote = (i >> 31) & 1;
    i &= ~(1 << 31);
    if (remote) {
        d->d0 = __ldg(l->re + 0 + 3 * i);
        d->d1 = __ldg(l->re + 1 + 3 * i);
        d->d2 = __ldg(l->re + 2 + 3 * i);
    } else {
        d->d0 = l->lo[0 + 3 * i];
        d->d1 = l->lo[1 + 3 * i];
        d->d2 = l->lo[2 + 3 * i];
    }
}

__device__ void ini_TLo(float2 *pp, float4 *zip0, ushort4 *zip1, /**/ TLo *l) {
    l->pp = pp; l->zip0 = zip0; l->zip1 = zip1;
}

__device__ void D2rv(Da *d, /**/ float r[3], float v[3]) {
    enum {X, Y, Z};
    float2 d0, d1, d2;
    d0 = d->d0; d1 = d->d1; d2 = d->d2;
    r[X] = d0.x; r[Y] = d0.y; r[Z] = d1.x;
    v[X] = d1.y; v[Y] = d2.x; v[Z] = d2.y;
}

__device__ void zip(float r[3], float v[3], int ws, int dw, int dwe, /**/
                    float4  *zip0, ushort4 *zip1) { /* collective */
    enum {X, Y, Z};
    float3 s0, s1;    
    s0 = make_float3(r[X], r[Y], r[Z]);
    s1 = make_float3(v[X], v[Y], v[Z]);
    xchg(dw, &s0, &s1); /* collective */
    if (dw < 2 * dwe)
        zip0[2 * ws + dw] = make_float4(s0.x, s0.y, s0.z, 0);
    if (dw + 32 < 2 * dwe)
        zip0[2 * ws + dw + 32] = make_float4(s1.x, s1.y, s1.z, 0);
    if (dw < dwe)
        zip1[ws + dw] = make_ushort4(__float2half_rn(r[X]),
                                     __float2half_rn(r[Y]),
                                     __float2half_rn(r[Z]),
                                     0);
}

__device__ void D2TLo(Da *d, int ws, int dw, int dwe, /**/ TLo *l) { /* collective */
    enum {X, Y, Z};
    float2 *pp;
    float4  *zip0;
    ushort4 *zip1;
    float2 d0, d1, d2;
    float r[3], v[3];

    pp = l->pp; zip0 = l->zip0; zip1 = l->zip1;
    d0 = d->d0; d1 = d->d1; d2 = d->d2;
    D2rv(d, /**/ r, v);
    zip(r, v, ws, dw, dwe, /**/ zip0, zip1);    /* collective */
    k_write::AOS6f(pp + 3*ws, dwe, d0, d1, d2); /* collective */
}

__global__ void gather_pp(const float2  *pp_lo, const float2 *pp_re, int n, const uint *iidx,
                          /**/ float2  *pp, float4  *zip0, ushort4 *zip1) {
    /* pp_lo, pp_re, pp: local, remote and output particles */
    int dw, ws, dwe;
    FLo f; /* "from" location */
    Da  d; /* data */
    TLo t; /* "to" location */
    ini_FLo(pp_lo, pp_re, &f);

    warpco(&ws, &dw);
    dwe = min(32, n - ws);
    if (ws + dw < n)
        FLo2D(&f, iidx[ws + dw], /**/ &d);

    ini_TLo(pp, zip0, zip1, /**/ &t);
    D2TLo(&d, ws, dw, dwe, /**/ &t);
}

}}} /* namespace */
