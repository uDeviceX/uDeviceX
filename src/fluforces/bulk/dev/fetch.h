template <typename Parray>
static __device__ void bpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = a.pp[2*i + 0];
    v = a.pp[2*i + 1];

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}

static __device__ void fetch(BPaArray_v a, int i, PairPa *p) {
    bpa_fetch_p(a, i, p);
}

static __device__ void fetch(BPaCArray_v a, int i, PairPa *p) {
    bpa_fetch_p(a, i, p);
    p->color = a.cc[i];
}


template <typename Parray>
static __device__ void tbpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = fetch(a.pp, 2*i + 0);
    v = fetch(a.pp, 2*i + 1);

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}

static __device__ void fetch(TBPaArray_v a, int i, PairPa *p) {
    tbpa_fetch_p(a, i, p);
}

static __device__ void fetch(TBPaCArray_v a, int i, PairPa *p) {
    tbpa_fetch_p(a, i, p);    
    p->color = fetch(a.cc, i);
}
