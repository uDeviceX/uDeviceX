#define _S_ static __device__
#define _I_ static __device__

/* bulk parray fetch */

template <typename Parray>
_S_ void bpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = a.pp[2*i + 0];
    v = a.pp[2*i + 1];

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}

_I_ void fetch(BPaArray_v a, int i, PairPa *p) {
    bpa_fetch_p(a, i, p);
}

_I_ void fetch(BPaCArray_v a, int i, PairPa *p) {
    bpa_fetch_p(a, i, p);
    p->color = a.cc[i];
}


/* texture bulk parray fetch */

template <typename Parray>
_S_ void tbpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = texo_fetch(a.pp, 2*i + 0);
    v = texo_fetch(a.pp, 2*i + 1);

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}

_I_ void fetch(TBPaArray_v a, int i, PairPa *p) {
    tbpa_fetch_p(a, i, p);
}

_I_ void fetch(TBPaCArray_v a, int i, PairPa *p) {
    tbpa_fetch_p(a, i, p);    
    p->color = texo_fetch(a.cc, i);
}

#undef _S_
#undef _I_
