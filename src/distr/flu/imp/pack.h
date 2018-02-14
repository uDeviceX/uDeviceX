static int reduce(int n, const int d[]) {
    int s, i;
    for (i = s = 0; i < n; ++i) s += d[i];
    return s;
}

static void pack_pp(const DMap m, const Particle *pp, /**/ dBags bags) {
    int n;
    const int S = sizeof(Particle) / sizeof(float2);
    float2p26 wrap;
    bag2Sarray(bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dev::pack<float2, S>), (k_cnf(S*n)), ((const float2*)pp, m, /**/ wrap));
}

static void pack_ii(const DMap m, const int *ii, /**/ dBags bags) {
    int n;
    const int S = 1;
    intp26 wrap;
    bag2Sarray(bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dev::pack<int, S>), (k_cnf(S*n)), (ii, m, /**/ wrap));
}

void dflu_pack(const FluQuants *q, /**/ DFluPack *p) {
    pack_pp(p->map, q->pp, /**/ p->dpp);
    if (p->opt.ids)    pack_ii(p->map, q->ii, /**/ p->dii);
    if (p->opt.colors) pack_ii(p->map, q->cc, /**/ p->dcc);
}

struct ExceedData { int cap, cnt, fid; };
enum   {OK, FAIL};
static int check_counts(int nfrags, const int *counts, const hBags *hpp, /**/ ExceedData *e) {
    int fid, cnt, cap;
    for (fid = 0; fid < nfrags; ++fid) {
        cnt = counts[fid];
        cap = comm_get_number_capacity(fid, hpp);
        if (cnt > cap) {
            e->cap = cap; e->cnt = cnt; e->fid = fid;
            return FAIL;
        }        
    }
    return OK;
}
static void fail_exceed(ExceedData *e) {
    enum {X, Y, Z};
    int cap, cnt, fid, d[3];
    cap = e->cap; cnt = e->cnt; fid = e->fid;
    fraghst::i2d3(fid, d);
    ERR("exceed capacity, fragment %d = [%d %d %d]: %d/%d",
        fid, d[X], d[Y], d[Z], cnt, cap);
}
static void dflu_download0(DFluPack *p) {
    size_t sz;
    int *cnt;
    sz = NFRAGS * sizeof(int);
    cnt = p->map.hcounts;
    dSync(); /* wait for pack kernels */
    memcpy(p->hpp.counts, cnt, sz);
    if (p->opt.ids)    memcpy(p->hii.counts, cnt, sz);
    if (p->opt.colors) memcpy(p->hcc.counts, cnt, sz);
    p->nhalo = reduce(NFRAGS, cnt);
}
void dflu_download(DFluPack *p, /**/ DFluStatus *s) {
    ExceedData e;
    int r;
    int *cnt;
    cnt = p->map.hcounts;    
    r = check_counts(NFRAGS, cnt, &p->hpp, /**/ &e);
    if (r == OK) {
        UC(dflu_download0(p));
    }
    else {
        if   (dflu_status_nullp(s)) UC(fail_exceed(&e));
        else UC(dflu_status_exceed(e.fid, e.cnt, e.cap, /**/ s));
    } 
}
