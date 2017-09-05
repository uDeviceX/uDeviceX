#define O(p, n) {dSync(); dbg::check_pos_pu(p, n, __FILE__, __LINE__, ""); dSync();}
namespace rex {

enum {OK, FAIL};
static struct {
    float x, y, z;
    int fid;
    int n;
    int i; /* paritcle id */
} context;
static int check_one(Particle p) {
    enum {X, Y, Z};
    float x, y, z;
    x = p.r[X]; y = p.r[Y]; z = p.r[Z];
    if (isnan(x)) {
        context.x = x;
        context.y = y;
        context.z = z;
        return FAIL;
    }
    return OK;
}
static int check_hst0(Particle *pp, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (check_one(pp[i]) != OK) {
            context.i = i;
            return FAIL;
        }
    }
    return OK;
}
static int check_hst(Pap26 PP, int counts[26]) {
    int n, fid;
    for (fid = 0; fid < 26; fid++) {
        n = counts[fid];
        if (check_hst0(PP.d[fid], n) != OK) {
            context.fid = fid;
            context.n = n;
            return FAIL;
        }
    }
    return OK;
}
static void report_hst0() {
    MSG("hst0: fid         : %d", context.fid);
    MSG("hst0: r           : [%g %g %g]", context.x, context.y, context.z);
    MSG("hst0: fid, n, i   : [%d %d %d]", context.fid, context.n, context.i);    
}
static void report_hst() {
    report_hst0();
    assert(0);
}

static void fill(Particle* pp, int n) {
    enum {X, Y, Z};
    Particle *p;
    float *r, *v;
    int i;
    for (i = 0; i < n; i++) {
        p = &pp[i];
        r = p->r;
        v = p->v;
        r[X] = 99; r[Y] = 999; r[Z] = 9999;
        v[X] = 77; v[Y] = 777; v[Z] = i;
    }
}

static void pre(ParticlesWrap *w, int nw) {
    using namespace sub;
    clear(nw, tp);
    scanA(w, nw, tp);
    scanB(nw, tp);
    copy_offset(nw, tp, /**/ ti);
    copy_starts(tp, /**/ ti);

    fill(buf_pi, MAX_PART_NUM);
    
    pack(w, nw, tp, /**/ buf);
}

static void rex0(ParticlesWrap *w, int nw) {
    using namespace sub;
    dSync();
    recvF(tc.ranks, tr.tags, tt, ti.counts, local);
    dSync();

    /** C **/
    recvC(tc.ranks, tr.tags, tt, recv_counts);
    sendC(tc.ranks, tt, ti.counts);
    s::waitC();
    r::waitC();

    /** P **/
    recvP(tc.ranks, tr.tags, tt, recv_counts, PP_pi);
    sendP(tc.ranks, tt, ti, buf_pi, ti.counts);
    s::waitP();
    r::waitP();
    if (check_hst(PP_pi, recv_counts) != OK) report_hst();

    if (!first) s::waitA(); else first = 0;

    dSync();
    for (int i = 0; i < 26; i++) {
        MSG("recv_counts[%d]: %d/%d", i, recv_counts[i], MAX_OBJ_DENSITY*frag_ncell(i));
        O(PP.d[i], recv_counts[i]);
    }
    if (fsiforces)     fsi::halo(PP, FF, recv_counts);
    dSync();
    if (contactforces) cnt::halo(PP, FF, recv_counts);
    dSync();

    dSync();
    sendF(tc.ranks, tt, recv_counts, FF_pi); /* (sic) */
    r::waitA();
    unpack(w, nw, tp);
}

void rex(std::vector<ParticlesWrap> w0) {
    int nw;
    ParticlesWrap *w;

    nw = w0.size();
    w  = w0.data();

    for (int i = 0; i < nw; ++i) O(w[i].p, w[i].n);
    
    if (nw) {
        pre(w, nw);
        rex0(w, nw);
    }
}

}
