namespace sdstr
{
enum {X, Y, Z};

/* decode neighbors linear index to "delta"
   0 -> { 0, 0, 0}
   1 -> { 1, 0, 0}
   ...
   20 -> {-1, 0, -1}
   ...
   26 -> {-1, -1, -1}
*/
#define i2del(i) {((i) + 1) % 3 - 1,            \
            ((i) / 3 + 1) % 3 - 1,              \
            ((i) / 9 + 1) % 3 - 1}

static void _post_recvcnt() {
    recv_counts[0] = 0;
    for (int i = 1; i < 27; ++i) {
        MPI_Request req;
        l::m::Irecv(recv_counts + i, 1, MPI_INTEGER, ank_ne[i], i + btc, cart, &req);
        recvcntreq.push_back(req);
    }
}

/* generate ranks and anti-ranks of the neighbors */
static void gen_ne(MPI_Comm cart, /* */ int* rnk_ne, int* ank_ne) {
    rnk_ne[0] = m::rank;
    for (int i = 1; i < 27; ++i) {
        int d[3] = i2del(i); /* index to delta */
        int co_ne[3];
        for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] + d[c];
        l::m::Cart_rank(cart, co_ne, &rnk_ne[i]);
        for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] - d[c];
        l::m::Cart_rank(cart, co_ne, &ank_ne[i]);
    }
}

void ini(/*io*/ basetags::TagGen *tg) {
    l::m::Comm_dup(l::m::cart, &cart);
    gen_ne(cart,   rnk_ne, ank_ne); /* generate ranks and anti-ranks */

    btc = get_tag(tg);
    btp = get_tag(tg);
    bts = get_tag(tg);

    _post_recvcnt();
}

int post(const int nv) {
    {
        MPI_Status statuses[27];
        l::m::Waitall(recvcntreq.size(), &recvcntreq.front(), statuses);
        recvcntreq.clear();
    }

    int ncome = 0;
    for (int i = 1; i < 27; ++i) {
        int count = recv_counts[i];
        ncome += count;
        srbuf[i].resize(count);
        prbuf[i].resize(count * nv);
    }

    MPI_Status statuses[26];
    l::m::Waitall(26, sendcntreq, statuses);

    for (int i = 1; i < 27; ++i)
    if (srbuf[i].size() > 0) {
        MPI_Request request;
        l::m::Irecv(srbuf[i].data(), srbuf[i].size(), datatype::solid, ank_ne[i], i + bts, cart, &request);
        srecvreq.push_back(request);

        l::m::Irecv(prbuf[i].data(), prbuf[i].size(), datatype::particle, ank_ne[i], i + btp, cart, &request);
        precvreq.push_back(request);
    }

    for (int i = 1; i < 27; ++i)
    if (ssbuf[i].size() > 0) {
        MPI_Request request;
        l::m::Isend(ssbuf[i].data(), ssbuf[i].size(), datatype::solid, rnk_ne[i], i + bts, cart, &request);
        ssendreq.push_back(request);

        l::m::Isend(psbuf[i].data(), psbuf[i].size(), datatype::particle, rnk_ne[i], i + btp, cart, &request);
        psendreq.push_back(request);
    }

    return nstay + ncome;
}

static void shift_copy_ss(const Solid *ss_src, const int n, const int code, /**/ Solid *ss_dst) {
    const int d[3] = i2del(code);
    const int L[3] = {XS, YS, ZS};

    for (int j = 0; j < n; ++j) {
        Solid snew = ss_src[j];

        for (int c = 0; c < 3; ++c)
        snew.com[c] -= d[c] * L[c];

        ss_dst[j] = snew;
    }
}

static void shiftpp_hst(const int n, const float3 s, /**/ Particle *pp) {
    for (int i = 0; i < n; ++i) {
        float *r = pp[i].r;
        r[X] += s.x; r[Y] += s.y; r[Z] += s.z;
    }
}

static __global__ void shiftpp_dev(const int n, const float3 s, /**/ Particle *pp) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float *r = pp[i].r;
        r[X] += s.x; r[Y] += s.y; r[Z] += s.z;
    }
}

void fin() {
    l::m::Comm_free(&cart);
}
}
