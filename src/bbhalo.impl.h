namespace bbhalo
{
enum {X, Y, Z};

#define i2del(i) {((i) + 1) % 3 - 1,            \
            ((i) / 3 + 1) % 3 - 1,              \
            ((i) / 9 + 1) % 3 - 1}

/* generate ranks and anti-ranks of the neighbors */
void gen_ne(const MPI_Comm cart, /* */ int* rnk_ne, int* ank_ne)
{
    rnk_ne[0] = m::rank;
    for (int i = 1; i < 27; ++i)
    {
        int d[3] = i2del(i); /* index to delta */
        int co_ne[3];
        for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] + d[c];
        l::m::Cart_rank(cart, co_ne, &rnk_ne[i]);
        for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] - d[c];
        l::m::Cart_rank(cart, co_ne, &ank_ne[i]);
    }
}

static void _shift_copy_ss(const Solid *ss_src, const int n, const int code, /**/ Solid *ss_dst)
{
    const int d[3] = {(code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1, (code / 9 + 1) % 3 - 1};
    const int L[3] = {XS, YS, ZS};

    for (int j = 0; j < n; ++j)
    {
        Solid snew = ss_src[j];

        for (int c = 0; c < 3; ++c)
        snew.com[c] -= d[c] * L[c];

        ss_dst[j] = snew;
    }
}

static void _shift_hst(const float3 s, const int n, /**/ Particle *pp)
{
    for (int i = 0; i < n; ++i)
    {
        float *r = pp[i].r;
        r[X] += s.x; r[Y] += s.y; r[Z] += s.z;
    }
}

static __global__ void _shift_dev(const float3 s, const int n, /**/ Particle *pp)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        float *r = pp[i].r;
        r[X] += s.x; r[Y] += s.y; r[Z] += s.z;
    }
}

template <bool tohst>
static void _shift_copy_pp(const Particle *ss_src, const int n, const int nps, const int code, /**/ Particle *ss_dst)
{
    const int d[3] = {(code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1, (code / 9 + 1) % 3 - 1};
    const float3 shift = make_float3(-XS * d[X], -YS * d[Y], -ZS * d[Z]);

    if (tohst)
    {
        memcpy(ss_dst, ss_src, n*nps*sizeof(Particle));
        _shift_hst(shift, n*nps, /**/ ss_dst); 
    }
    else
    {
        cH2D(ss_dst, ss_src, n*nps);
        _shift_dev <<< k_cnf(n*nps) >>> (shift, n*nps, /**/ ss_dst);
    }        
}

void _post_recvcnt()
{
    recv_counts[0] = 0;
    for (int i = 1; i < 27; ++i)
    {
        MPI_Request req;
        l::m::Irecv(&recv_counts[i], 1, MPI_INTEGER, ank_ne[i], i + BT_C_BBHALO, cart, &req);
        recvcntreq.push_back(req);
    }
}

void ini()
{
    l::m::Comm_dup(m::cart, &cart);
    
    gen_ne(cart, /**/ rnk_ne, ank_ne);

    _post_recvcnt();
}

void fin()
{
    l::m::Comm_free(&cart);
}


template <bool fromhst>
void pack_sendcnt(const Solid *ss_hst, const int ns, const Particle *pp, const int nps, const float3 *minbb, const float3 *maxbb)
{
    for (int i = 0; i < 27; ++i) sshalo[i].clear();

    std::vector<int> hhindices[27]; /* who will be sent in which buffer */
    
    const int L[3] = {XS, YS, ZS};
    const float M[3] = {XBBM, YBBM, ZBBM};

    int vcode[3];
    
    for (int i = 0; i < ns; ++i)
    {
        const float3 minb = minbb[i];
        const float3 maxb = maxbb[i];

        // i contributes to my node
        hhindices[0].push_back(i);
            
        auto hhcontrib = [&](int dx, int dy, int dz) {

            const float r[3] = {dx ? minb.x : maxb.x,
                                dy ? minb.y : maxb.y,
                                dz ? minb.z : maxb.z};

            for (int c = 0; c < 3; ++c)
            vcode[c] = (2 + (r[c] >= -L[c] / 2 + M[c]) + (r[c] >= L[c] / 2 - M[c])) % 3;

            const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

            if (hhindices[code].size() == 0 || hhindices[code].back() != i)
            hhindices[code].push_back(i);
        };
            
        hhcontrib(0, 0, 0);
        hhcontrib(0, 0, 1);
        hhcontrib(0, 1, 0);
        hhcontrib(0, 1, 1);
            
        hhcontrib(1, 0, 0);
        hhcontrib(1, 0, 1);
        hhcontrib(1, 1, 0);
        hhcontrib(1, 1, 1);
    }
        
    // resize packs
    for (int i = 0; i < 27; ++i)
    {
        const int sz = hhindices[i].size();
        sshalo[i].resize(sz);
        pshalo[i].resize(sz*nps);
    }
        
    // copy data into packs
    for (int i = 0; i < 27; ++i)
    for (uint j = 0; j < hhindices[i].size(); ++j)
    {
        const int id = hhindices[i][j];
        sshalo[i][j] = ss_hst[id];

        if (fromhst) memcpy(pshalo[i].data() + j*nps, pp + id*nps, nps*sizeof(Particle));
        else  cD2H(pshalo[i].data() + j*nps, pp + id*nps, nps);
    }

    // send counts
    for (int i = 0; i < 27; ++i) send_counts[i] = sshalo[i].size();

    for (int i = 1; i < 27; ++i)
    l::m::Isend(send_counts + i, 1, MPI_INTEGER, rnk_ne[i], i + BT_C_BBHALO, cart, &sendcntreq[i - 1]);
}

int post(const int nps)
{
    {
        MPI_Status statuses[recvcntreq.size()];
        l::m::Waitall(recvcntreq.size(), &recvcntreq.front(), statuses);
        recvcntreq.clear();
    }

    int ncome = sshalo[0].size(); // bulk
    for (int i = 1; i < 27; ++i)  // halo
    {
        int count = recv_counts[i];
        ncome += count;
        srhalo[i].resize(count);
        prhalo[i].resize(count*nps);
    }

    MPI_Status statuses[26];
    l::m::Waitall(26, sendcntreq, statuses);

    for (int i = 1; i < 27; ++i)
    if (srhalo[i].size() > 0)
    {
        MPI_Request request;
        l::m::Irecv(srhalo[i].data(), srhalo[i].size(), Solid::datatype(), ank_ne[i], i + BT_S_BBHALO, cart, &request);
        srecvreq.push_back(request);

        l::m::Irecv(prhalo[i].data(), prhalo[i].size(), Particle::datatype(), ank_ne[i], i + BT_P_BBHALO, cart, &request);
        precvreq.push_back(request);
    }

    for (int i = 1; i < 27; ++i)
    if (sshalo[i].size() > 0)
    {
        MPI_Request request;
        l::m::Isend(sshalo[i].data(), sshalo[i].size(), Solid::datatype(), rnk_ne[i], i + BT_S_BBHALO, cart, &request);
        ssendreq.push_back(request);

        l::m::Isend(pshalo[i].data(), pshalo[i].size(), Particle::datatype(), rnk_ne[i], i + BT_P_BBHALO, cart, &request);
        psendreq.push_back(request);
    }
        
    return ncome;
}

template <bool tohst>
void unpack(const int nps, /**/ Solid *ss_buf, Particle *pp_buf)
{
    MPI_Status statuses[26];
    l::m::Waitall(srecvreq.size(), &srecvreq.front(), statuses);
    l::m::Waitall(ssendreq.size(), &ssendreq.front(), statuses);
    srecvreq.clear(); ssendreq.clear();
        
    l::m::Waitall(precvreq.size(), &precvreq.front(), statuses);
    l::m::Waitall(psendreq.size(), &psendreq.front(), statuses);
    precvreq.clear(); psendreq.clear();
        
    const int nbulk = sshalo[0].size();
    
    // copy bulk
    for (int j = 0; j < nbulk; ++j)
    ss_buf[j] = sshalo[0][j];
        
    if (tohst)  memcpy(pp_buf, pshalo[0].data(), nbulk*nps*sizeof(Particle));
    else          cH2D(pp_buf, pshalo[0].data(), nbulk*nps);
        
    // copy and shift halo
    for (int i = 1, start = nbulk; i < 27; ++i)
    {
        const int count = srhalo[i].size();

        if (count > 0)
        {
            _shift_copy_ss(srhalo[i].data(), count, i, /**/ ss_buf + start);
            _shift_copy_pp <tohst> (prhalo[i].data(), count, nps, i, /**/ pp_buf + start * nps);
        }
            
        start += count;
    }
    _post_recvcnt();
}

void pack_back(const Solid *ss_buf)
{
    // prepare recv buffers

    for (int i = 1; i < 27; ++i) srhalo[i].resize(send_counts[i]);
        
    // bulk

    const int nbulk = sshalo[0].size();
    srhalo[0].resize(nbulk);

    for (int j = 0; j < nbulk; ++j)
    srhalo[0][j] = ss_buf[j];

    // halo

    int start = nbulk;
        
    for (int i = 1; i < 27; ++i)
    {
        const int count = recv_counts[i];

        //printf("[%d] halo %d sending %d\n", m::rank, i, count);

        sshalo[i].resize(count);

        for (int j = 0; j < count; ++j)
        sshalo[i][j] = ss_buf[start + j];

        start += count;
    }
}

void post_back()
{
    for (int i = 1; i < 27; ++i)
    if (srhalo[i].size() > 0)
    {
        MPI_Request request;
        l::m::Irecv(srhalo[i].data(), srhalo[i].size(), Solid::datatype(), rnk_ne[i], i + BT_S2_BBHALO, cart, &request);
        //printf("[%d] halo %d recv %d\n", m::rank, i, srhalo[i].size());
        srecvreq.push_back(request);
    }

    for (int i = 1; i < 27; ++i)
    if (sshalo[i].size() > 0)
    {
        MPI_Request request;
        l::m::Isend(sshalo[i].data(), sshalo[i].size(), Solid::datatype(), ank_ne[i], i + BT_S2_BBHALO, cart, &request);
        ssendreq.push_back(request);
    }
}

void unpack_back(/**/ Solid *ss_hst)
{
    MPI_Status statuses[26];
    l::m::Waitall(srecvreq.size(), &srecvreq.front(), statuses);
    l::m::Waitall(ssendreq.size(), &ssendreq.front(), statuses);
    srecvreq.clear();
    ssendreq.clear();

    const int nbulk = sshalo[0].size();
    
    // copy bulk
    for (int j = 0; j < nbulk; ++j) ss_hst[j] = sshalo[0][j];

    // add forces and torques from halo BB
    for (int i = 1; i < 27; ++i)
    {
        const int count = srhalo[i].size();

        for (int j = 0; j < count; ++j)
        {
            const int  my_id = srhalo[i][j].id;
            const float *sfo = srhalo[i][j].fo;
            const float *sto = srhalo[i][j].to;

            int k = -1;
            for (int kk = 0; kk < nbulk; ++kk)
            {
                const int kid = ss_hst[kk].id;

                if (kid == my_id)
                {
                    float *fo = ss_hst[kk].fo;
                    float *to = ss_hst[kk].to;

                    fo[X] += sfo[X];
                    fo[Y] += sfo[Y];
                    fo[Z] += sfo[Z];

                    to[X] += sto[X];
                    to[Y] += sto[Y];
                    to[Z] += sto[Z];

                    k = kk;
                    break;
                }
            }
            assert(k != -1);
        }
    }
}
    
}
