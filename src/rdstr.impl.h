namespace rdstr
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
                  ((i) / 3 + 1) % 3 - 1,        \
                  ((i) / 9 + 1) % 3 - 1}

    static void _post_recvcnt()
    {
        recv_counts[0] = 0;
        for (int i = 1; i < 27; ++i)
        {
            MPI_Request req;
            MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, ank_ne[i], i + BT_C_RDSTR, cart, &req);
            recvcntreq.push_back(req);
        }
    }

    /* generate ranks and anti-ranks of the neighbors */
    static void gen_ne(MPI_Comm cart, /* */ int* rnk_ne, int* ank_ne)
    {
        rnk_ne[0] = m::rank;
        for (int i = 1; i < 27; ++i)
        {
            int d[3] = i2del(i); /* index to delta */
            int co_ne[3];
            for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] + d[c];
            MPI_Cart_rank(cart, co_ne, &rnk_ne[i]);
            for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] - d[c];
            MPI_Cart_rank(cart, co_ne, &ank_ne[i]);
        }
    }

    void init()
    {        
        MPI_Comm_dup(m::cart, &cart);
        gen_ne(cart,   rnk_ne, ank_ne); /* generate ranks and anti-ranks */

        _post_recvcnt();
    }

    template <bool hst>
    void pack_sendcnt(const Solid *ss_hst, const Particle *pp, const int ns, const int nv)
    {
        const int L[3] = {XS, YS, ZS};
        int vcode[3];

        // decide where to put data
        std::vector<int> dstindices[27];
        
        for (int i = 0; i < ns; ++i)
        {
            const float *r = ss_hst[i].com;
            
            for (int c = 0; c < 3; ++c)
            vcode[c] = (2 + (r[c] >= -L[c] / 2) + (r[c] >= L[c] / 2)) % 3;

            const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
            dstindices[code].push_back(i);
        }

        // resize buufers
        
        for (int i = 0; i < 27; ++i)
        {
            const int c = dstindices[i].size();
            send_counts[i] = c;
            ssbuf[i].resize(c);
            psbuf[i].resize(c*nv);
        }
        
        nstay = send_counts[0];

        // send counts
        
        for (int i = 1; i < 27; ++i)
        MPI_Isend(send_counts + i, 1, MPI_INTEGER, rnk_ne[i], i + BT_C_RDSTR, cart, &sendcntreq[i - 1]);

        // copy data into buffers

        for (int i = 0; i < 27; ++i)
        for (int j = 0; j < send_counts[i]; ++j)
        {
            const int id = dstindices[i][j];
            ssbuf[i][j] = ss_hst[id];

            if (hst)    memcpy(psbuf[i].data() + j*nv, pp + id*nv, nv*sizeof(Particle));
            else CC(cudaMemcpy(psbuf[i].data() + j*nv, pp + id*nv, nv*sizeof(Particle), D2H));
        }
    }

    int post(const int nv)
    {
        {
            MPI_Status statuses[27];
            MPI_Waitall(recvcntreq.size(), &recvcntreq.front(), statuses);
            recvcntreq.clear();
        }

        int ncome = 0;
        for (int i = 1; i < 27; ++i)
        {
            int count = recv_counts[i];
            ncome += count;
            srbuf[i].resize(count);
            prbuf[i].resize(count * nv);
        }

        MPI_Status statuses[26];
        MPI_Waitall(26, sendcntreq, statuses);

        for (int i = 1; i < 27; ++i)
        if (srbuf[i].size() > 0)
        {
            MPI_Request request;
            MPI_Irecv(srbuf[i].data(), srbuf[i].size(), Solid::datatype(), ank_ne[i], i + BT_S_RDSTR, cart, &request);
            srecvreq.push_back(request);

            MPI_Irecv(prbuf[i].data(), prbuf[i].size(), Particle::datatype(), ank_ne[i], i + BT_P_RDSTR, cart, &request);
            precvreq.push_back(request);
        }

        for (int i = 1; i < 27; ++i)
        if (ssbuf[i].size() > 0)
        {
            MPI_Request request;
            MPI_Isend(ssbuf[i].data(), ssbuf[i].size(), Solid::datatype(), rnk_ne[i], i + BT_S_RDSTR, cart, &request);
            ssendreq.push_back(request);

            MPI_Isend(psbuf[i].data(), psbuf[i].size(), Particle::datatype(), rnk_ne[i], i + BT_P_RDSTR, cart, &request);
            psendreq.push_back(request);
        }
        
        return nstay + ncome;
    }

    static void shift_copy_ss(const Solid *ss_src, const int n, const int code, /**/ Solid *ss_dst)
    {
        const int d[3] = i2del(code);
        const int L[3] = {XS, YS, ZS};

        for (int j = 0; j < n; ++j)
        {
            Solid snew = ss_src[j];

            for (int c = 0; c < 3; ++c)
            snew.com[c] -= d[c] * L[c];

            ss_dst[j] = snew;
        }
    }

    static void shiftpp_hst(const int n, const float3 s, /**/ Particle *pp)
    {
        for (int i = 0; i < n; ++i)
        {
            float *r = pp[i].r;
            r[X] += s.x; r[Y] += s.y; r[Z] += s.z;
        }
    }

    static __global__ void shiftpp_dev(const int n, const float3 s, /**/ Particle *pp)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n)
        {
            float *r = pp[i].r;
            r[X] += s.x; r[Y] += s.y; r[Z] += s.z;
        }
    }

    template <bool hst>
    static void shift_copy_pp(const Particle *pp_src, const int n, const int code, /**/ Particle *pp_dst)
    {
        const int d[3] = i2del(code);
        const float3 shift = make_float3(-d[X] * XS, -d[Y] * YS, -d[Z] * ZS);

        if (hst)
        {
            memcpy(pp_dst, pp_src, n*sizeof(Particle));
            shiftpp_hst(n, shift, /**/ pp_dst);
        }
        else
        {
            CC(cudaMemcpy(pp_dst, pp_src, n*sizeof(Particle), H2D));
            shiftpp_dev <<< k_cnf(n) >>>(n, shift, /**/ pp_dst);
        }
    }

    template <bool hst>
    void unpack(const int ns, const int nv, /**/ Solid *ss_hst, Particle *pp)
    {
        MPI_Status statuses[26];
        MPI_Waitall(srecvreq.size(), &srecvreq.front(), statuses);
        MPI_Waitall(ssendreq.size(), &ssendreq.front(), statuses);
        srecvreq.clear(); ssendreq.clear();
        
        MPI_Waitall(precvreq.size(), &precvreq.front(), statuses);
        MPI_Waitall(psendreq.size(), &psendreq.front(), statuses);
        precvreq.clear(); psendreq.clear();

        // copy bulk
        for (int j = 0; j < nstay; ++j) ss_hst[j] = ssbuf[0][j];

        if (hst)    memcpy(pp, psbuf[0].data(), nstay*nv*sizeof(Particle));
        else CC(cudaMemcpy(pp, psbuf[0].data(), nstay*nv*sizeof(Particle), H2D));
        

        // copy and shift halo
        for (int i = 1, start = nstay; i < 27; ++i)
        {
            const int count = recv_counts[i];

            if (count > 0)
            {
                shift_copy_ss       (srbuf[i].data(), count, i, /**/ ss_hst + start);
                shift_copy_pp <hst> (prbuf[i].data(), count * nv, i, /**/ pp + start * nv);
            }

            start += count;
        }
        _post_recvcnt();
    }

    void close()
    {
        MPI_Comm_free(&cart);
    }
}
