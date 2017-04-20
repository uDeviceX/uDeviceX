namespace rdstr
{
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

    void _post_recvcnt()
    {
        recv_counts[0] = 0;
        for (int i = 1; i < 27; ++i)
        {
            MPI_Request req;
            MPI_Irecv(&recv_counts[i], 1, MPI_INTEGER, ank_ne[i], i + 1024, cart, &req);
            recvcntreq.push_back(req);
        }
    }

    /* generate ranks and anti-ranks of the neighbors */
    void gen_ne(MPI_Comm cart, /* */ int* rnk_ne, int* ank_ne)
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

    void pack_sendcnt(const Solid *ss_hst, const int ns)
    {
        for (int i = 0; i < 27; ++i) sbuf[i].clear();
        
        const int L[3] = {XS, YS, ZS};
        int vcode[3];
        
        for (int i = 0; i < ns; ++i)
        {
            const float *p = ss_hst[i].com;
            
            for (int c = 0; c < 3; ++c)
            vcode[c] = (2 + (p[c] >= -L[c] / 2) + (p[c] >= L[c] / 2)) % 3;

            const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
            sbuf[code].push_back(ss_hst[i]);
        }

        for (int i = 0; i < 27; ++i) send_counts[i] = sbuf[i].size();
        nstay = send_counts[0];
        
        for (int i = 1; i < 27; ++i)
        MPI_Isend(send_counts + i, 1, MPI_INTEGER, rnk_ne[i], i + 1024, cart, &sendcntreq[i - 1]);
    }

    int post()
    {
        {
            MPI_Status statuses[recvcntreq.size()];
            MPI_Waitall(recvcntreq.size(), &recvcntreq.front(), statuses);
            recvcntreq.clear();
        }

        int ncome = 0;
        for (int i = 1; i < 27; ++i)
        {
            int count = recv_counts[i];
            ncome += count;
            rbuf[i].resize(count);
        }

        MPI_Status statuses[26];
        MPI_Waitall(26, sendcntreq, statuses);

        for (int i = 1; i < 27; ++i)
        if (rbuf[i].size() > 0)
        {
            MPI_Request request;
            MPI_Irecv(rbuf[i].data(), rbuf[i].size(), Solid::datatype(), ank_ne[i], i + 1155, cart, &request);
            recvreq.push_back(request);
        }

        for (int i = 1; i < 27; ++i)
        if (sbuf[i].size() > 0)
        {
            MPI_Request request;
            MPI_Isend(sbuf[i].data(), sbuf[i].size(), Solid::datatype(), rnk_ne[i], i + 1155, cart, &request);
            sendreq.push_back(request);
        }
        
        return nstay + ncome;
    }

    void shift_copy(const Solid *ss_src, const int n, const int code, /**/ Solid *ss_dst)
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

    void unpack(const int ns, /**/ Solid *ss_hst)
    {
        MPI_Status statuses[26];
        MPI_Waitall(recvreq.size(), &recvreq.front(), statuses);
        MPI_Waitall(sendreq.size(), &sendreq.front(), statuses);
        recvreq.clear();
        sendreq.clear();

        // copy bulk
        for (int j = 0; j < nstay; ++j) ss_hst[j] = sbuf[0][j];

        // copy and shift halo
        for (int i = 1, start = nstay; i < 27; ++i)
        {
            const int count = rbuf[i].size();

            if (count > 0)
            shift_copy(rbuf[i].data(), count, i, /**/ ss_hst + start);

            start += count;
        }
        _post_recvcnt();
    }

    void close()
    {
        MPI_Comm_free(&cart);
    }
}
