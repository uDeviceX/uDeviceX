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
            MPI_Cart_rank(cart, co_ne, &rnk_ne[i]);
            for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] - d[c];
            MPI_Cart_rank(cart, co_ne, &ank_ne[i]);
        }
    }

    void _shift_copy(const Solid *ss_src, const int n, const int code, /**/ Solid *ss_dst)
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

    void _post_recvcnt()
    {
        recv_counts[0] = 0;
        for (int i = 1; i < 27; ++i)
        {
            MPI_Request req;
            MPI_Irecv(&recv_counts[i], 1, MPI_INTEGER, ank_ne[i], i + 51024, cart, &req);
            recvcntreq.push_back(req);
        }
    }

    void init()
    {
        MPI_Comm_dup(m::cart, &cart);
    
        gen_ne(cart, /**/ rnk_ne, ank_ne);

        _post_recvcnt();
    }

    void close()
    {
        MPI_Comm_free(&cart);
    }


    void pack_sendcnt(const Solid *ss_hst, const int ns, const float *bbox)
    {
        for (int i = 0; i < 27; ++i) shalo[i].clear();

        std::vector<int> sids[27];
    
        const int L[3] = {XS, YS, ZS};
        const float M[3] = {XMARGIN_BB, YMARGIN_BB, ZMARGIN_BB};

        int vcode[3];

        const int dx = bbox[X] * 0.5;
        const int dy = bbox[Y] * 0.5;
        const int dz = bbox[Z] * 0.5;
    
        for (int i = 0; i < ns; ++i)
        {
            const float *r0 = ss_hst[i].com;
            const int sid = ss_hst[i].id;
            
            auto vcontrib = [&](float dx_, float dy_, float dz_) {
            
                const float r[3] = {r0[X] + dx_, r0[Y] + dy_, r0[Z] + dz_};

                for (int c = 0; c < 3; ++c)
                vcode[c] = (2 + (r[c] >= -L[c] / 2 + M[c]) + (r[c] >= L[c] / 2 - M[c])) % 3;

                const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

                if (sids[code].size() == 0 || sids[code].back() != sid)
                {
                    sids[code].push_back(sid);
                    shalo[code].push_back(ss_hst[i]);
                }
            };

            vcontrib(+dx, +dy, +dz);
            vcontrib(+dx, +dy, -dz);
            vcontrib(+dx, -dy, +dz);
            vcontrib(+dx, -dy, -dz);

            vcontrib(-dx, +dy, +dz);
            vcontrib(-dx, +dy, -dz);
            vcontrib(-dx, -dy, +dz);
            vcontrib(-dx, -dy, -dz);

            assert(sids[0].back() == sid);
            assert(sids[0].size() == i+1);
        }

        for (int i = 0; i < 27; ++i) send_counts[i] = shalo[i].size();

        for (int i = 1; i < 27; ++i)
        MPI_Isend(send_counts + i, 1, MPI_INTEGER, rnk_ne[i], i + 51024, cart, &sendcntreq[i - 1]);
    }

    int post()
    {
        {
            MPI_Status statuses[recvcntreq.size()];
            MPI_Waitall(recvcntreq.size(), &recvcntreq.front(), statuses);
            recvcntreq.clear();
        }

        int ncome = shalo[0].size(); // bulk
        for (int i = 1; i < 27; ++i)
        {
            int count = recv_counts[i];
            ncome += count;
            rhalo[i].resize(count);
        }

        MPI_Status statuses[26];
        MPI_Waitall(26, sendcntreq, statuses);

        for (int i = 1; i < 27; ++i)
        if (rhalo[i].size() > 0)
        {
            MPI_Request request;
            MPI_Irecv(rhalo[i].data(), rhalo[i].size(), Solid::datatype(), ank_ne[i], i + 51155, cart, &request);
            recvreq.push_back(request);
        }

        for (int i = 1; i < 27; ++i)
        if (shalo[i].size() > 0)
        {
            MPI_Request request;
            MPI_Isend(shalo[i].data(), shalo[i].size(), Solid::datatype(), rnk_ne[i], i + 51155, cart, &request);
            sendreq.push_back(request);
        }
        
        return ncome;
    }

    void unpack(/**/ Solid *ss_buf)
    {
        MPI_Status statuses[26];
        MPI_Waitall(recvreq.size(), &recvreq.front(), statuses);
        MPI_Waitall(sendreq.size(), &sendreq.front(), statuses);
        recvreq.clear();
        sendreq.clear();

        const int nbulk = shalo[0].size();
    
        // copy bulk
        for (int j = 0; j < nbulk; ++j) ss_buf[j] = shalo[0][j];

        // copy and shift halo
        for (int i = 1, start = nbulk; i < 27; ++i)
        {
            const int count = rhalo[i].size();

            if (count > 0)
            _shift_copy(rhalo[i].data(), count, i, /**/ ss_buf + start);

            start += count;
        }
        _post_recvcnt();
    }

    void pack_back(const Solid *ss_buf)
    {
        // prepare recv buffers

        for (int i = 1; i < 27; ++i) rhalo[i].resize(send_counts[i]);
        
        // bulk

        const int nbulk = shalo[0].size();
        rhalo[0].resize(nbulk);

        for (int j = 0; j < nbulk; ++j)
        rhalo[0][j] = ss_buf[j];

        // halo

        int start = nbulk;
        
        for (int i = 1; i < 27; ++i)
        {
            const int count = recv_counts[i];

            //printf("[%d] halo %d sending %d\n", m::rank, i, count);

            shalo[i].resize(count);

            for (int j = 0; j < count; ++j)
            shalo[i][j] = ss_buf[start + j];

            start += count;
        }
    }

    void post_back()
    {
        for (int i = 1; i < 27; ++i)
        if (rhalo[i].size() > 0)
        {
            MPI_Request request;
            MPI_Irecv(rhalo[i].data(), rhalo[i].size(), Solid::datatype(), rnk_ne[i], i + 51255, cart, &request);
            //printf("[%d] halo %d recv %d\n", m::rank, i, rhalo[i].size());
            recvreq.push_back(request);
        }

        for (int i = 1; i < 27; ++i)
        if (shalo[i].size() > 0)
        {
            MPI_Request request;
            MPI_Isend(shalo[i].data(), shalo[i].size(), Solid::datatype(), ank_ne[i], i + 51255, cart, &request);
            sendreq.push_back(request);
        }
    }

    void unpack_back(/**/ Solid *ss_hst)
    {
        MPI_Status statuses[26];
        MPI_Waitall(recvreq.size(), &recvreq.front(), statuses);
        MPI_Waitall(sendreq.size(), &sendreq.front(), statuses);
        recvreq.clear();
        sendreq.clear();

        const int nbulk = shalo[0].size();
    
        // copy bulk
        for (int j = 0; j < nbulk; ++j) ss_hst[j] = shalo[0][j];

        // add forces and torques from halo BB
        for (int i = 1; i < 27; ++i)
        {
            const int count = rhalo[i].size();

            for (int j = 0; j < count; ++j)
            {
                const int  my_id = rhalo[i][j].id;
                const float *sfo = rhalo[i][j].fo;
                const float *sto = rhalo[i][j].to;

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
