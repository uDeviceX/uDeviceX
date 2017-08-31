namespace sdstr
{
template <bool hst>
void pack_sendcnt(const Solid *ss_hst, const Particle *pp, const int ns, const int nv) {
    const int L[3] = {XS, YS, ZS};
    int vcode[3];

    // decide where to put data
    std::vector<int> dstindices[27];

    for (int i = 0; i < ns; ++i) {
        const float *r = ss_hst[i].com;

        for (int c = 0; c < 3; ++c)
        vcode[c] = (2 + (r[c] >= -L[c] / 2) + (r[c] >= L[c] / 2)) % 3;

        const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
        dstindices[code].push_back(i);
    }

    // resize buufers

    for (int i = 0; i < 27; ++i) {
        const int c = dstindices[i].size();
        send_counts[i] = c;
        ssbuf[i].resize(c);
        psbuf[i].resize(c*nv);
    }

    nstay = send_counts[0];

    // send counts

    for (int i = 1; i < 27; ++i)
    m::Isend(send_counts + i, 1, MPI_INTEGER, rnk_ne[i], i + btc, cart, &sendcntreq[i - 1]);

    // copy data into buffers

    for (int i = 0; i < 27; ++i)
    for (int j = 0; j < send_counts[i]; ++j) {
        const int id = dstindices[i][j];
        ssbuf[i][j] = ss_hst[id];

        if (hst) memcpy(psbuf[i].data() + j*nv, pp + id*nv, nv*sizeof(Particle));
        else     CC(cudaMemcpyAsync(psbuf[i].data() + j*nv, pp + id*nv, nv * sizeof(Particle), D2H));
    }
    if (!hst) dSync();
}

template <bool hst>
static void shift_copy_pp(const Particle *pp_src, const int n, const int code, /**/ Particle *pp_dst) {
    const int d[3] = i2del(code);
    const float3 shift = make_float3(-d[X] * XS, -d[Y] * YS, -d[Z] * ZS);

    if (hst) {
        memcpy(pp_dst, pp_src, n*sizeof(Particle));
        shiftpp_hst(n, shift, /**/ pp_dst);
    }
    else {
        CC(cudaMemcpyAsync(pp_dst, pp_src, n * sizeof(Particle), H2D));
        KL(shiftpp_dev, (k_cnf(n)), (n, shift, /**/ pp_dst));
    }
}

template <bool hst>
void unpack(const int nv, /**/ Solid *ss_hst, Particle *pp) {
    MPI_Status statuses[26];
    m::Waitall(srecvreq.size(), &srecvreq.front(), statuses);
    m::Waitall(ssendreq.size(), &ssendreq.front(), statuses);
    srecvreq.clear(); ssendreq.clear();

    m::Waitall(precvreq.size(), &precvreq.front(), statuses);
    m::Waitall(psendreq.size(), &psendreq.front(), statuses);
    precvreq.clear(); psendreq.clear();

    // copy bulk
    for (int j = 0; j < nstay; ++j) ss_hst[j] = ssbuf[0][j];

    if (nstay) {
        if (hst) memcpy(pp, psbuf[0].data(), nstay*nv*sizeof(Particle));
        else       cH2D(pp, psbuf[0].data(), nstay*nv);
    }

    // copy and shift halo
    for (int i = 1, start = nstay; i < 27; ++i) {
        const int count = recv_counts[i];

        if (count) {
            shift_copy_ss       (srbuf[i].data(), count,      i, /**/ ss_hst + start);
            shift_copy_pp <hst> (prbuf[i].data(), count * nv, i, /**/ pp + start * nv);
        }

        start += count;
    }
    _post_recvcnt();
}

}
