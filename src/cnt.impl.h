namespace cnt {
void init() {
    cellsstart = new DeviceBuffer<int>(k_cnt::NCELLS + 16);
    cellscount = new DeviceBuffer<int>(k_cnt::NCELLS + 16);
    compressed_cellscount = new DeviceBuffer<unsigned char>(k_cnt::NCELLS + 16);

    cellsentries = new DeviceBuffer<int>;
    subindices = new DeviceBuffer<uchar4>;
    local_trunk = new Logistic::KISS;
    *local_trunk = Logistic::KISS(7119 - m::rank, 187 + m::rank, 18278, 15674);
}

void build_cells(std::vector<ParticlesWrap> wsolutes) {
    nsolutes = wsolutes.size();

    int ntotal = 0;
    for (int i = 0; i < (int) wsolutes.size(); ++i) ntotal += wsolutes[i].n;

    subindices->resize(ntotal);
    cellsentries->resize(ntotal);

    CC(cudaMemsetAsync(cellscount->D, 0, sizeof(int) * cellscount->S));


    int ctr = 0;
    for (int i = 0; i < (int) wsolutes.size(); ++i) {
        ParticlesWrap it = wsolutes[i];
        if (it.n)
        k_common::subindex_local<true><<<k_cnf(it.n)>>>
            (it.n, (float2 *)it.p, cellscount->D, subindices->D + ctr);
        ctr += it.n;
    }

    k_common::compress_counts<<<k_cnf(compressed_cellscount->S)>>>
        (compressed_cellscount->S, (int4 *)cellscount->D,
         (uchar4 *)compressed_cellscount->D);

    k_scan::scan(compressed_cellscount->D, compressed_cellscount->S,
                 (uint *)cellsstart->D);

    ctr = 0;
    for (int i = 0; i < (int) wsolutes.size(); ++i) {
        ParticlesWrap it = wsolutes[i];

        if (it.n)
        k_cnt::populate<<<k_cnf(it.n)>>>
            (subindices->D + ctr, cellsstart->D, it.n, i, ntotal,
             (k_cnt::CellEntry *)cellsentries->D);
        ctr += it.n;
    }

    k_cnt::bind(cellsstart->D, cellsentries->D, ntotal, wsolutes,
                cellscount->D);
}

void bulk(std::vector<ParticlesWrap> wsolutes) {
    if (wsolutes.size() == 0) return;

    for (int i = 0; i < (int) wsolutes.size(); ++i) {
        ParticlesWrap it = wsolutes[i];
        if (it.n)
        k_cnt::bulk_3tpp<<<k_cnf(3 * it.n)>>>
            ((float2 *)it.p, it.n, cellsentries->S, wsolutes.size(), (float *)it.f,
             local_trunk->get_float(), i);

    }
}

void halo(ParticlesWrap halos[26]) {
    int nremote_padded = 0;
    {
        int recvpackcount[26], recvpackstarts_padded[27];

        for (int i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;

        CC(cudaMemcpyToSymbolAsync(k_cnt::packcount, recvpackcount,
                                   sizeof(recvpackcount), 0, H2D));
        recvpackstarts_padded[0] = 0;
        for (int i = 0, s = 0; i < 26; ++i)
        recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));
        nremote_padded = recvpackstarts_padded[26];
        CC(cudaMemcpyToSymbolAsync
           (k_cnt::packstarts_padded, recvpackstarts_padded,
            sizeof(recvpackstarts_padded), 0, H2D));

        const Particle *recvpackstates[26];
        for (int i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;

        CC(cudaMemcpyToSymbolAsync(k_cnt::packstates, recvpackstates,
                                   sizeof(recvpackstates), 0,
                                   H2D));
        Force *packresults[26];
        for (int i = 0; i < 26; ++i) packresults[i] = halos[i].f;
        CC(cudaMemcpyToSymbolAsync(k_cnt::packresults, packresults,
                                   sizeof(packresults), 0, H2D));
    }

    if (nremote_padded)
    k_cnt::halo<<<k_cnf(nremote_padded)>>>
        (nremote_padded, cellsentries->S, nsolutes, local_trunk->get_float());


}

void close() {
    delete subindices;
    delete compressed_cellscount;
    delete cellsentries;
    delete cellsstart;
    delete  cellscount;
    delete local_trunk;
}
}
