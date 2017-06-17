namespace fsi {
void bind_solvent(SolventWrap wrap) {*wsolvent = wrap;}
void init() {
    local_trunk = new l::rnd::d::KISS;
    wsolvent    = new SolventWrap;
    *local_trunk = l::rnd::d::KISS(1908 - m::rank, 1409 + m::rank, 290, 12968);
}

void close() {
    delete local_trunk;
    delete wsolvent;
}

void bulk(std::vector<ParticlesWrap> wsolutes) {
    if (wsolutes.size() == 0) return;

    k_fsi::setup(wsolvent->p, wsolvent->n, wsolvent->cellsstart,
                 wsolvent->cellscount);



    for (std::vector<ParticlesWrap>::iterator it = wsolutes.begin();
         it != wsolutes.end(); ++it)
    if (it->n)
    k_fsi::
        interactions_3tpp<<<k_cnf(3 * it->n)>>>
        ((float2 *)it->p, it->n, wsolvent->n, (float *)it->f,
         (float *)wsolvent->f, local_trunk->get_float());


}

void halo(ParticlesWrap halos[26]) {
    k_fsi::setup(wsolvent->p, wsolvent->n, wsolvent->cellsstart,
                 wsolvent->cellscount);



    int nremote_padded = 0;

    {
        int recvpackcount[26], recvpackstarts_padded[27];

        for (int i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;

        CC(cudaMemcpyToSymbolAsync(k_fsi::packcount, recvpackcount,
                                   sizeof(recvpackcount), 0, H2D));

        recvpackstarts_padded[0] = 0;
        for (int i = 0, s = 0; i < 26; ++i)
        recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));

        nremote_padded = recvpackstarts_padded[26];

        CC(cudaMemcpyToSymbolAsync(
                                   k_fsi::packstarts_padded, recvpackstarts_padded,
                                   sizeof(recvpackstarts_padded), 0, H2D));
    }

    {
        const Particle *recvpackstates[26];

        for (int i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;

        CC(cudaMemcpyToSymbolAsync(k_fsi::packstates, recvpackstates,
                                   sizeof(recvpackstates), 0,
                                   H2D));
    }

    {
        Force *packresults[26];

        for (int i = 0; i < 26; ++i) packresults[i] = halos[i].f;

        CC(cudaMemcpyToSymbolAsync(k_fsi::packresults, packresults,
                                   sizeof(packresults), 0, H2D));
    }

    if (nremote_padded)
    k_fsi::
        interactions_halo<<<k_cnf(nremote_padded)>>>
        (nremote_padded, wsolvent->n, (float *)wsolvent->f,
         local_trunk->get_float());
}
}
