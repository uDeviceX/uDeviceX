static void bind0(const int *const starts, const int *const cellentries,
                  const int nc, int nw, PaWrap *pw, FoWrap *fw) {
    size_t textureoffset;
    int ncells, i;
    int ns[MAX_OBJ_TYPES];
    float2 *ps[MAX_OBJ_TYPES];
    float *fs[MAX_OBJ_TYPES];

    textureoffset = 0;
    if (nc)
        CC(cudaBindTexture(&textureoffset, &dev::t::id, cellentries,
                           &dev::t::id.channelDesc,
                           sizeof(int) * nc));
    ncells = XS * YS * ZS;
    CC(cudaBindTexture(&textureoffset, &dev::t::start, starts,
                       &dev::t::start.channelDesc, sizeof(int) * ncells));

    assert(nw <= MAX_OBJ_TYPES);
    for (i = 0; i < nw; ++i) {
        ns[i] = pw[i].n;
        ps[i] = (float2*)pw[i].pp;
        fs[i] = (float*)fw[i].ff;
    }

    CC(cudaMemcpyToSymbolAsync(dev::g::ns, ns, sizeof(int)*nw, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::csolutes, ps, sizeof(float2*)*nw, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::csolutesacc, fs, sizeof(float*)*nw, 0, H2D));
}
