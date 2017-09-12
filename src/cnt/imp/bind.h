static void bind(const int *const starts, const int *const cellentries,
          const int nc, std::vector<ParticlesWrap> wr) {
    size_t textureoffset;
    int ncells, n, i;
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
    n = wr.size();
    assert(n <= MAX_OBJ_TYPES);
    for (i = 0; i < n; ++i) {
        ns[i] = wr[i].n;
        ps[i] = (float2*)wr[i].p;
        fs[i] = (float*)wr[i].f;
    }

    CC(cudaMemcpyToSymbolAsync(dev::g::ns, ns, sizeof(int)*n, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::csolutes, ps, sizeof(float2*)*n, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::csolutesacc, fs, sizeof(float*)*n, 0, H2D));
}
