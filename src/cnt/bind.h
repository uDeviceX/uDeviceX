namespace cnt {
static void bind(const int *const starts, const int *const cellentries,
          const int nc, std::vector<ParticlesWrap> wr) {
    size_t textureoffset = 0;

    if (nc)
        CC(cudaBindTexture(&textureoffset, &k_cnt::t::id, cellentries,
                           &k_cnt::t::id.channelDesc,
                           sizeof(int) * nc));
    int ncells = XS * YS * ZS;
    CC(cudaBindTexture(&textureoffset, &k_cnt::t::start, starts,
                       &k_cnt::t::start.channelDesc, sizeof(int) * ncells));
    int n = wr.size();

    int ns[n];
    float2 *ps[n];
    float *fs[n];

    for (int i = 0; i < n; ++i) {
        ns[i] = wr[i].n;
        ps[i] = (float2*)wr[i].p;
        fs[i] = (float*)wr[i].f;
    }

    CC(cudaMemcpyToSymbolAsync(k_cnt::g::ns, ns, sizeof(int)*n, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(k_cnt::g::csolutes, ps, sizeof(float2*)*n, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(k_cnt::g::csolutesacc, fs, sizeof(float*)*n, 0, H2D));
}
}
