static void bind(const int *const starts, const int *const cellentries,
          const int nc, std::vector<ParticlesWrap> wr) {
    size_t textureoffset = 0;

    if (nc)
        CC(cudaBindTexture(&textureoffset, &dev::t::id, cellentries,
                           &dev::t::id.channelDesc,
                           sizeof(int) * nc));
    int ncells = XS * YS * ZS;
    CC(cudaBindTexture(&textureoffset, &dev::t::start, starts,
                       &dev::t::start.channelDesc, sizeof(int) * ncells));
    int n = wr.size();

    int ns[n];
    float2 *ps[n];
    float *fs[n];

    for (int i = 0; i < n; ++i) {
        ns[i] = wr[i].n;
        ps[i] = (float2*)wr[i].p;
        fs[i] = (float*)wr[i].f;
    }

    CC(cudaMemcpyToSymbolAsync(dev::g::ns, ns, sizeof(int)*n, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::csolutes, ps, sizeof(float2*)*n, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::csolutesacc, fs, sizeof(float*)*n, 0, H2D));
}
