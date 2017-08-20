namespace rex {
void halo(int recv_counts[26]) {
    int i, n;
    ParticlesWrap halos[26];
    for (i = 0; i < 26; ++i) {
        n = recv_counts[i];
        halos[i] = ParticlesWrap(remote[i].pp, n, remote[i].ff);
    }

    dSync();
    if (fsiforces)     fsi::halo(halos);
    if (contactforces) cnt::halo(halos);
}
}
