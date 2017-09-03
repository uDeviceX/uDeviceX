namespace rex {
void halo(int counts[26], RFrag *remote) {
    int i, n;
    ParticlesWrap halos[26];
    for (i = 0; i < 26; ++i) {
        n = counts[i];
        halos[i] = ParticlesWrap(remote[i].pp, n, remote[i].ff);
    }

    dSync();
    if (fsiforces)     fsi::halo(halos);
    if (contactforces) cnt::halo(halos);
}
}
