namespace rex {
void halo() {
    int i, n;
    ParticlesWrap halos[26];
    for (i = 0; i < 26; ++i) {
        n = remote[i].n;
        halos[i] = ParticlesWrap(remote[i].dstate, n, remote[i].ff);
    }

    dSync();
    if (fsiforces)     fsi::halo(halos);
    if (contactforces) cnt::halo(halos);
}
}
