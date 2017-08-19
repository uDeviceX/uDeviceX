namespace rex {
void halo() {
    ParticlesWrap halos[26];
    for (int i = 0; i < 26; ++i) halos[i] = ParticlesWrap(remote[i]->dstate, remote[i]->n, remote[i]->ff);

    dSync();
    if (fsiforces)     fsi::halo(halos);
    if (contactforces) cnt::halo(halos);
}
}
