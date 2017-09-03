namespace rex {
void halo(int counts[26], RFrag *remote) {
    int i, n;
    ParticlesWrap halos[26];

    Pap26 PP; /* usage: pp = PP.d[i], n = nn[i]  */
    Fop26 FF;

    for (i = 0; i < 26; ++i) {
        PP.d[i] = remote[i].pp;
        FF.d[i] = remote[i].ff;

        n = counts[i];
        halos[i] = ParticlesWrap(remote[i].pp, n, remote[i].ff);
    }

    dSync();
    if (fsiforces)     fsi::halo(PP, FF, counts);
    if (contactforces) cnt::halo(halos, PP, FF, counst);
}
}
