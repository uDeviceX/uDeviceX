static void get_start(const SFrag sfrag[26], /**/ int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * ((sfrag[i].n + 15) / 16);
}

void interactions(const SFrag26 ssfrag, const Frag26 ffrag, const Rnd26 rrnd, /**/ float *ff) {
    int27 start;
    int n; /* number of threads */
    get_start(ssfrag.d, /**/ start.d);
    n = start.d[26];
    KL(dev::force, (k_cnf(n)), (start, ssfrag, ffrag, rrnd, /**/ ff));
}
