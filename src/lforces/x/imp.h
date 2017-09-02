void flocal(Particle *pp, int *sstart, const int n, const float seed, /**/ Force *ff) {
    if (n) {
        Texo<float2> texpp;
        Texo<int> texstart;
        TE(&texpp, (float2*)pp, 3*n);
        TE(&texstart, sstart, XS*YS*ZS);

        enum { THREADS = 256 };
        KL(dev::flocal, (ceiln((3*n), THREADS), THREADS), (texpp, texstart, n, seed, /**/ (float *) ff));

        texpp.destroy();
        texstart.destroy();
    }
}
