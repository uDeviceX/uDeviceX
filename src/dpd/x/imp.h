
void flocal(Particle *pp, int *sstart, const int n, const float seed, /**/ Force *ff) {
    if (n) {
        Texo<float2> texpp;
        Texo<int> texstart;
        texpp.setup((float2 *) pp, 3*n);
        texstart.setup(sstart, XS*YS*ZS);

        enum { THREADS = 256 };
        dev::flocal <<< ceiln((n), THREADS), THREADS >>> (texpp, texstart, n, seed, /**/ (float *) ff);

        texpp.destroy();
        texstart.destroy();
    }
}
