
void flocal(Particle *pp, int *sstart, const int n, const float seed, /**/ Force *ff) {
    if (n) {
        Texo<float2> texpp;
        Texo<int> texstart;
        texpp.setup((float2 *) pp, 3*n);
        texstart.setup(sstart, XS*YS*ZS);
        dev::flocal <<< k_cnf(3*n)>>> (texpp, texstart, n, seed, /**/ (float *) ff);
        texpp.destroy();
        texstart.destroy();
        //printf("bye\n");
    }
}
