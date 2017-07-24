
void flocal(Particle *pp, const int *sstart, const int n, const float seed, /**/ Force *ff) {
    if (n) {
        Texo<float2> texpp;
        texpp.setup((float2 *) pp, 3*n);
        dev::flocal <<< k_cnf(3*n)>>> (texpp, sstart, n, seed, /**/ (float *) ff);
        texpp.destroy();
        //printf("bye\n");
    }
}
