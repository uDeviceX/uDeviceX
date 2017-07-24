
void flocal(const Texo<float2> texpp, const int *sstart, const int n, const float seed, /**/ Force *ff) {
    if (n)
        dev::flocal <<< k_cnf(3*n)>>> (texpp, sstart, n, seed, /**/ ff);
}
