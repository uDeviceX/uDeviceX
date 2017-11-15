void ini0(D *d, int n) {
    Dalloc(&d->r, n);
    CU(curandCreateGenerator(&d->g, CURAND_RNG_PSEUDO_DEFAULT));
    CU(curandSetPseudoRandomGeneratorSeed(d->g,  1234ULL));
}
void ini(D **pd, int n) {
    D* d;
    d = (D*)malloc(sizeof(D));
    ini0(d, n);
    *pd = d;
}

void fin0(D *d) {
    Dfree(d->r);
    CU(curandDestroyGenerator(d->g));
}
void fin(D *d) {
    fin0(d);
    free(d);
}
void gen(D *d, int n) {
    float mean, std;
    mean = 0; std = 1;
    CU(curandGenerateNormal(d->g, d->r, n, mean, std));
}
float get_hst(const D *d, int i) {
    float x;
    cD2H(&x, &d[i], 1);
    return x;
}
