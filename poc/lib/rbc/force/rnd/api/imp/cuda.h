struct Generator_st { curandGenerator_t g; };

static int CreateGenerator0(curandGenerator_t *g) {
    return curandCreateGenerator(g, CURAND_RNG_PSEUDO_DEFAULT);
}
int CreateGenerator(Generator_t *pg) {
    int rc;
    Generator_st *s;
    EMALLOC(1, &s);
    rc = CreateGenerator0(&s->g);
    *pg = s;
    return rc;
}

int DestroyGenerator(Generator_t s) {
    int rc;
    rc = curandDestroyGenerator(s->g);
    EFREE(s);
    return rc;
}

int SetPseudoRandomGeneratorSeed(Generator_t s,  long seed) {
    return curandSetPseudoRandomGeneratorSeed(s->g, seed);
}

int GenerateNormal(Generator_t s, float *outputPtr, size_t n) {
    float mean, std;
    mean = 0; std = 1;
    return curandGenerateNormal(s->g, outputPtr, n, mean, std);
}
