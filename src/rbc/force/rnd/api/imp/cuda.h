struct Generator_st { curandGenerator_t g; };

static int CreateGenerator0(curandGenerator_t *g) {
    return curandCreateGenerator(g, CURAND_RNG_PSEUDO_DEFAULT);
}
int CreateGenerator(Generator_t *pg) {
    int rc;
    Generator_st *s;
    UC(emalloc(sizeof(Generator_st), (void**) &s));
    rc = CreateGenerator0(&s->g);
    *pg = s;
    return rc;
}

static int DestroyGenerator0(curandGenerator_t g) {
    return curandDestroyGenerator(g);
}
int DestroyGenerator(Generator_t s) {
    int rc;
    rc = DestroyGenerator0(s->g);
    free(s);
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
