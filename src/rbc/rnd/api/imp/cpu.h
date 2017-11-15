#error

struct Generator_st {
    int phase ;
    double V1, V2, S;
};
static int CreateGenerator0(curandGenerator_t* g) {
    //    return curandCreateGenerator(g, CURAND_RNG_PSEUDO_DEFAULT);
}
int CreateGenerator(Generator_t *pg) {
    //    int rc;
    //    Generator_st *s;
    //    s = (Generator_st*)malloc(sizeof(Generator_st));
    //    rc = CreateGenerator0(&s->g);
    //    *pg = s;
    return 0;
}

static int DestroyGenerator0(curandGenerator_t g) {
    //    return curandDestroyGenerator(g);
    return 0;
}
int DestroyGenerator(Generator_t s) {
    //    int rc;
    //    rc = DestroyGenerator0(s->g);
    //    free(s);
    return 0;
}

int SetPseudoRandomGeneratorSeed(Generator_t s,  long seed) {
    //    return curandSetPseudoRandomGeneratorSeed(s->g, seed);
    return 0;
}

int GenerateNormal(Generator_t s, float *outputPtr, size_t n) {
    //    float mean, std;
    //    mean = 0; std = 1;
    //    curandGenerateNormal(s->g, outputPtr, n, mean, std);
    return 0;
}
