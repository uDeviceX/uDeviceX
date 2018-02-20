int CreateGenerator(Generator_t*) { return 0; }
int DestroyGenerator(Generator_t) { return 0; }
int SetPseudoRandomGeneratorSeed(Generator_t,  long seed) { srand(seed); return 0; }
int GenerateNormal(Generator_t, float *outputPtr, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) outputPtr[i] = gaussrand();
    return 0;
}
