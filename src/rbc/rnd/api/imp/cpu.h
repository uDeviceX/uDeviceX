struct Generator_st { };
int CreateGenerator(Generator_t*) { return 0; }
int DestroyGenerator(Generator_t) { return 0; }
int SetPseudoRandomGeneratorSeed(Generator_t s,  long seed) { return 0; }
int GenerateNormal(Generator_t s, float *outputPtr, size_t n) {
    return 0;
}
