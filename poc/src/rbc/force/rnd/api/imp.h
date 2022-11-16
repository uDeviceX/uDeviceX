namespace rnd_api {
typedef struct Generator_st *Generator_t;
int CreateGenerator(Generator_t *g);
int SetPseudoRandomGeneratorSeed(Generator_t g,  long seed);
int DestroyGenerator(Generator_t g);
int GenerateNormal(Generator_t g, float *outputPtr, size_t n);
}
