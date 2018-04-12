static void ini_grid(int3 L, int3 M, int3 N, Grid *g) {
    long n = N.x * N.y * N.z;
    g->L = L;
    g->M = M;
    g->N = N;

    Dalloc(&g->pp, n);
    Dalloc(&g->ss, 6*n);
}

static void fin_grid(Grid *g) {
    Dfree(g->pp);
    Dfree(g->ss);
}

void sampler_ini(int3 L, int3 M, int3 N, Sampler **s0) {
    Sampler *s;
    EMALLOC(1, s0);
    s = *s0;
    UC(ini_grid(L, M, N, &s->grid));
    UC(sampler_reset(s));
}

void sampler_fin(Sampler *s) {
    UC(fin_grid(&s->grid));
    EFREE(s);
}

void sampler_reset(Sampler *s) {
    
}

void sampler_add(const SampleData, Sampler*);
void sampler_dump(Sampler*);
