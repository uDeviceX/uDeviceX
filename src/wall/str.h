void start(const int id, /**/ float4 *pp, int *n) {
    Particle pptmp = new Particle[MAX_NUM_PART];

    restart::read("wall", id, /**/ pptmp, n);

    if (*n) dev::particle2float4 <<<k_cnf(*n)>>> (pptmp, *n, /**/ pp);
    
    delete[] pptmp;
}
