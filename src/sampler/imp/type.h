struct Sampler {
    int3 L, N, M; /* subdomain size, grid size, margin */
    float4 *pp;   /* density, velocity                 */
    float *ss;    /* stresses                          */
};
