void alloc_quants(Quants **pq) {
    Quants *q;
    UC(emalloc(sizeof(Quants), (void**)&q));

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray(&q->arrsdf, &fmt, make_cudaExtent(XTE, YTE, ZTE)));

    *pq = q;
}

void  free_quants(Quants *q) {
    CC(cudaFreeArray(q->arrsdf));
    q->texsdf.destroy();
    UC(efree(q));
}

void ini(MPI_Comm cart, Quants *q) {
    UC(sub::ini(cart, q->arrsdf, &q->texsdf));
}

void bulk_wall(const Quants* q, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n) {
    UC(sub::bulk_wall(q->texsdf, /*io*/ s_pp, s_n, /*o*/ w_pp, w_n));
}

int who_stays(const Quants *q, Particle *pp, int n, int nc, int nv, int *stay) {
    return sub::who_stays(q->texsdf, pp, n, nc, nv, /**/ stay);
}

void bounce(const Quants *q, int n, /**/ Particle *pp) {
    UC(sub::bounce(q->texsdf, n, /**/ pp));
}
