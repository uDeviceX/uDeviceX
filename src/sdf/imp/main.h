void alloc_quants(Sdf **pq) {
    Sdf *q;
    UC(emalloc(sizeof(Sdf), (void**)&q));

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray(&q->arrsdf, &fmt, make_cudaExtent(XTE, YTE, ZTE)));

    *pq = q;
}

void  free_quants(Sdf *q) {
    CC(cudaFreeArray(q->arrsdf));
    q->texsdf.destroy();
    UC(efree(q));
}

void ini(MPI_Comm cart, Sdf *q) {
    UC(sub::ini(cart, q->arrsdf, &q->texsdf));
}

void bulk_wall(const Sdf* q, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n) {
    UC(sub::bulk_wall(q->texsdf, /*io*/ s_pp, s_n, /*o*/ w_pp, w_n));
}

int who_stays(const Sdf *q, Particle *pp, int n, int nc, int nv, int *stay) {
    return sub::who_stays(q->texsdf, pp, n, nc, nv, /**/ stay);
}

void bounce(const Sdf *q, int n, /**/ Particle *pp) {
    UC(bounce_back(q->texsdf, n, /**/ pp));
}
