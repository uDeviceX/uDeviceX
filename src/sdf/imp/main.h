void ini(Sdf **pq) {
    Sdf *q;
    UC(emalloc(sizeof(Sdf), (void**)&q));

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray(&q->arrsdf, &fmt, make_cudaExtent(XTE, YTE, ZTE)));

    *pq = q;
}

void fin(Sdf *q) {
    CC(cudaFreeArray(q->arrsdf));
    fin(q);
    UC(efree(q));
}

void bounce(Wvel_v wv, Coords c, const Sdf *q, int n, /**/ Particle *pp) {
    UC(bounce_back(wv, c, q->texsdf, n, /**/ pp));
}
