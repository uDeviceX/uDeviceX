struct Quants {
    cudaArray *arrsdf;
    sub::dev::tex3Dca<float> texsdf;
};

void alloc_quants(Quants *q) {
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray(&q->arrsdf, &fmt, make_cudaExtent(XTE, YTE, ZTE)));
}

void  free_quants(Quants *q) {
    CC(cudaFreeArray(q->arrsdf));
    q->texsdf.destroy();
}

void ini(Quants *q) {
    sub::ini(q->arrsdf, &q->texsdf);
}

int who_stays(const Quants q, Particle *pp, int n, int nc, int nv, int *stay) {
    return sub::who_stays(q.texsdf, pp, n, nc, nv, /**/ stay);
}
