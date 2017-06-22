struct Quants {
    cudaArray *arrsdf;
    tex3Dca<float> texsdf;
};

void alloc_quants(Quants *q) {
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray(&q->arrsdf, &fmt, make_cudaExtent(XTE, YTE, ZTE)));
}

void  free_quants(Quants *q) {
    if (q->arrsdf) {
        CC(cudaFreeArray(q->arrsdf));
    }
}

