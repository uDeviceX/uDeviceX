struct Quants {
    cudaArray *arrsdf;
    tex3Dca<float> texsdf;
};

void alloc_quants(Quants *q) {
    q->arrsdf = NULL;
}

void  free_quants(Quants *q) {
    if (q->arrsdf) {
        CC(cudaFreeArray(q->arrsdf));
        CC(cudaDestroyTextureObject(q->rexsdf));
    }
}

