struct Quants {
    cudaArray *arrsdf;
    cudaTextureObject_t texsdf;
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

