template <typename T, int N>
void alloc_pinned(const int i, const int sz, /**/ Pbufs<T, N> *b) {
    if (sz){
        CC(cudaHostAlloc(&b->hst[i], sizeof(T) * sz, cudaHostAllocMapped));
        CC(cudaHostGetDevicePointer(&b->dp[i], b->hst[i], 0));
    } else {
        b->hst[i] = NULL;
    }
}

template <typename T, int N>
void alloc_dev(/**/ Pbufs<T, N> *b) {
    CC(cudaMalloc(&b->dev, SZ_PTR_ARR(b->dp)));
    CC(cudaMemcpy(b->dev, b->dp, sizeof(b->dp), H2D));
}

template <typename T, int N>
void dealloc(Pbufs<T, N> *b) {
    for (int i = 0; i < N; ++i) {
        if (b->dp[i] != NULL) CC(cudaFreeHost(b->hst[i]));
    }
    CC(cudaFree(b->dev));
}
