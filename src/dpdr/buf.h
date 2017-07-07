typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

struct Sbufs {
    intp26 str, cnt, cum;      /* cellstarts for each fragment (frag coords)      */
    Particlep26 pp;            /* buffer of particles for each fragment           */
    intp26 ii;                 /* scattered indices                               */
    
    /* pinned buffers */
    Particlep26 ppdev, pphst;  /* pinned memory for transfering particles         */
    intp26 cumdev, cumhst;     /* pinned memory for transfering local cum sum     */
};

static void alloc_Sbuf_frag(const int i, const int est, const int nfragcells, /**/ Sbufs *b) {
    CC(cudaMalloc(&b->str.d[i], (nfragcells + 1) * sizeof(int)));
    CC(cudaMalloc(&b->cnt.d[i], (nfragcells + 1) * sizeof(int)));
    CC(cudaMalloc(&b->cum.d[i], (nfragcells + 1) * sizeof(int)));

    CC(cudaMalloc(&b->ii.d[i], est * sizeof(int)));
    CC(cudaMalloc(&b->pp.d[i], est * sizeof(Particle)));

    CC(cudaHostAlloc(&b->pphst.d[i], est * sizeof(Particle), cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&b->ppdev.d[i], b->pphst.d[i], 0));

    CC(cudaHostAlloc(&b->cumhst.d[i], est * sizeof(int), cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&b->cumdev.d[i], b->cumhst.d[i], 0));
};

static void free_Sbuf_frag(const int i, /**/ Sbufs *b) {
    CC(cudaFree(b->str.d[i]));
    CC(cudaFree(b->cnt.d[i]));
    CC(cudaFree(b->cum.d[i]));

    CC(cudaFree(b->ii.d[i]));
    CC(cudaFree(b->pp.d[i]));
        
    CC(cudaFreeHost(b->cumhst.d[i]));
    CC(cudaFreeHost(b->pphst.d[i]));
}

void alloc_Sbufs(const int26 estimates, const int26 nfragcells, /**/ Sbufs *b) {
    for (int i = 0; i < 26; ++i)
    alloc_Sbuf_frag(i, estimates.d[i], nfragcells.d[i], /**/ b);
}

void free_Sbufs(const int26 estimates, const int26 nfragcells, /**/ Sbufs *b) {
    for (int i = 0; i < 26; ++i)
    free_Sbuf_frag(i, /**/ b);
}


struct Rbufs {
    
    intp26 cum;                /* cellstarts for each fragment (frag coords)      */
    Particlep26 pp;            /* buffer of particles for each fragment           */

    /* pinned buffers */
    Particlep26 ppdev, pphst;  /* pinned memory for transfering particles         */
    intp26 cumdev, cumhst;     /* pinned memory for transfering local cum sum     */
};

static void alloc_Rbuf_frag(const int i, const int est, const int nfragcells, /**/ Rbufs *b) {
    CC(cudaMalloc(&b->cum.d[i], (nfragcells + 1) * sizeof(int)));
    CC(cudaMalloc(&b->pp.d[i], est * sizeof(Particle)));

    CC(cudaHostAlloc(&b->pphst.d[i], est * sizeof(Particle), cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&b->ppdev.d[i], b->pphst.d[i], 0));

    CC(cudaHostAlloc(&b->cumhst.d[i], est * sizeof(int), cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&b->cumdev.d[i], b->cumhst.d[i], 0));
};

static void free_Rbuf_frag(const int i, /**/ Rbufs *b) {
    CC(cudaFree(b->cum.d[i]));
    CC(cudaFree(b->pp.d[i]));
        
    CC(cudaFreeHost(b->cumhst.d[i]));
    CC(cudaFreeHost(b->pphst.d[i]));
}

void alloc_Rbufs(const int26 estimates, const int26 nfragcells, /**/ Rbufs *b) {
    for (int i = 0; i < 26; ++i)
    alloc_Rbuf_frag(i, estimates.d[i], nfragcells.d[i], /**/ b);
}

void free_Rbufs(const int26 estimates, const int26 nfragcells, /**/ Rbufs *b) {
    for (int i = 0; i < 26; ++i)
    free_Rbuf_frag(i, /**/ b);
}
