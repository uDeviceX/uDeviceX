struct Quants {
    Particle *pp, *pp_hst;
    Texo <float2> texvert;       /* vertices */

    int *adj0, *adj1;            /*adjacency lists */
    Texo <int> texadj0, texadj1;

    int4 *tri;                   /* triangles */
    Texo <int4> textri;

    int *tri_hst;
    float *av;
};


void alloc_quants(Quants *q) {
    mpDeviceMalloc(&q->pp);
    mpHostMalloc(&q->pp_hst);

    q->tri = q->adj0 = q->adj1 = NULL;
    
    q->tri_hst = new int[MAX_FACE_NUM];
    CC(cudaMalloc(&q->av, MAX_CELL_NUM));
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp));
    CC(cudaFree(q->av));

    if (q->tri)  CC(cudaFree(q->tri));
    if (q->adj0) CC(cudaFree(q->adj0));
    if (q->adj1) CC(cudaFree(q->adj1));

    q->textri.destroy();
    q->texadj0.destroy();
    q->texadj1.destroy();
    q->texvert.destroy();
}


