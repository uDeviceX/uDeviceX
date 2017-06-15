struct Quants {
    Particle *pp;
    int n;
    Logistic::KISS *rnd;
    x::Clist *cells;
    cudaTextureObject_t texstart;
};

void alloc_quants(Quants *q) {
    //CC(cudaMalloc(&(q->pp), MAX_PART_NUM * sizeof(float4)));
    q->rnd   = new Logistic::KISS;
    q->cells = new x::Clist(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM);
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp));
    delete q->cells;
}

int create(int n, Particle* pp, Quants *q) {
    Particle *frozen;
    CC(cudaMalloc(&frozen, sizeof(Particle) * MAX_PART_NUM));

    n = sub::init(pp, n, frozen, &q->n);
    sub::build_cells(q->n, /**/ frozen, q->cells);

    CC(cudaMalloc(&q->pp, q->n * sizeof(Particle)));
    cD2D(q->pp, frozen, q->n);
    
    sub::make_texstart(q->cells->start, q->cells->ncells, /**/ &q->texstart);
    
    CC(cudaFree(frozen));

    return n;
}

void interactions(const Quants q, const int type, const Particle *pp, const int n, Force *ff) {
    sub::interactions(type, pp, n, q.rnd->get_float(), q.texstart, q.pp, q.n, /**/ ff);
}
