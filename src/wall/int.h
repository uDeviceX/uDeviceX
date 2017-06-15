struct Quants {
    float4 *pp;
    int n;
    Logistic::KISS *rnd;
    x::Clist *cells;
    cudaTextureObject_t texstart;
    cudaTextureObject_t texpp;
};

void alloc_quants(Quants *q) {
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

    MSG0("consolidating wall particles");

    CC(cudaMalloc(&q->pp, q->n * sizeof(float4)));

    if (q->n > 0)
    sub::dev::strip_solid4 <<<k_cnf(q->n)>>> (frozen, q->n, /**/ q->pp);
    
    sub::make_texstart(q->cells->start, q->cells->ncells, /**/ &q->texstart);
    sub::make_texpp   (q->pp,           q->n,             /**/ &q->texpp);
    
    CC(cudaFree(frozen));
    return n;
}

void interactions(const Quants q, const int type, const Particle *pp, const int n, Force *ff) {
    sub::interactions(type, pp, n, q.rnd->get_float(), q.texstart, q.texpp, q.n, /**/ ff);
}
