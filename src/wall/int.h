struct Quants {
    Particle *pp;
    int n;
    Logistic::KISS *rnd;
    x::Clist *cells;
    cudaTextureObject_t texstart;
};

void alloc_quants(Quants *q) {
    // allocated in wall::init
    //CC(cudaMalloc(&(q->pp), MAX_PART_NUM * sizeof(Particle)));
    q->rnd   = new Logistic::KISS;
    q->cells = new x::Clist(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM);
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp));
    delete q->cells;
}

int create(int n, Particle* pp, Quants *q) {
    n = sub::init(pp, n, &q->pp, &q->n);
    sub::build_cells(q->n, /**/ q->pp, q->cells);    
    
    return n;
}

void interactions(const Quants q, const int type, const Particle *pp, const int n, Force *ff) {
    sub::interactions(type, pp, n, q.rnd->get_float(), q.cells, q.pp, q.n, /**/ ff);
}
