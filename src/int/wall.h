namespace wall {

struct Quants {
  Particle *pp;
  int n;
  Logistic::KISS *rnd;
  x::Clist *cells;
};

void alloc_quants(Quants *q) {
    // allocated in wall::init
    //CC(cudaMalloc(&(q->pp), MAX_PART_NUM * sizeof(Particle)));
    q->rnd = new Logistic::KISS;
}

void free_wuants(Quants *q) {
    CC(cudaFree(q->pp));
    delete q->cells;
}

int create(int n, Particle* pp, Quants *q) {
    // TMP !!
    wall::w_pp = q->pp;
    wall::w_n = q->n;
    wall::trunk = q->rnd;
    wall::cells = q->cells;
    // !!
    
    n = wall::init(pp, n);

    // TMP !!
    q->pp = wall::w_pp;
    q->n  = wall::w_n;
    q->rnd = wall::trunk;
    q->cells = wall::cells;
    // !!
    return n;
}

void interactions(const Quants q, const int type, const Particle *pp, const int n, Force *ff) {
    /* unpack q TMP */
    wall::w_pp = q.pp;
    wall::w_n = q.n;
    wall::trunk = q.rnd;
    wall::cells = q.cells;
        
    wall::interactions(type, pp, n, ff);
}


} /* namespace wall */
