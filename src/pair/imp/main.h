void pair_ini(PairParams **par) {
    PairParams *p;
    UC(emalloc(sizeof(PairParams), (void **) par));
    p = *par;

    p->ncol = 0;
    p->lj = false;
}

void pair_fin(PairParams *p) {
    UC(efree(p));
}

