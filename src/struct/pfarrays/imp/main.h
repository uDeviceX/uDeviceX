void pfarrays_ini(PFarrays **pfa) {
    EMALLOC(1, pfa);
    pfarrays_clear(*pfa);
}

void pfarrays_fin(PFarrays *p) {
    EFREE(p);
}

void pfarrays_clear(PFarrays *p) {
    p->n = 0;
}

void pfarrays_push(PFarrays *pf, long n, PaArray p, FoArray f) {
    int i = pf->n ++;
    PFarray *pfa;
    if ( pf->n > MAX_SIZE )
        ERR("pushed too many pfarrays: %d/%d", pf->n, MAX_SIZE);
    pfa = &pf->a[i];
    pfa->p = p;
    pfa->f = f;
    pfa->n = n;
}

int pfarrays_size(const PFarrays *p) {
    return p->n;
}

void pfarrays_get(int i, const PFarrays *pf, long *n, PaArray *p, FoArray *f) {
    const PFarray *pfa;

    if (i >= pf->n)
        ERR("Out of bounds: %d/%d", i, pf->n);
    
    pfa = &pf->a[i];
    *n = pfa->n;
    *p = pfa->p;
    *f = pfa->f;
}