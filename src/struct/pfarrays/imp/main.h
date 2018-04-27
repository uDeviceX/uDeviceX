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

void pfarrays_push(PFarrays *pp, PFarray p) {
    int i = pp->n ++;
    if ( pp->n > MAX_SIZE )
        ERR("pushed too many pfarrays: %d/%d", pp->n, MAX_SIZE);
    pp->a[i] = p;
}

void pfarrays_push(PFarrays *pp, long n, PaArray p, FoArray f) {
    PFarray pf;
    pf.p = p;
    pf.n = n;
    pf.f = f;
    UC(pfarrays_push(pp, pf));
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

void pfarrays_get(int i, const PFarrays *pp, PFarray *p) {
    UC(pfarrays_get(i, pp, &p->n, &p->p, &p->f));
}
