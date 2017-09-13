
void free_map(/**/ Map *m) {
    CC(d::Free(m->counts));
    CC(d::Free(m->starts));    
    for (int i = 0; i < NFRAGS; ++i)
        CC(d::Free(m->ids[i]));
}
