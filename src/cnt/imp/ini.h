void ini(Contact *c) {
    clist::ini(XS, YS, ZS, /**/ &c->cells);
    clist::ini_map(MAX_OBJ_TYPES, &c->cells, /**/ &c->cmap);
    c->rgen = new rnd::KISS(7119 - m::rank, 187 + m::rank, 18278, 15674);
}
