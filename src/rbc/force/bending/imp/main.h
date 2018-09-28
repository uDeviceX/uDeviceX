struct Bending {
    struct Bending_vtable *vtable;
};

struct Bending_vtable {
    void (*apply)(Bending*, const RbcParams*, const RbcQuants*, /**/ Force*);
    void (*fin)(Bending*);
};

void bending_apply(Bending *q, const RbcParams *params, const RbcQuants *quants, /**/ Force *ff) {
    q->vtable->apply(q, params, quants, /**/ ff);
}

void bending_fin(Bending *q) {
    q->vtable->fin(q);
}

struct BendingKantor {
    Bending bending;
    Kantor *kantor;
};


static void method_kantor_apply(Bending *q, const RbcParams *params, const RbcQuants *quants, /**/ Force *ff) {
    BendingKantor *b = CONTAINER_OF(q, BendingKantor, bending);
    UC(kantor_apply(b->kantor, params, quants, ff));
}

static void method_kantor_fin(Bending *q) {
    BendingKantor *b = CONTAINER_OF(q, BendingKantor, bending);
    UC(kantor_fin(b->kantor));
}    


static Bending_vtable BendingKantor_vtable = { method_kantor_apply, method_kantor_fin};

void bending_kantor_ini(const MeshRead *cell, /**/ Bending **pq) {
    BendingKantor *q;
    EMALLOC(1, &q);    
    UC(kantor_ini(cell, &q->kantor));
    q->bending.vtable = &BendingKantor_vtable;
    *pq = &q->bending;
}
