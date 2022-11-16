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
    EFREE(q);
}    
static Bending_vtable BendingKantor_vtable = { method_kantor_apply, method_kantor_fin};
void bending_kantor_ini(const MeshRead *cell, /**/ Bending **pq) {
    BendingKantor *q;
    EMALLOC(1, &q);    
    UC(kantor_ini(cell, &q->kantor));
    q->bending.vtable = &BendingKantor_vtable;
    *pq = &q->bending;
}


struct BendingJuelicher {
    Bending bending;
    Juelicher *juelicher;
};
static void method_juelicher_apply(Bending *q, const RbcParams *params, const RbcQuants *quants, /**/ Force *ff) {
    BendingJuelicher *b = CONTAINER_OF(q, BendingJuelicher, bending);
    UC(juelicher_apply(b->juelicher, params, quants, ff));
}
static void method_juelicher_fin(Bending *q) {
    BendingJuelicher *b = CONTAINER_OF(q, BendingJuelicher, bending);
    UC(juelicher_fin(b->juelicher));
    EFREE(q);
}    
static Bending_vtable BendingJuelicher_vtable = { method_juelicher_apply, method_juelicher_fin};
void bending_juelicher_ini(const MeshRead *cell, /**/ Bending **pq) {
    BendingJuelicher *q;
    EMALLOC(1, &q);    
    UC(juelicher_ini(cell, &q->juelicher));
    q->bending.vtable = &BendingJuelicher_vtable;
    *pq = &q->bending;
}

struct BendingNone { Bending bending; };
static void method_none_apply(Bending*, const RbcParams*, const RbcQuants*, /**/ Force*) { }
static void method_none_fin(Bending*) { }
static Bending_vtable BendingNone_vtable = { method_none_apply, method_none_fin};
void bending_none_ini(const MeshRead*, /**/ Bending **pq) {
    BendingNone *q;
    EMALLOC(1, &q);    
    q->bending.vtable = &BendingNone_vtable;
    *pq = &q->bending;
}
