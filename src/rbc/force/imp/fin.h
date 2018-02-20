static void fin_rnd(RbcRnd *rnd) {
    rbc_rnd_fin(rnd);
}

static void fin_stress(RbcForce *f) {
    if (is_stress_free(f)) {
        StressFree_v v = f->sinfo.sfree;
        Dfree(v.ll);
        Dfree(v.aa);
    }
}

void rbc_force_fin(RbcForce *q) {
    if (RBC_RND) fin_rnd(q->rnd);
    UC(fin_stress(q));
    EFREE(q);
}
