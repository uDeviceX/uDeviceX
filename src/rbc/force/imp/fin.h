static void fin_rnd(RbcRnd *rnd) {
    rbc_rnd_fin(rnd);
}

void rbc_force_fin(RbcForce *q) {
    if (RBC_RND) fin_rnd(q->rnd);
    EFREE(q);
}
