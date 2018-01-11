static void fin_rnd(rbc::rnd::RbcRnd *rnd) {
    rbc::rnd::fin(rnd);
}

void rbcforce_fin(RbcForce *t) {
    if (RBC_RND) fin_rnd(t->rnd);
}
