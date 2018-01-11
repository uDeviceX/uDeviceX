static void fin_rnd(rbc::rnd::D *rnd) {
    rbc::rnd::fin(rnd);
}

void rbcforce_fin(RbcForce *t) {
    if (RBC_RND) fin_rnd(t->rnd);
}
