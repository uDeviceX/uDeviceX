void rbc_com_fin(/**/ RbcCom *q) {
    CC(d::FreeHost(q->hrr));
    CC(d::FreeHost(q->hvv));
    Dfree(q->drr);
    Dfree(q->dvv);
    EFREE(q);
}
