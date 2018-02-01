void rbc_com_fin(/**/ RbcCom *q) {
    CC(d::FreeHost(q->hrr));
    CC(d::FreeHost(q->hvv));
    CC(d::Free(q->drr));
    CC(d::Free(q->dvv));
    EFREE(q);
}
