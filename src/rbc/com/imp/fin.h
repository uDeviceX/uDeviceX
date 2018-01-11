void rbc_com_fin(/**/ RbcComProps *com) {
    CC(d::FreeHost(com->hrr));
    CC(d::FreeHost(com->hvv));
    CC(d::Free(com->drr));
    CC(d::Free(com->dvv));
}
