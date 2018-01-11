void fin(/**/ ComProps *com) {
    CC(d::FreeHost(com->hrr));
    CC(d::Free(com->drr));
}
