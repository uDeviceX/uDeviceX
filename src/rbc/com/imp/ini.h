void rbc_com_ini(int maxcells, /**/ RbcComProps *com) {
    size_t sz = maxcells * sizeof(float3);
    CC(d::alloc_pinned((void**) &com->hrr, sz));
    CC(d::alloc_pinned((void**) &com->hvv, sz));
    CC(d::Malloc((void**) &com->drr, sz));
    CC(d::Malloc((void**) &com->dvv, sz));
}
