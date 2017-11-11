void ini(int maxcells, /**/ ComHelper *com) {
    size_t sz = maxcells * sizeof(float3);
    CC(d::alloc_pinned((void**) &com->hrr, sz));
    CC(d::Malloc((void**) &com->drr, sz));
}
