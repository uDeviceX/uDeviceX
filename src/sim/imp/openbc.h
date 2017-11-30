void mark_outflow(const Flu *f, Outflow *o) {
    if (!OUTFLOW) return;
    float3 normal, r;
    normal = make_float3(1, 0, 0);
    r = make_float3(XS/2-1, 0, 0);
    filter_particles_plane(normal, r, f->q.n, f->q.pp, /**/ o);
}
