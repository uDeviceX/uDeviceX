void apply_inflow(Inflow *i, Flu *f) {
    create_pp(i, &f->q.n, f->q.pp);
}

static void mark_outflow_plane(const Flu *f, Outflow *o) {
    float3 normal, r;
    normal = make_float3(1, 0, 0);
    r = make_float3(XS/2-1, 0, 0);
    filter_particles_plane(normal, r, f->q.n, f->q.pp, /**/ o);
    download_ndead(o);
}

static void mark_outflow_circle(const Flu *f, Outflow *o) {
    float R = OUTFLOW_CIRCLE_R;
    filter_particles_circle(R, f->q.n, f->q.pp, /**/ o);
    download_ndead(o);
}

void mark_outflow(const Flu *f, Outflow *o) {
    if (OUTFLOW_CIRCLE)
        mark_outflow_circle(f, o);
    else
        mark_outflow_plane(f, o);
}
