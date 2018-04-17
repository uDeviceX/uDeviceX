/* velocity controller */

static bool valid_step(long id, const int freq) {
    return (freq != 0) && (id % freq == 0);
}

void sample_vcont(const Coords *coords, long id, const Flu *f, /**/ Vcon *c) {
    if (valid_step(id, c->sample_freq)) {
        const FluQuants *q = &f->q;
        vcont_sample(coords, q->n, q->pp, q->cells.starts, q->cells.counts, /**/ c->vcont);
    }
}

void adjust_bforce(long id, /**/ Vcon *c, BForce *bforce) {
    if (valid_step(id, c->adjust_freq)) {
        float3 f;
        f = vcont_adjustF(/**/ c->vcont);
        bforce_adjust(f, /**/ bforce);
    }
}

void log_vcont(long id, const Vcon *c) {
    if (valid_step(id, c->log_freq))
        vcont_log(c->vcont);
}

/* set colors of particles according to the RBCs */

void colors_from_rbc(Sim *s) {
    int nm, nv, n, nmhalo;
    const Rbc *r = &s->rbc;
    Colorer *c = &s->colorer;
    Flu *f = &s->flu;
        
    n  = f->q.n;
    nm = r->q.nc;
    nv = r->q.nv;

    UC(emesh_build_map(nm, nv, r->q.pp, /**/ c->e.p));
    UC(emesh_pack(nv, r->q.pp, /**/ c->e.p));
    UC(emesh_download(c->e.p));

    UC(emesh_post_send(c->e.p, c->e.c));
    UC(emesh_post_recv(c->e.c, c->e.u));

    if (nm*nv && n > 0)
        aD2D(c->pp, r->q.pp, nm*nv);

    UC(emesh_wait_send(c->e.c));
    UC(emesh_wait_recv(c->e.c, c->e.u));

    UC(emesh_unpack(nv, c->e.u, /**/ &nmhalo, c->pp + nm * nv));
    nm += nmhalo;

    /* compute extents */
    if (n > 0) {
        UC(minmax(c->pp, nv, nm, /**/ c->minext, c->maxext));
        UC(collision_get_colors(f->q.pp, f->q.n, c->pp, r->tri, nv, nm,
                                c->minext, c->maxext, /**/ f->q.cc));
    }
}

void recolor_flux(const Coords *c, const Recolorer *opt, Flu *f) {
    int3 L = make_int3(xs(c), ys(c), zs(c));
    if (opt->flux_active)
        UC(color_linear_flux(c, L, opt->flux_dir, RED_COLOR, f->q.n, f->q.pp, /**/ f->q.cc));
}


static void apply_inflow(float kBT, int numdensity, float dt, Inflow *i, Flu *f) {
    if (f->q.colors)
        UC(inflow_create_pp_cc(kBT, numdensity, dt, RED_COLOR, i, &f->q.n, f->q.pp, f->q.cc));
    else
        UC(inflow_create_pp(kBT, numdensity, dt, i, &f->q.n, f->q.pp));
}

static void mark_outflow(const Flu *f, Outflow *o) {
    UC(outflow_filter_particles(f->q.n, f->q.pp, /**/ o));
    UC(outflow_download_ndead(o));
}

static void mark_outflowden(Params params, const Flu *f, const DContMap *m, /**/ DCont *d) {
    const int *ss, *cc;
    int n;
    n = f->q.n;
    ss = f->q.cells.starts;
    cc = f->q.cells.counts;

    UC(den_reset(n, /**/ d));
    UC(den_filter_particles(params.numdensity, m, ss, cc, /**/ d));
    UC(den_download_ndead(/**/ d));
}

static void sample(Sim *s) {
    Flu *flu = &s->flu;
    Sampler *sam = &s->dump.field_sampler;

    UC(grid_sampler_data_reset(sam->d));
    UC(grid_sampler_data_push(flu->q.n, flu->q.pp, flu->q.cc, flu->ss, sam->d));
    UC(grid_sampler_add(sam->d, sam->s));
}

static void field_sample(Sim *s) {
    if (s->opt.dump_field && is_sampling_time(s))
        sample(s);
}
