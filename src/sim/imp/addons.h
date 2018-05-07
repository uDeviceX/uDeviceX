/* velocity controller */

_S_ bool valid_step(long id, const int freq) {
    return (freq != 0) && (id % freq == 0);
}

_I_ void sample_vcont(const Coords *coords, long id, const Flu *f, /**/ Vcon *c) {
    if (valid_step(id, c->sample_freq)) {
        const FluQuants *q = &f->q;
        vcont_sample(coords, q->n, q->pp, q->cells.starts, q->cells.counts, /**/ c->vcont);
    }
}

_I_ void adjust_bforce(long id, /**/ Vcon *c, BForce *bforce) {
    if (valid_step(id, c->adjust_freq)) {
        float3 f;
        f = vcont_adjustF(/**/ c->vcont);
        bforce_adjust(f, /**/ bforce);
    }
}

_I_ void log_vcont(long id, const Vcon *c) {
    if (valid_step(id, c->log_freq))
        vcont_log(c->vcont);
}

/* set colors of particles according to the RBCs */

_I_ void recolor_flux(const Coords *c, const Recolorer *opt, Flu *f) {
    int3 L = subdomain(c);
    if (opt->flux_active)
        UC(color_linear_flux(c, L, opt->flux_dir, RED_COLOR, f->q.n, f->q.pp, /**/ f->q.cc));
}


_I_ void apply_inflow(float kBT, int numdensity, float dt, Inflow *i, Flu *f) {
    if (f->q.colors)
        UC(inflow_create_pp_cc(kBT, numdensity, dt, RED_COLOR, i, &f->q.n, f->q.pp, f->q.cc));
    else
        UC(inflow_create_pp(kBT, numdensity, dt, i, &f->q.n, f->q.pp));
}

_I_ void mark_outflow(const Flu *f, Outflow *o) {
    UC(outflow_filter_particles(f->q.n, f->q.pp, /**/ o));
    UC(outflow_download_ndead(o));
}

_I_ void mark_outflowden(OptParams params, const Flu *f, const DContMap *m, /**/ DCont *d) {
    const int *ss, *cc;
    int n;
    n = f->q.n;
    ss = f->q.cells.starts;
    cc = f->q.cells.counts;

    UC(den_reset(n, /**/ d));
    UC(den_filter_particles(params.numdensity, m, ss, cc, /**/ d));
    UC(den_download_ndead(/**/ d));
}

_S_ void sample(Sim *s) {
    Flu *flu = &s->flu;
    Sampler *sam = &s->dump.field_sampler;

    UC(grid_sampler_data_reset(sam->d));
    UC(grid_sampler_data_push(flu->q.n, flu->q.pp, flu->q.cc, flu->ss, sam->d));
    UC(grid_sampler_add(sam->d, sam->s));
}

_I_ void field_sample(Sim *s) {
    if (s->opt.dump.field && is_sampling_time(s))
        sample(s);
}

_I_ void colors_from_rbc(Sim *s) {
    PFarray flu;
    utils_get_pf_flu(s, &flu);
    UC(objects_recolor_flu(s->obj, &flu));
}

_I_ void recolor_from_rbc(long it, Sim *s) {
    bool cond;
    const Opt *opt = &s->opt;
    cond = opt->flucolors && opt->recolor_freq && it % opt->recolor_freq == 0;
    if (cond) {
        msg_print("recolor");
        UC(colors_from_rbc(s));
    }
}
