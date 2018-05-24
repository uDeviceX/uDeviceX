_S_ void fin_flu_exch(/**/ FluExch *e) {
    UC(eflu_pack_fin(/**/ e->p));
    UC(eflu_comm_fin(/**/ e->c));
    UC(eflu_unpack_fin(/**/ e->u));
}

_S_ void fin_flu_distr(/**/ FluDistr *d) {
    UC(dflu_pack_fin(/**/ d->p));
    UC(dflu_comm_fin(/**/ d->c));
    UC(dflu_unpack_fin(/**/ d->u));
}

_S_ void fin_outflow(Outflow *o) {
    UC(outflow_fin(/**/ o));
}

_S_ void fin_denoutflow(DCont *d, DContMap *m) {
    UC(den_fin(d));
    UC(den_map_fin(m));
}

_S_ void fin_inflow(Inflow *i) {
    UC(inflow_fin(/**/ i));
}


_S_ void fin_flu(const Opt *opt, Flu *f) {
    UC(flu_fin(&f->q));
    UC(fluforces_bulk_fin(/**/ f->bulk));
    UC(fluforces_halo_fin(/**/ f->halo));
 
    UC(fin_flu_distr(/**/ &f->d));
    UC(fin_flu_exch(/**/ &f->e));

    UC(Dfree(f->ff));
    EFREE(f->ff_hst);

    if (opt->flu.ss) {
        UC(Dfree(f->ss));
        EFREE(f->ss_hst);        
    }
}
    
_S_ void fin_vcon(Vcon *c) {
    UC(vcont_fin(c->vcont));
}

_S_ void fin_pair_params(Sim *s) {
    UC(pair_fin(s->flu.params));
}

_S_ void fin_sampler(Sampler *s) {
    UC(grid_sampler_data_fin(s->d));
    UC(grid_sampler_fin(s->s));
}

_S_ void fin_dump(const Opt *opt, Dump *d) {
    UC(io_bop_fin(d->bop));
    UC(diag_part_fin(d->diagpart));
    if (opt->dump.field) UC(fin_sampler(&d->field_sampler));
    EFREE(d->pp);
}

_S_ void fin_time(Time *t) {
    UC(time_step_fin(t->step));
    UC(time_step_accel_fin(t->accel));    
}

_S_ void fin_optional_features(const Opt *opt, Sim *s) {
    if (opt->tracers)    UC(tracers_fin());
    if (opt->vcon)       UC(fin_vcon(/**/ &s->vcon));
    if (opt->outflow)    UC(fin_outflow(/**/ s->outflow));
    if (opt->inflow)     UC(fin_inflow (/**/ s->inflow ));
    if (opt->denoutflow) UC(fin_denoutflow(/**/ s->denoutflow, s->mapoutflow));
}

void sim_fin(Sim *s) {
    const Opt *opt = &s->opt;
    
    UC(fin_optional_features(opt, s));
    
    UC(fin_flu(opt, &s->flu));
    if (opt->wall.active) UC(wall_fin(s->wall));

    UC(obj_inter_fin(s->objinter));
    UC(objects_fin(s->obj));
    
    UC(bforce_fin(s->bforce));
    
    UC(fin_dump(opt, &s->dump));
    UC(scheme_restrain_fin(s->restrain));
    UC(coords_fin(/**/ s->coords));
    UC(fin_time(&s->time));
    UC(fin_pair_params(s));
    UC(inter_color_fin(s->gen_color));
    UC(dbg_fin(s->dbg));

    MC(m::Comm_free(&s->cart));
    EFREE(s);
}
