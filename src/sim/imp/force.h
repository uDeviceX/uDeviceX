_S_ void body_force(const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;

    if (opt->push_flu)
        UC(bforce_apply(s->coords, flu->mass, bforce, flu->q.n, flu->q.pp, /**/ flu->ff));
    UC(objects_body_forces(bforce, s->obj));
}

_S_ void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

_S_ void clear_stresses(int n, float* ss) {
    if (n) DzeroA(ss, 6*n);
}

_S_ void forces_wall(bool fluss, Sim *s) {
    PaArray po;
    FoArray fo;
    PFarrays *pf;
    Flu *flu = &s->flu;
    Wall *w =  s->wall;
    const PairParams *par = flu->params;
    const Opt *opt = &s->opt;

    if (!w) return;
    
    parray_push_pp(flu->q.pp, &po);
    if (opt->flucolors)
        parray_push_cc(flu->q.cc, &po);

    farray_push_ff(flu->ff, &fo);
    if (fluss)
        farray_push_ss(flu->ss, &fo);

    UC(pfarrays_ini(&pf));
    UC(pfarrays_push(pf, flu->q.n, po, fo));
    UC(objects_get_particles_all(s->obj, pf));
    UC(wall_interact(s->coords, par, w, pf));
    UC(pfarrays_fin(pf));
    
    // if (active_rig(s) && rig->q.n) wall_repulse(rig->q.n, rig->q.pp, w->sdf, /**/ rig->ff);
    // if (active_rbc(s) && rbc->q.n) wall_repulse(rbc->q.n, rbc->q.pp, w->sdf, /**/ rbc->ff);
}

_S_ void forces_dpd(bool stress, Flu *f) {
    const int *count = f->q.cells.counts;
    const int *start = f->q.cells.starts;
    PaArray parray;
    FoArray farray;
    flu::LFrag26 lfrags;
    flu::RFrag26 rfrags;

    FluExch *e = &f->e;
    
    parray_push_pp(f->q.pp, /**/ &parray);
    if (f->q.colors)
        parray_push_cc(f->q.cc, /**/ &parray);

    farray_push_ff(f->ff, /**/ &farray);
    if (stress)
        farray_push_ss(f->ss, /**/ &farray);
    
    UC(eflu_compute_map(start, count, /**/ e->p));
    UC(eflu_download_cell_starts(/**/ e->p));
    UC(eflu_pack(&parray, /**/ e->p));
    UC(eflu_download_data(/**/ e->p));

    UC(eflu_post_recv(e->c, e->u));
    UC(eflu_post_send(e->p, e->c));
    
    UC(fluforces_bulk_prepare(f->q.n, &parray, /**/ f->bulk));
    UC(fluforces_bulk_apply(f->params, f->q.n, f->bulk, start, count, /**/ &farray));

    UC(eflu_wait_recv(e->c, e->u));
    UC(eflu_wait_send(e->c));

    UC(eflu_unpack(e->u));

    UC(eflu_get_local_frags(e->p, /**/ &lfrags));
    UC(eflu_get_remote_frags(e->u, /**/ &rfrags));

    UC(fluforces_halo_prepare(lfrags, rfrags, /**/ f->halo));
    UC(fluforces_halo_apply(f->params, f->halo, /**/ &farray));
}

_S_ void forces_objects(Sim *sim) {
    PFarrays *obj;
    PFarray flu;
    
    UC(pfarrays_ini(&obj));
    UC(objects_get_particles_all(sim->obj, obj));
    UC(utils_get_pf_flu(sim, &flu));
    
    UC(obj_inter_forces(sim->objinter, &flu, sim->flu.q.cells.starts, obj));

    UC(pfarrays_fin(obj));
}

_I_ void forces(float dt, TimeLine *time, const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    bool fluss, tfluss;
    Opt *opt = &s->opt;

    tfluss = time_line_cross(time, opt->dump.freq_parts) ||
        is_sampling_time(s);
    fluss = opt->fluss && tfluss;

    UC(clear_forces(flu->q.n, flu->ff));
    if (fluss) UC(clear_stresses(flu->q.n, flu->ss));
    UC(objects_clear_forces(s->obj));    
    
    UC(forces_dpd(fluss, flu));
    if (active_walls(s)) UC(forces_wall(fluss, s));
    UC(objects_internal_forces(dt, s->obj));

    UC(forces_objects(s));

    if (!s->equilibrating) UC(body_force(bforce, s));
    
    dSync();
}
