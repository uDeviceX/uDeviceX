_I_ void clear_vel(Sim *s) {
    Flu *flu = &s->flu;
    UC(scheme_move_clear_vel(flu->q.n, flu->q.pp));
    UC(objects_clear_vel(s->obj));
}

_I_ void update_solvent(float dt, /**/ Flu *f) {
    UC(scheme_move_apply(dt, f->mass, f->q.n, f->ff, f->q.pp));
}

_I_ void store_solvent(Flu *f) {
    FluQuants *q = &f->q;
    long n = q->n;
    if (n)
        aD2D(q->pp0, q->pp, n);
}

_I_ void restrain(long it, Sim *s) {
    SchemeQQ qq;
    PFarrays *pf;
    PFarray p;
    UC(pfarrays_ini(&pf));
    UC(objects_get_particles_mbr(s->obj, pf));

    if (pfarrays_size(pf) > 0) {
        UC(pfarrays_get(0, pf, &p));
        qq.r  = (Particle*) p.p.pp;
        qq.rn = p.n;
    } else {
        qq.r  = NULL;
        qq.rn = 0;
    }
    qq.o = s->flu.q.pp;    
    qq.on = s->flu.q.n;
    
    UC(scheme_restrain_apply(s->cart, s->flu.q.cc, it, /**/ s->restrain, qq));

    UC(pfarrays_fin(pf));
}

_I_ void bounce_wall(float dt, Sim *s) {
    Flu *f = &s->flu;
    PaArray pa;
    FoArray fo;
    PFarrays *pf;
    UC(pfarrays_ini(&pf));
    UC(parray_push_pp(f->q.pp, &pa));
    UC(pfarrays_push(pf, f->q.n, pa, fo));

    UC(objects_get_particles_mbr(s->obj, pf));
    
    UC(wall_bounce(s->wall, s->coords, dt, pf));
    UC(pfarrays_fin(pf));
}

_I_ void bounce_objects(float dt, Sim *s) {
    PFarray pfflu;
    const Flu *flu = &s->flu;
    UC(utils_get_pf_flu(s, &pfflu));
    UC(objects_bounce(dt, flu->mass, flu->q.cells, &pfflu, s->obj));
}

_S_ void flu_update_dpd_prms(float dt, float kBT, Sim *s) {
    UC(pair_compute_dpd_sigma(kBT, dt, /**/ s->flu.params));
}

_I_ void update_params(float dt, Sim *s) {
    float kBT;
    const Opt *opt = &s->opt;
    kBT = opt->params.kBT;
    UC(flu_update_dpd_prms(dt, kBT, s));
    UC(obj_inter_update_dpd_prms(dt, kBT, s->objinter));
    UC(objects_update_dpd_prms(dt, kBT, s->obj));
}
