static void get_pf_objs(Sim *sim, PFarrays *o) {
    PaArray pa;
    FoArray fa;

    Rbc *r = &sim->rbc;
    Rig *s = &sim->rig;
    
    if (active_rig(sim)) {
        UC(parray_push_pp(s->q.pp, &pa));
        UC(farray_push_ff(s->ff, &fa));
        UC(pfarrays_push(o, s->q.n, pa, fa));
    }
    
    if (active_rbc(sim)) {
        UC(parray_push_pp(r->q.pp, &pa));
        UC(farray_push_ff(r->ff, &fa));
        UC(pfarrays_push(o, r->q.n, pa, fa));        
    }    
}

void forces_objects(Sim *sim) {
    PFarrays *obj;
    PFarray flu;
    
    UC(pfarrays_ini(&obj));
    get_pf_objs(sim, obj);
    utils_get_pf_flu(sim, &flu);
    
    UC(obj_inter_forces(sim->objinter, &flu, sim->flu.q.cells.starts, obj));

    UC(pfarrays_fin(obj));
}

