void forces_objects(Sim *sim) {
    PFarrays *obj;
    PFarray flu;

    PaArray pa;
    FoArray fa;

    Rbc *r = &sim->rbc;
    Rig *s = &sim->rig;
    
    UC(pfarrays_ini(&obj));

    utils_get_pf_flu(sim, &flu);
    
    if (active_rig(sim)) {
        UC(parray_push_pp(s->q.pp, &pa));
        UC(farray_push_ff(s->ff, &fa));
        UC(pfarrays_push(obj, s->q.n, pa, fa));
    }
    
    if (active_rbc(sim)) {
        UC(parray_push_pp(r->q.pp, &pa));
        UC(farray_push_ff(r->ff, &fa));
        UC(pfarrays_push(obj, r->q.n, pa, fa));        
    }

    UC(obj_inter_forces(sim->objinter, &flu, sim->flu.q.cells.starts, obj));

    UC(pfarrays_fin(obj));
}

