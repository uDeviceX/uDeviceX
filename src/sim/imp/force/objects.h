void forces_objects(Sim *sim) {
    PFarrays *obj;
    PFarrays *flu;

    PaArray pa;
    FoArray fa;

    Flu *f = &sim->flu;
    Rbc *r = &sim->rbc;
    Rig *s = &sim->rig;

    if (!sim->objinter) return;
    
    UC(pfarrays_ini(&flu));
    UC(pfarrays_ini(&obj));
    
    UC(parray_push_pp(f->q.pp, &pa));
    if (sim->opt.flucolors)
        parray_push_cc(f->q.cc, &pa);

    UC(farray_push_ff(f->ff, &fa));

    UC(pfarray_push(flu, f->q.n, pa, fa));
    
    if (active_rig(sim)) {
        UC(parray_push_pp(s->q.pp, &pa));
        UC(farray_push_ff(s->ff, &fa));
        UC(pfarray_push(obj, s->q.n, pa, fa));
    }
    
    if (active_rbc(sim)) {
        UC(parray_push_pp(r->q.pp, &pa));
        UC(farray_push_ff(r->ff, &fa));
        UC(pfarray_push(obj, r->q.n, pa, fa));        
    }

    UC(obj_inter_forces(sim->objinter, flu, f->q.cells.starts, obj));

    UC(pfarrays_fin(flu));
    UC(pfarrays_fin(obj));
}

