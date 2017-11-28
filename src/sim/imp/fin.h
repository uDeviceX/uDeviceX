static void fin_flu_exch(/**/ Fluexch *e) {
    using namespace exch::flu;
    fin(/**/ &e->p);
    fin(/**/ &e->c);
    fin(/**/ &e->u);
}

static void fin_obj_exch(/**/ Objexch *e) {
    using namespace exch::obj;
    fin(/**/ &e->p);
    fin(/**/ &e->c);
    fin(/**/ &e->u);
    fin(/**/ &e->pf);
    fin(/**/ &e->uf);
}

static void fin_mesh_exch(/**/ Mexch *e) {
    using namespace exch::mesh;
    fin(/**/ &e->p);
    fin(/**/ &e->c);
    fin(/**/ &e->u);
}

static void fin_bb_exch(/**/ BBexch *e) {
    fin_mesh_exch(/**/ e);
    
    using namespace exch::mesh;
    fin(/**/ &e->pm);
    fin(/**/ &e->cm);
    fin(/**/ &e->um);
}

static void fin_flu_distr(/**/ FluDistr *d) {
    using namespace distr::flu;
    fin(/**/ &d->p);
    fin(/**/ &d->c);
    fin(/**/ &d->u);
}

static void fin_rbc_distr(/**/ RbcDistr *d) {
    using namespace distr::rbc;
    fin(/**/ &d->p);
    fin(/**/ &d->c);
    fin(/**/ &d->u);
}

static void fin_rig_distr(/**/ RigDistr *d) {
    using namespace distr::rig;
    fin(/**/ &d->p);
    fin(/**/ &d->c);
    fin(/**/ &d->u);
}

static void fin_colorer(Colorer *c) {
    fin_mesh_exch(/**/ &c->e);
    Dfree(c->pp);
    Dfree(c->minext);
    Dfree(c->maxext);
}

static void fin_flu(Flu *f) {
    flu::fin(&f->q);
    fin(/**/ f->bulkdata);
    fin(/**/ f->halodata);
 
    fin_flu_distr(/**/ &f->d);
    fin_flu_exch(/**/ &f->e);

    UC(Dfree(f->ff));
    UC(efree(f->ff_hst));
}

void fin() {
    cnt::fin(&rs::c);
    bop::fin(&dumpt);
    if (rbcs || solids)
        fin_obj_exch(&rs::e);

    if (VCON) fin(/**/ &o::vcont);
    if (fsiforces)  fsi::fin();
    
    if (walls) {
        sdf::free_quants(&w::qsdf);
        wall::free_quants(&w::q);
        wall::free_ticket(&w::t);
    }

    flu::fin(&o::q);
    fin(/**/ o::bulkdata);
    fin(/**/ o::halodata);
 
    fin_flu_distr(/**/ &o::d);
    fin_flu_exch(/**/ &o::e);

    if (multi_solvent && rbcs)
        fin_colorer(/**/ &colorer);

    if (solids) {
        rig::fin(&s::q);
        scan::free_work(/**/ &s::ws);
        Dfree(s::ff);
        free(s::ff_hst);

        meshbb::fin(/**/ &bb::bbd);
        Dfree(bb::mm);

        fin_rig_distr(/**/ &s::d);
        
        if (sbounce_back)
            fin_bb_exch(/**/ &s::e);
    }

    if (rbcs) {
        rbc::main::fin(&r::q);
        rbc::force::fin_ticket(&r::tt);

        fin_rbc_distr(/**/ &r::d);
        
        Dfree(r::ff);

        if (rbc_com_dumps) rbc::com::fin(/**/ &r::com);
        if (RBC_STRETCH)   rbc::stretch::fin(/**/ r::stretch);
            
    }
    datatype::fin();
}
