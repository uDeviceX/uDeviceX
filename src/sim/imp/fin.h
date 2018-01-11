static void fin_flu_exch(/**/ FluExch *e) {
    using namespace exch::flu;
    fin(/**/ &e->p);
    fin(/**/ &e->c);
    fin(/**/ &e->u);
}

static void fin_obj_exch(/**/ ObjExch *e) {
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
    UC(dflu_pack_fin(/**/ &d->p));
    UC(dflu_comm_fin(/**/ &d->c));
    UC(dflu_unpack_fin(/**/ &d->u));
}

static void fin_rbc_distr(/**/ RbcDistr *d) {
    using namespace distr::rbc;
    UC(drbc_pack_fin(/**/ &d->p));
    UC(drbc_comm_fin(/**/ &d->c));
    UC(drbc_unpack_fin(/**/ &d->u));
}

static void fin_rig_distr(/**/ RigDistr *d) {
    UC(drig_pack_fin(/**/ &d->p));
    UC(drig_comm_fin(/**/ &d->c));
    UC(drig_unpack_fin(/**/ &d->u));
}

static void fin_colorer(Colorer *c) {
    fin_mesh_exch(/**/ &c->e);
    Dfree(c->pp);
    Dfree(c->minext);
    Dfree(c->maxext);
}

static void fin_outflow(Outflow *o) {
    fin(/**/ o);
}

static void fin_denoutflow(DCont *d, DContMap *m) {
    UC(den_fin(d));
    UC(den_fin_map(m));
}

static void fin_inflow(Inflow *i) {
    UC(inflow_fin(/**/ i));
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

static void fin_rbc(Rbc *r) {
    rbc::main::fin(&r->q);
    rbc::force::fin_ticket(&r->tt);

    fin_rbc_distr(/**/ &r->d);
        
    Dfree(r->ff);

    if (rbc_com_dumps) rbc::com::fin(/**/ &r->com);
    if (RBC_STRETCH)   rbc::stretch::fin(/**/ r->stretch);
}

static void fin_rig(Rig *s) {
    rig::fin(&s->q);
    scan::free_work(/**/ &s->ws);
    Dfree(s->ff);
    UC(efree(s->ff_hst));

    UC(fin_rig_distr(/**/ &s->d));
}

static void fin_bounce_back(BounceBack *bb) {
    meshbb::fin(/**/ &bb->d);
    Dfree(bb->mm);
    UC(fin_bb_exch(/**/ &bb->e));
}

static void fin_wall(Wall *w) {
    fin(w->sdf);
    wall::free_quants(&w->q);
    wall::free_ticket(&w->t);
}
    
static void fin_objinter(ObjInter *o) {
    UC(fin_obj_exch(&o->e));
    if (contactforces) cnt_fin(o->cnt);
    if (fsiforces)     fsi_fin(o->fsi);
}

static void fin_vcon(Vcon *c) {
    UC(vcont_fin(c->vcont));
}

void sim_fin(Sim *s) {

    bop::fin(&s->dumpt);
    if (rbcs || solids)
        fin_objinter(&s->objinter);

    if (s->opt.vcon)       UC(fin_vcon(/**/ &s->vcon));
    if (s->opt.outflow)    UC(fin_outflow(/**/ s->outflow));
    if (s->opt.inflow)     UC(fin_inflow (/**/ s->inflow ));
    if (s->opt.denoutflow) UC(fin_denoutflow(/**/ s->denoutflow, s->mapoutflow));
    
    if (walls) fin_wall(&s->wall);

    fin_flu(&s->flu);

    if (multi_solvent && rbcs)
        fin_colorer(/**/ &s->colorer);

    if (solids) {
        fin_rig(/**/ &s->rig);
        
        if (sbounce_back)
            fin_bounce_back(&s->bb);
    }

    UC(efree(s->pp_dump));
    
    if (rbcs) fin_rbc(/**/ &s->rbc);

    UC(coords_fin(/**/ &s->coords));

    UC(conf_fin(s->cfg));
    datatype::fin();

    MC(m::Comm_free(&s->cart));
    UC(efree(s));
}
