namespace sim {
/* see bund.cu for more sim:: functions */

void create_walls() {
    int nold = o::n;

    dSync();
    sdf::ini();
    wall::gen_quants(&o::n, o::pp, &w::q);
    wall::gen_ticket(w::q, &w::t);
    MSG("solvent particles survived: %d/%d", o::n, nold);
    if (o::n) k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);
    o::cells->build(o::pp, o::n);
    flu::create_ticketZ(o::pp, o::n, &o::tz);

    CC( cudaPeekAtLastError() );
}

void update_solid() {
    if (s::q.n) update_solid0();
}

void update_solvent() {
    float mass = 1;
    if (o::n) k_sim::update<<<k_cnf(o::n)>>> (mass, o::pp, o::ff, o::n);
}

void update_rbc() {
    float mass = rbc_mass;
    if (r::q.n) k_sim::update<<<k_cnf(r::q.n)>>> (mass, r::q.pp, r::ff, r::q.n);
}

void bounce() {
    if (o::n) k_sdf::bounce<<<k_cnf(o::n)>>>((float2*)o::pp, o::n);
    //if (rbcs && r::n) k_sdf::bounce<<<k_cnf(r::n)>>>((float2*)r::pp, r::n);
}

void step(float driving_force0, bool wall0, int it) {
    assert(o::n <= MAX_PART_NUM);
    assert(r::q.n <= MAX_PART_NUM);
    flu::distr(&o::pp, &o::n, o::cells, &o::td, &o::tz, &o::w);
    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    forces(wall0);
    dump_diag0(it);
    if (wall0) dump_diag_after(it);
    body_force(driving_force0);
    update_solvent();
    if (solids0) update_solid();
    if (rbcs)    update_rbc();
    if (wall0) bounce();
    if (sbounce_back && solids0) bounce_solid(it);
}

void run_eq(long te) { /* equilibrate */
  long it;
  float driving_force0 = 0;
  bool wall0 = false;
  for (it = 0; it < te; ++it) step(driving_force0, wall0, it);
}

void create_solids() {
  cD2H(o::pp_hst, o::pp, o::n);
  rig::create(/*io*/ o::pp_hst, &o::n, /**/ &s::q);
  MC(l::m::Barrier(m::cart));
  rig::gen_pp_hst(s::q);
  rig::gen_ipp_hst(s::q);
  rig::cpy_H2D(s::q);
  
  cH2D(o::pp, o::pp_hst, o::n);
  MC(l::m::Barrier(m::cart));
  MSG("created %d solids.", s::q.ns);
}

void freeze() {
  if (walls) create_walls();
  MC(MPI_Barrier(m::cart));
  if (solids) create_solids();
  if (walls && rbcs  ) remove_rbcs();
  if (walls && solids) remove_solids();
  if (solids)          rig::set_ids(s::q);
  if (solids && s::q.n) k_sim::clear_velocity<<<k_cnf(s::q.n)>>>(s::q.pp, s::q.n);
  if (rbcs   && r::q.n) k_sim::clear_velocity<<<k_cnf(r::q.n)>>>(r::q.pp, r::q.n);
}

void run(long ts, long te) {
  /* ts, te: time start and end */
  long it;
  float driving_force0 = pushflow ? driving_force : 0;
  for (it = ts; it < te; ++it) step(driving_force0, walls, it);
}

void gen() { /* generate */
  run_eq(wall_creation);
  freeze();
}

void sim() {
  long nsteps = (int)(tend / dt);
  MSG0("will take %ld steps", nsteps);
  if (walls || solids) {
    solids0 = false;  /* global */
    gen();
    dSync();
    solids0 = solids;
    run(wall_creation, nsteps);
  } else {
    solids0 = solids;
    run(            0, nsteps);
  }
}

}
