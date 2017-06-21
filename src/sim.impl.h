namespace sim {
/* see bund.cu for more sim:: functions */

void create_walls() {
  int nold = o::n;
  wall::gen_quants(&o::n, o::pp, &w::q); o::cells->build(o::pp, o::n);
  MSG("solvent particles survived: %d/%d", o::n, nold);
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
  MC(MPI_Barrier(m::cart));
  if (solids)           create_solids();
  if (walls && rbcs  )  remove_rbcs();
  if (walls && solids)  remove_solids();
  if (solids)           rig::set_ids(s::q);
}

void clear_velocity() {
  if (o::n)             k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);  
  if (solids && s::q.n) k_sim::clear_velocity<<<k_cnf(s::q.n)>>>(s::q.pp, s::q.n);
  if (rbcs   && r::q.n) k_sim::clear_velocity<<<k_cnf(r::q.n)>>>(r::q.pp, r::q.n);
}

void gen() { /* generate */
  run_eq(wall_creation);
  if (walls) {
    dSync();
    sdf::ini();
    create_walls();
  }
  freeze();
  clear_velocity();
}

void sim() {
  o::n = ic::gen(o::pp, /*w*/ o::pp_hst);
  o::cells->build(o::pp, o::n);
  get_ticketZ(o::pp, o::n, &o::tz);
  if (rbcs) rbc::gen_quants("rbc.off", "rbcs-ic.txt", /**/ &r::q);
  MC(MPI_Barrier(m::cart));
  
  long nsteps = (int)(tend / dt);
  MSG0("will take %ld steps", nsteps);
  if (walls || solids) {
    solids0 = false;  /* global */
    gen();
    dSync();
    if (walls) wall::gen_ticket(w::q, &w::t);
    flu::get_ticketZ(o::pp, o::n, &o::tz);
    solids0 = solids;
    run(wall_creation, nsteps);
  } else {
    solids0 = solids;
    run(            0, nsteps);
  }
}

}
