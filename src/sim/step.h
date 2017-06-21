namespace sim {

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

}
