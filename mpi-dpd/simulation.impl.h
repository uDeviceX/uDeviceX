namespace Sim {
static void sim_update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(make_texture, cudaFuncCachePreferShared));
  int np = s_pp->S;
  xyzouvwo->resize(2 * np);
  xyzo_half->resize(np);
  if (np)
    make_texture<<<(np + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(
	xyzouvwo->D, xyzo_half->D, (float *)s_pp->D, np);

}

/* set initial velocity of a particle */
static void sim_ic_vel0(float x, float y, float z,
			float *vx, float *vy, float *vz) {
  *vx = gamma_dot * z; *vy = 0; *vz = 0; /* TODO: works only for one
					    processor */
}

static void sim_ic_vel(Particle* pp, int np) { /* assign particle
					velocity based on position */
  for (int ip = 0; ip < np; ip++) {
    Particle p = pp[ip];
    float x = p.x[0], y = p.x[1], z = p.x[2], vx, vy, vz;
    sim_ic_vel0(x, y, z, &vx, &vy, &vz);
    p.u[0] = vx; p.u[1] = vy; p.u[2] = vz;
  }
}

static std::vector<Particle> _ic_pos() { /* generate particle position */
  srand48(0);
  std::vector<Particle> pp;
  int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
  int iz, iy, ix, l, nd = numberdensity;
  float x, y, z, dr = 0.99;
  for (iz = 0; iz < L[2]; iz++)
    for (iy = 0; iy < L[1]; iy++)
      for (ix = 0; ix < L[0]; ix++) {
	/* edge of a cell */
	int xlo = -L[0]/2 + ix, ylo = -L[1]/2 + iy, zlo = -L[2]/2 + iz;
	for (l = 0; l < nd; l++) {
	  Particle p = Particle();
	  x = xlo + dr * drand48(), y = ylo + dr * drand48(), z = zlo + dr * drand48();
	  p.x[0] = x; p.x[1] = y; p.x[2] = z;
	  pp.push_back(p);
	}
      }
  fprintf(stderr, "(simulation) generated %d particles\n", pp.size());
  return pp;
}

static std::vector<Particle> _ic() { /* initial conditions for position and
				 velocity */
  std::vector<Particle> pp = _ic_pos();
  sim_ic_vel(&pp.front(), pp.size());
  return pp;
}

static void sim_redistribute() {
  RedistPart::pack(s_pp->D, s_pp->S);
  if (rbcs) RedistRBC::extent(r_pp->D, Cont::ncells);
  RedistPart::send();
  if (rbcs) RedistRBC::pack_sendcount(r_pp->D, Cont::ncells);
  RedistPart::bulk(s_pp->S, cells->start, cells->count);
  const int newnp = RedistPart::recv_count();
  if (rbcs) {
    Cont::ncells = RedistRBC::post();
    r_pp->resize(Cont::ncells*Cont::nvertices); r_aa->resize(Cont::ncells*Cont::nvertices);
  }
  s_pp0->resize(newnp); s_aa0->resize(newnp);
  xyzouvwo->resize(newnp * 2);
  xyzo_half->resize(newnp);
  RedistPart::recv_unpack(s_pp0->D,
			  xyzouvwo->D, xyzo_half->D,
			  newnp, cells->start, cells->count);
  
  swap(s_pp, s_pp0); swap(s_aa, s_aa0);
  if (rbcs) RedistRBC::unpack(r_pp->D, Cont::ncells);
}

void sim_remove_bodies_from_wall() {
  if (!rbcs)         return;
  if (!Cont::ncells) return;
  DeviceBuffer<int> marks(Cont::pcount());
  k::wall::fill_keys<<<(Cont::pcount() + 127) / 128, 128>>>
    (r_pp->D, Cont::pcount(), marks.D);

  vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, cudaMemcpyDeviceToHost));
  int nbodies = Cont::ncells;
  std::vector<int> tokill;
  for (int i = 0; i < nbodies; ++i) {
    bool valid = true;
    for (int j = 0; j < Cont::nvertices && valid; ++j)
      valid &= 0 == tmp[j + Cont::nvertices * i];
    if (!valid) tokill.push_back(i);
  }
  rbc_remove_resize(r_pp, r_aa, &tokill.front(), tokill.size());
  Cont::clear_velocity(r_pp);
}

void sim_create_walls() {
  int nsurvived = 0;
  ExpectedMessageSizes new_sizes;
  Wall::init(s_pp->D, s_pp->S,
	    nsurvived, new_sizes); wall_created = true;
  resize2(s_pp, s_aa, nsurvived);
  Cont::clear_velocity(s_pp);
  cells->build(s_pp->D, s_pp->S, NULL, NULL);
  sim_update_helper_arrays();

  // remove cells touching the wall
  sim_remove_bodies_from_wall();
  H5PartDump sd("survived-particles.h5part", activecomm, Cont::cartcomm);
  Particle *pp = new Particle[s_pp->S];
  CC(cudaMemcpy(pp, s_pp->D,
			sizeof(Particle) * s_pp->S,
			cudaMemcpyDeviceToHost));
  sd.dump(pp, s_pp->S);
  delete[] pp;
}

void sim_forces() {
  SolventWrap wsolvent(s_pp->D, s_pp->S,
		       s_aa->D, cells->start, cells->count);
  std::vector<ParticlesWrap> wsolutes;
  if (rbcs)
    wsolutes.push_back(ParticlesWrap(r_pp->D,
				     Cont::pcount(), r_aa->D));
  FSI::bind_solvent(wsolvent);
  SolEx::bind_solutes(wsolutes);
  Cont::clear_acc(s_aa);
  if (rbcs) Cont::clear_acc(r_aa);
  DPD::pack(s_pp->D, s_pp->S, cells->start, cells->count);
  SolEx::pack_p();
  if (contactforces) cnt::build_cells(wsolutes);
  DPD::local_interactions(s_pp->D, xyzouvwo->D, xyzo_half->D,
			 s_pp->S, s_aa->D, cells->start,
			 cells->count);
  DPD::post(s_pp->D, s_pp->S);
  SolEx::post_p();
  if (rbcs && wall_created) Wall::interactions(r_pp->D, Cont::pcount(), r_aa->D);
  if (wall_created)         Wall::interactions(s_pp->D, s_pp->S, s_aa->D);
  DPD::recv();
  SolEx::recv_p();
  SolEx::halo();
  DPD::remote_interactions(s_pp->D, s_pp->S, s_aa->D);
  FSI::bulk(wsolutes);
  if (contactforces) cnt::bulk(wsolutes);
  if (rbcs)
    CudaRBC::forces_nohost(Cont::ncells,
			   (float *)r_pp->D, (float *)r_aa->D);
  SolEx::post_a();
  SolEx::recv_a();
}

void sim_tmp_init() {
  MC(MPI_Comm_dup(activecomm, &myactivecomm));
  MC(MPI_Comm_dup(Cont::cartcomm, &mycartcomm));
  dump_part = new H5PartDump("allparticles.h5part", activecomm, Cont::cartcomm);
  dump_field = new H5FieldDump (Cont::cartcomm);

  int rank;
  MC(MPI_Comm_rank(myactivecomm, &rank));
  MC(MPI_Barrier(myactivecomm));
}

void sim_tmp_final() {
  delete dump_part;
  delete dump_field;
  if (dump_part_solvent) delete dump_part_solvent;
  CC(cudaEventDestroy(evdownloaded));
}

static void sim_datadump_async(int idtimestep) {
  static int iddatadump = 0;
    CC(cudaEventSynchronize(evdownloaded));

    int n = particles_datadump->S;
    Particle *p = particles_datadump->D;
    Acceleration *a = accelerations_datadump->D;

    diagnostics(myactivecomm, mycartcomm, p, n, dt, datadump_idtimestep);
    if (hdf5part_dumps) {
      if (!dump_part_solvent && walls && datadump_idtimestep >= wall_creation_stepid) {
	dump_part->close();
	dump_part_solvent =
	    new H5PartDump("solvent-particles.h5part", activecomm, Cont::cartcomm);
      }
      if (dump_part_solvent) dump_part_solvent->dump(p, n);
      else                   dump_part->dump(p, n);
    }

    if (hdf5field_dumps && (datadump_idtimestep % steps_per_hdf5dump == 0))
      dump_field->dump(activecomm, p, datadump_nsolvent, datadump_idtimestep);

    if (rbcs)
      Cont::rbc_dump(myactivecomm, p + datadump_nsolvent,
		     a + datadump_nsolvent, datadump_nrbcs, iddatadump);
    ++iddatadump;
}

void sim_datadump(const int idtimestep) {
  int n = s_pp->S;
  if (rbcs) n += Cont::pcount();
  particles_datadump->resize(n);
  accelerations_datadump->resize(n);
#include "simulation.hack.h"
  CC(cudaMemcpyAsync(particles_datadump->D, s_pp->D,
			     sizeof(Particle) * s_pp->S,
			     cudaMemcpyDeviceToHost, 0));
  CC(cudaMemcpyAsync(accelerations_datadump->D, s_aa->D,
			     sizeof(Acceleration) * s_pp->S,
			     cudaMemcpyDeviceToHost, 0));
  int start = s_pp->S;
  if (rbcs) {
    CC(cudaMemcpyAsync(
	particles_datadump->D + start, r_pp->D,
	sizeof(Particle) * Cont::pcount(), cudaMemcpyDeviceToHost, 0));
    CC(cudaMemcpyAsync(
	accelerations_datadump->D + start, r_aa->D,
	sizeof(Acceleration) * Cont::pcount(), cudaMemcpyDeviceToHost, 0));
    start += Cont::pcount();
  }
  CC(cudaEventRecord(evdownloaded, 0));

  datadump_idtimestep = idtimestep;
  datadump_nsolvent = s_pp->S;
  datadump_nrbcs = rbcs ? Cont::pcount() : 0;
  sim_datadump_async(idtimestep);
}

static void sim_update_and_bounce() {
  Cont::upd_stg2_and_1(s_pp, s_aa, false, driving_acceleration);
  if (rbcs) Cont::upd_stg2_and_1(r_pp, r_aa, true, driving_acceleration);
  if (wall_created) {
    Wall::bounce(s_pp->D, s_pp->S);
    if (rbcs) Wall::bounce(r_pp->D, Cont::pcount());
  }
}

void sim_init(MPI_Comm cartcomm_, MPI_Comm activecomm_) {
  Cont::cartcomm = cartcomm_; activecomm = activecomm_;
  RedistRBC::redistribute_rbcs_init(Cont::cartcomm);
  DPD::init(Cont::cartcomm);
  FSI::init(Cont::cartcomm);
  SolEx::init(Cont::cartcomm);
  cnt::init(Cont::cartcomm);
  cells   = new CellLists(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);
  particles_datadump     = new PinnedHostBuffer<Particle>;
  accelerations_datadump = new PinnedHostBuffer<Acceleration>;

  xyzouvwo    = new DeviceBuffer<float4>;
  xyzo_half = new DeviceBuffer<ushort4>;
  if (rbcs) {
    r_pp = new StaticDeviceBuffer<Particle>;
    r_aa = new StaticDeviceBuffer<Acceleration>;
  }

  Wall::trunk = new Logistic::KISS;
  RedistPart::redist_part_init(Cont::cartcomm);
  nsteps = (int)(tend / dt);
  MC(MPI_Comm_rank(activecomm, &Cont::rank));

  int dims[3], periods[3]; /* `coords' is global */
  MC(MPI_Cart_get(Cont::cartcomm, 3, dims, periods, Cont::coords));
  Cont::origin = make_float3((0.5 + Cont::coords[0]) * XSIZE_SUBDOMAIN,
			     (0.5 + Cont::coords[1]) * YSIZE_SUBDOMAIN,
			     (0.5 + Cont::coords[2]) * ZSIZE_SUBDOMAIN);
  Cont::globalextent = make_float3(dims[0] * XSIZE_SUBDOMAIN,
				   dims[1] * YSIZE_SUBDOMAIN,
				   dims[2] * ZSIZE_SUBDOMAIN);
  s_pp  = new StaticDeviceBuffer<Particle>;
  s_aa  = new StaticDeviceBuffer<Acceleration>;
  s_pp0 = new StaticDeviceBuffer<Particle>;
  s_aa0 = new StaticDeviceBuffer<Acceleration>;

  vector<Particle> ic = _ic();
  resize2(s_pp, s_aa  , ic.size());
  resize2(s_pp0, s_aa , ic.size());
  CC(cudaMemcpy(s_pp->D, &ic.front(),
			sizeof(Particle) * ic.size(),
			cudaMemcpyHostToDevice));
  cells->build(s_pp->D, s_pp->S, NULL, NULL);
  sim_update_helper_arrays();

  if (rbcs) {
    Cont::rbc_init();
    Cont::setup(r_pp, r_aa, "rbcs-ic.txt");
  }

  CC(cudaEventCreate(&evdownloaded,
			     cudaEventDisableTiming | cudaEventBlockingSync));
  particles_datadump->resize(s_pp->S * 1.5);
  accelerations_datadump->resize(s_pp->S * 1.5);

  sim_tmp_init();
}

void sim_run() {
  if (Cont::rank == 0 && !walls) printf("will take %ld steps\n", nsteps);
  sim_redistribute();
  sim_forces();
  if (!walls && pushtheflow) driving_acceleration = hydrostatic_a;
  int it;
  for (it = 0; it < nsteps; ++it) {
    sim_redistribute();
    if (walls && it >= wall_creation_stepid && !wall_created) {
      CC(cudaDeviceSynchronize());
      sim_create_walls();
      sim_redistribute();
      if (pushtheflow) driving_acceleration = hydrostatic_a;
      if (Cont::rank == 0)
	fprintf(stderr, "the simulation consists of %ld steps\n", nsteps - it);
    }
    sim_forces();
    if (it % steps_per_dump == 0) sim_datadump(it);
    sim_update_and_bounce();
  }
  sim_is_done = true;
}

void sim_close() {

  sim_tmp_final();
  RedistPart::redist_part_close();

  delete r_pp; delete r_aa;
  
  cnt::close();
  delete cells;
  SolEx::close();
  FSI::close();
  DPD::close();
  RedistRBC::redistribute_rbcs_close();

  delete particles_datadump;
  delete accelerations_datadump;
  delete xyzouvwo;
  delete xyzo_half;
  delete Wall::trunk;
  delete s_pp; delete s_aa;
  delete s_pp0; delete s_aa0;  
}
}
