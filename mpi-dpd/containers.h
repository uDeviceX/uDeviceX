/*
 *  containers.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-05.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

struct ParticleArray
{
  int S; /* size */

  SimpleDeviceBuffer<Particle>     pp; /* xyzuvw */
  SimpleDeviceBuffer<Acceleration> aa; /* axayaz */

  void pa_resize(int n);
  void pa_preserve_resize(int n);
  void upd_stg1      (bool rbcflag, float driving_acceleration, cudaStream_t stream);
  void upd_stg2_and_1(bool rbcflag, float driving_acceleration, cudaStream_t stream);
  void clear_velocity();

  void clear_acc(cudaStream_t stream) {
    CC(cudaMemsetAsync(aa.D, 0, sizeof(Acceleration) * aa.S, stream));
  }
};

void rbc_dump(MPI_Comm comm, MPI_Comm cartcomm,
	      Particle* p, Acceleration* a, int n, int iddatadump);

extern int nvertices;
class CollectionRBC : public ParticleArray {
 protected:
  MPI_Comm cartcomm;
  int myrank;
  void _initialize(float *device_pp, float (*transform)[4]);
 public:
  int ncells;
  CollectionRBC(MPI_Comm cartcomm);
  void setup(const char *path2ic);
  void remove(int *  entries, int nentries);
  void rbc_resize(int rbcs_count);
  void rbc_preserve_resize(int n);
  int  pcount() {return ncells * nvertices;}
};
