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
  int size;
  float3 origin, globalextent;

  SimpleDeviceBuffer<Particle> xyzuvw;
  SimpleDeviceBuffer<Acceleration> axayaz;

  void resize(int n);
  void preserve_resize(int n);
  void upd_stg1      (bool rbcflag, float driving_acceleration, cudaStream_t stream);
  void upd_stg2_and_1(bool rbcflag, float driving_acceleration, cudaStream_t stream);
  void clear_velocity();

  void clear_acc(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(axayaz.data, 0,
			       sizeof(Acceleration) * axayaz.size, stream));
  }
};

extern int nvertices;
class CollectionRBC : public ParticleArray
{
 protected:
  MPI_Comm cartcomm;
  int ncells, myrank, coords[3];

  void _initialize(float *device_xyzuvw, float (*transform)[4]);

  static void _dump(const char * format4ply,
		    MPI_Comm comm, MPI_Comm cartcomm,  int ncells,
		    Particle *  p,  Acceleration *  a,  int n, int iddatadump);
 public:
  CollectionRBC(MPI_Comm cartcomm);

  void setup(const char *path2ic);

  Particle * data() { return xyzuvw.data; }
  Acceleration * acc() { return axayaz.data; }

  void remove(int *  entries, int nentries);
  void resize(int rbcs_count);
  void preserve_resize(int n);

  int count()  {return ncells; }
  int pcount() {return ncells * nvertices;}

  static void dump(MPI_Comm comm, MPI_Comm cartcomm,
                   Particle * p, Acceleration * a, int n, int iddatadump)
  {
    _dump("ply/rbcs-%05d.ply", comm, cartcomm, n / nvertices, p, a, n, iddatadump);
  }
};
