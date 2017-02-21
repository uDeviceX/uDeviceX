// see the vanilla version of this code for details about how this class
// operates
class ComputeDPD : public SolventExchange {
  Logistic::KISS* local_trunk;
  Logistic::KISS interrank_trunks[26];

  bool interrank_masks[26];

public:
  ComputeDPD(MPI_Comm cartcomm);
  ~ComputeDPD();

  void remote_interactions(Particle *p, int n, Acceleration *a,
                           cudaStream_t stream, cudaStream_t uploadstream);

  void local_interactions(Particle *xyzuvw, float4 *xyzouvwo,
                          ushort4 *xyzo_half, int n, Acceleration *a,
                          int *cellsstart, int *cellscount,
                          cudaStream_t stream);
  void init1(MPI_Comm cartcomm);
};
