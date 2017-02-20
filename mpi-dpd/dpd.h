//see the vanilla version of this code for details about how this class operates
class ComputeDPD : public SolventExchange
{
  Logistic::KISS local_trunk;
  Logistic::KISS interrank_trunks[26];

  bool interrank_masks[26];

 public:

  ComputeDPD(MPI_Comm cartcomm);

  void remote_interactions(const Particle * const p, const int n, Acceleration * const a, cudaStream_t stream, cudaStream_t uploadstream);

  void local_interactions(const Particle * const xyzuvw, const float4 * const xyzouvwo, const ushort4 * const xyzo_half, const int n, Acceleration * const a,
                          const int * const cellsstart, const int * const cellscount, cudaStream_t stream);
};
