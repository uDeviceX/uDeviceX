namespace FSI {
class ComputeFSI {
  SolventWrap wsolvent;
  Logistic::KISS* local_trunk;
public:
  void bind_solvent(SolventWrap wrap) {wsolvent = wrap;}
  explicit ComputeFSI(MPI_Comm comm);
  ~ComputeFSI();
  void bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);
  /*override of SoluteExchange::Visitor::halo*/
  void halo(ParticlesWrap solutes[26], cudaStream_t stream);
};
}
