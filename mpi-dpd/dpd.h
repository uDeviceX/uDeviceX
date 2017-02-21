// see the vanilla version of this code for details about how this class
// operates
namespace DPD {
class ComputeDPD {
public:
  Logistic::KISS *local_trunk;

  /* allocated inside init1 */
  Logistic::KISS *interrank_trunks[26]; 

  bool interrank_masks[26];

  MPI_Comm cartcomm;
  MPI_Request sendreq[26 * 2], recvreq[26], sendcellsreq[26], recvcellsreq[26],
    sendcountreq[26], recvcountreq[26];
  int recv_tags[26], recv_counts[26], nlocal, nactive;
  bool firstpost;
  int myrank, nranks, dims[3], periods[3], coords[3], dstranks[26];

  // zero-copy allocation for acquiring the message offsets in the gpu send
  // buffer
  int *required_send_bag_size, *required_send_bag_size_host;

  // plain copy of the offsets for the cpu (i speculate that reading multiple
  // times the zero-copy entries is slower)
  int nsendreq;
  int3 halosize[26];
  float safety_factor;
  cudaEvent_t evfillall, evuploaded, evdownloaded;
  int basetag;

  struct SendHalo {
    int expected;
    DeviceBuffer<int> scattered_entries, tmpstart, tmpcount, dcellstarts;
    DeviceBuffer<Particle> dbuf;
    PinnedHostBuffer<int> hcellstarts;
    PinnedHostBuffer<Particle> hbuf;
    void setup(int estimate, int nhalocells) {
      adjust(estimate);
      dcellstarts.resize(nhalocells + 1);
      hcellstarts.resize(nhalocells + 1);
      tmpcount.resize(nhalocells + 1);
      tmpstart.resize(nhalocells + 1);
    }

    void adjust(int estimate) {
      expected = estimate;
      hbuf.resize(estimate);
      dbuf.resize(estimate);
      scattered_entries.resize(estimate);
    }
  } sendhalos[26];

  struct RecvHalo {
    int expected;
    PinnedHostBuffer<int> hcellstarts;
    PinnedHostBuffer<Particle> hbuf;
    DeviceBuffer<Particle> dbuf;
    DeviceBuffer<int> dcellstarts;
    void setup(int estimate, int nhalocells) {
      adjust(estimate);
      dcellstarts.resize(nhalocells + 1);
      hcellstarts.resize(nhalocells + 1);
    }
    void adjust(int estimate) {
      expected = estimate;
      hbuf.resize(estimate);
      dbuf.resize(estimate);
    }

  } recvhalos[26];

  
  ComputeDPD(MPI_Comm cartcomm);
  ~ComputeDPD();

  void remote_interactions(Particle *p, int n, Acceleration *a,
                           cudaStream_t stream, cudaStream_t uploadstream);

  void local_interactions(Particle *xyzuvw, float4 *xyzouvwo,
                          ushort4 *xyzo_half, int n, Acceleration *a,
                          int *cellsstart, int *cellscount,
                          cudaStream_t stream);
  void init1(MPI_Comm cartcomm);

  void post_expected_recv();
  void _pack_all(Particle *p, int n,
		 bool update_baginfos, cudaStream_t stream);
  int nof_sent_particles();
  void _cancel_recv();
  void init0(MPI_Comm cartcomm, int basetag);
  void pack(Particle *p, int n, int *cellsstart,
	    int *cellscount, cudaStream_t stream);
  void post(Particle *p, int n, cudaStream_t stream,
	    cudaStream_t downloadstream);
  void recv(cudaStream_t stream, cudaStream_t uploadstream);
  void adjust_message_sizes(ExpectedMessageSizes sizes);
};
}
