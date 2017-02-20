/*
 *  solute-exchange.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved. */

enum {
  TAGBASE_C = 113, TAGBASE_P = 365, TAGBASE_A = 668,
  TAGBASE_P2 = 1055, TAGBASE_A2 = 1501
};

class TimeSeriesWindow {
  static const int N = 200;
  int count, data[N];
public:
  TimeSeriesWindow() : count(0) {}
  void update(int val) { data[count++ % N] = ::max(0, val); }
  int max() const {
    int retval = 0;
    for (int i = 0; i < min(N, count); ++i) retval = ::max(data[i], retval);
    return retval;
  }
};

class RemoteHalo {
  TimeSeriesWindow history;
public:
  SimpleDeviceBuffer<Particle> dstate;
  PinnedHostBuffer<Particle> hstate;
  PinnedHostBuffer<Acceleration> result;
  std::vector<Particle> pmessage;
  void preserve_resize(int n) {
    dstate.resize(n);
    hstate.preserve_resize(n);
    result.resize(n);
    history.update(n);
  }
  int expected() const {return (int)ceil(history.max() * 1.1);}
  int capacity() const {return dstate.capacity; }
};

class LocalHalo {
  TimeSeriesWindow history;
public:
  SimpleDeviceBuffer<int> scattered_indices;
  PinnedHostBuffer<Acceleration> result;
  void resize(int n) {
    scattered_indices.resize(n);
    result.resize(n);
  }
  void update() { history.update(result.size);}
  int expected() const { return (int)ceil(history.max() * 1.1);}
  int capacity() const { return scattered_indices.capacity;}
};

class SoluteExchange {
public:
  MPI_Comm cartcomm;
  int iterationcount;
  int nranks, dstranks[26], dims[3], periods[3], coords[3], myrank,
    recv_tags[26], recv_counts[26], send_counts[26];
  cudaEvent_t evPpacked, evAcomputed;
  SimpleDeviceBuffer<int> *packscount, *packsstart, *packsoffset, *packstotalstart;
  PinnedHostBuffer<int> *host_packstotalstart, *host_packstotalcount;
  SimpleDeviceBuffer<Particle> *packbuf;
  PinnedHostBuffer<Particle> host_packbuf;
  
  std::vector<ParticlesWrap> wsolutes;
  std::vector<MPI_Request> reqsendC, reqrecvC, reqsendP, reqrecvP, reqsendA,
    reqrecvA;
  RemoteHalo remote[26];
  LocalHalo  local[26];

  void _adjust_packbuffers() {
    int s = 0;
    for (int i = 0; i < 26; ++i) s += 32 * ((local[i].capacity() + 31) / 32);
    packbuf->resize(s);
    host_packbuf.resize(s);
  }

  void _wait(std::vector<MPI_Request> &v) {
    MPI_Status statuses[v.size()];
    if (v.size()) MPI_CHECK(MPI_Waitall(v.size(), &v.front(), statuses));
    v.clear();
  }

  void _postrecvC() {
    for (int i = 0; i < 26; ++i) {
      MPI_Request reqC;
      MPI_CHECK(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
			  TAGBASE_C + recv_tags[i], cartcomm, &reqC));
      reqrecvC.push_back(reqC);
    }
  }

  void _postrecvP() {
    for (int i = 0; i < 26; ++i) {
      MPI_Request reqP;
      remote[i].pmessage.resize(remote[i].expected());
      MPI_CHECK(MPI_Irecv(&remote[i].pmessage.front(), remote[i].expected() * 6,
			  MPI_FLOAT, dstranks[i], TAGBASE_P + recv_tags[i],
			  cartcomm, &reqP));
      reqrecvP.push_back(reqP);
    }
  }

  void _postrecvA() {
    for (int i = 0; i < 26; ++i) {
      MPI_Request reqA;

      MPI_CHECK(MPI_Irecv(local[i].result.data, local[i].result.size * 3,
			  MPI_FLOAT, dstranks[i], TAGBASE_A + recv_tags[i],
			  cartcomm, &reqA));
      reqrecvA.push_back(reqA);
    }
  }

  void _not_nan(float*, int) const {}
  void _pack_attempt(cudaStream_t stream);
  SoluteExchange(MPI_Comm cartcomm);
  void bind_solutes(std::vector<ParticlesWrap> wsolutes) {this->wsolutes = wsolutes;}
  void pack_p(cudaStream_t stream);
  void post_p(cudaStream_t stream, cudaStream_t downloadstream);
  void recv_p(cudaStream_t uploadstream);
  void halo(cudaStream_t uploadstream, cudaStream_t stream);
  void post_a();
  void recv_a(cudaStream_t stream);
  ~SoluteExchange();
};
