namespace dpd { /* declaration of bufers */
struct SendHalo {
  DeviceBuffer<int> *scattered_entries, *tmpstart, *tmpcount, *dcellstarts;
  DeviceBuffer<Particle> *dbuf;
  PinnedHostBuffer<Particle> *hbuf;
  PinnedHostBuffer<int> *hcellstarts;
  int expected;

  SendHalo() {
    dbuf = new DeviceBuffer<Particle>;
    hbuf = new PinnedHostBuffer<Particle>;

    scattered_entries = new DeviceBuffer<int>;
    tmpstart = new DeviceBuffer<int>;
    tmpcount = new DeviceBuffer<int>;
    dcellstarts = new DeviceBuffer<int>;
    hcellstarts = new PinnedHostBuffer<int>;
  };

  ~SendHalo() {
    delete dbuf;
    delete hbuf;
    delete scattered_entries;
    delete tmpstart;
    delete tmpcount;
    delete dcellstarts;
    delete hcellstarts;
  }

  void setup(int estimate, int nhalocells) {
    adjust(estimate);
    dcellstarts->resize(nhalocells + 1);
    hcellstarts->resize(nhalocells + 1);
    tmpcount->resize(nhalocells + 1);
    tmpstart->resize(nhalocells + 1);
  }

  void adjust(int estimate) {
    expected = estimate;
    hbuf->resize(estimate);
    dbuf->resize(estimate);
    scattered_entries->resize(estimate);
  }
} * sendhalos[26];

struct RecvHalo {
  int expected;
  PinnedHostBuffer<int> *hcellstarts;
  PinnedHostBuffer<Particle> *hbuf;
  DeviceBuffer<Particle> *dbuf;
  DeviceBuffer<int> *dcellstarts;

  RecvHalo() {
    hcellstarts = new PinnedHostBuffer<int>;
    hbuf = new PinnedHostBuffer<Particle>;
    dbuf = new DeviceBuffer<Particle>;
    dcellstarts = new DeviceBuffer<int>;
  }
  ~RecvHalo() {
    delete hcellstarts;
    delete hbuf;
    delete dbuf;
    delete dcellstarts;
  }
  void setup(int estimate, int nhalocells) {
    adjust(estimate);
    dcellstarts->resize(nhalocells + 1);
    hcellstarts->resize(nhalocells + 1);
  }
  void adjust(int estimate) {
    expected = estimate;
    hbuf->resize(estimate);
    dbuf->resize(estimate);
  }
} * recvhalos[26];
} /* namespace dpd */
