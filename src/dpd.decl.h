// see the vanilla version of this code for details about how this class
// operates
namespace DPD {
Logistic::KISS *local_trunk;

/* allocated inside init1 */
Logistic::KISS *interrank_trunks[26];

bool interrank_masks[26];

MPI_Comm cart;
MPI_Request sendreq[26 * 2], recvreq[26], sendcellsreq[26], recvcellsreq[26],
    sendcountreq[26], recvcountreq[26];
int recv_tags[26], recv_counts[26], nlocal, nactive;
bool firstpost;
int dstranks[26];

// zero-copy allocation for acquiring the message offsets in the gpu send
// buffer
int *required_send_bag_size, *required_send_bag_size_host;

// plain copy of the offsets for the cpu (i speculate that reading multiple
// times the zero-copy entries is slower)
int nsendreq;
int3 halosize[26];
float safety_factor;
cudaEvent_t evfillall, evuploaded, evdownloaded;

struct SendHalo {
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

    DeviceBuffer<int> *scattered_entries, *tmpstart, *tmpcount, *dcellstarts;
    DeviceBuffer<Particle> *dbuf;
    PinnedHostBuffer<Particle> *hbuf;
    PinnedHostBuffer<int> *hcellstarts;

    int expected;
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
    int expected;
    PinnedHostBuffer<int> *hcellstarts;
    PinnedHostBuffer<Particle> *hbuf;
    DeviceBuffer<Particle> *dbuf;
    DeviceBuffer<int> *dcellstarts;
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
}
