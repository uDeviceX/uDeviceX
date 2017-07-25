namespace rex {

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
    DeviceBuffer<Particle> dstate;
    PinnedHostBuffer<Particle> hstate;
    PinnedHostBuffer<Force> result;
    std::vector<Particle> pmessage;
    void preserve_resize(int n) {
        dstate.resize(n);
        hstate.preserve_resize(n);
        result.resize(n);
        history.update(n);
    }
    int expected() const {return (int)ceil(history.max() * 1.1);}
};

class LocalHalo {
    TimeSeriesWindow history;
public:
    LocalHalo() {
        scattered_indices = new DeviceBuffer<int>;
        result            = new PinnedHostBuffer<Force>;
    }
    ~LocalHalo() {
        delete scattered_indices;
        delete result;
    }
    DeviceBuffer<int>* scattered_indices;
    PinnedHostBuffer<Force>* result;
    void resize(int n) {
        scattered_indices->resize(n);
        result->resize(n);
    }
    void update() { history.update(result->S);}
    int expected() const { return (int)ceil(history.max() * 1.1);}
    int capacity() const { return scattered_indices->C;}
};

int cnt;
int recv_counts[26], send_counts[26];

DeviceBuffer<int> *packscount, *packsstart, *packsoffset, *packstotalstart;
PinnedHostBuffer1<int> *host_packstotalstart, *host_packstotalcount;
DeviceBuffer<Particle> *packbuf;
PinnedHostBuffer<Particle> *host_packbuf;

std::vector<MPI_Request> reqsendC, reqrecvC, reqsendP, reqrecvP, reqsendA, reqrecvA;
RemoteHalo *remote[26];
LocalHalo  *local[26];
}
