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
    PinnedHostBuffer<Force> ff;
    std::vector<Particle> pmessage;
    void preserve_resize(int n) {
        dstate.resize(n);
        hstate.preserve_resize(n);
        ff.resize(n);
        history.update(n);
    }
    int expected() const {return (int)ceil(history.max() * 1.1);}
};

class LocalHalo {
    TimeSeriesWindow history;
public:
    LocalHalo() {
        indexes = new DeviceBuffer<int>;
        ff            = new PinnedHostBuffer<Force>;
    }
    ~LocalHalo() {
        delete indexes;
        delete ff;
    }
    DeviceBuffer<int>* indexes;
    PinnedHostBuffer<Force>* ff;
    void resize(int n) {
        indexes->resize(n);
        ff->resize(n);
    }
    void update() { history.update(ff->S);}
    int expected() const { return (int)ceil(history.max() * 1.1);}
    int size() const { return indexes->C;}
};
}
