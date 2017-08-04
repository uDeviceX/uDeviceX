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
        indexes = new DeviceBuffer<int>;
        result            = new PinnedHostBuffer<Force>;
    }
    ~LocalHalo() {
        delete indexes;
        delete result;
    }
    DeviceBuffer<int>* indexes;
    PinnedHostBuffer<Force>* result;
    void resize(int n) {
        indexes->resize(n);
        result->resize(n);
    }
    void update() { history.update(result->S);}
    int expected() const { return (int)ceil(history.max() * 1.1);}
    int size() const { return indexes->C;}
};
}
