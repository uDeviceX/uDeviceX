namespace rex {
class LocalHalo {
    TimeSeriesWindow history;
public:
    DeviceBuffer<int>* indexes;
    PinnedHostBuffer<Force>* ff;

    LocalHalo() {
        indexes = new DeviceBuffer<int>;
        ff      = new PinnedHostBuffer<Force>;
    }
    ~LocalHalo() {
        delete indexes;
        delete ff;
    }
    void resize(int n) {
        indexes->resize(n);
        ff->resize(n);
    }
    void update() { history.update(ff->S);}
    int expected() const { return (int)ceil(history.max() * 1.1);}
    int size() const { return indexes->C;}
};
}
