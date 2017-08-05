namespace rex {
class LocalHalo {
    History hist;
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
    void update() { hist.update(ff->S);}
    int expected() const { return (int)ceil(hist.max() * 1.1);}
    int size() const { return indexes->C;}
};
}
