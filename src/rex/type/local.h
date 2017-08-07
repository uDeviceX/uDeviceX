namespace rex {
struct LocalHalo {
    History h;
    DeviceBuffer<int>* indexes;
    PinnedHostBuffer<Force>* ff;

    void resize(int n) {
        indexes->resize(n);
        ff->resize(n);
    }
    void update() { h.update(ff->S);}
    int expected() const { return (int)ceil(h.max() * 1.1);}
    int size() const { return indexes->C;}
};

namespace lo {
void resize(LocalHalo l, int n) {
    l.resize(n);
    l.resize(n);
}

void update(LocalHalo l) {
    l.h.update(l.ff->S);
}

int expected(LocalHalo l) {
    return (int)ceil(l.h.max() * 1.1);
}

int size(LocalHalo l) {
    return l.indexes->C;
}
}
}
