namespace rex {
struct LocalHalo {
    History h;
    DeviceBuffer<int>* indexes;
    PinnedHostBuffer<Force>* ff;
};

namespace lo {
void update(LocalHalo *l) {
    l->h.update(l->ff->S);
}

int expected(LocalHalo *l) {
    return (int)ceil(l->h.max() * 1.1);
}

}
}
