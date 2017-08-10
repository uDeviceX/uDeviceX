namespace rex {
struct RemoteHalo {
    History h;
    DeviceBuffer<Particle> dstate;
    PinnedHostBuffer<Particle> hstate;
    PinnedHostBuffer<Force> ff;
    std::vector<Particle> pmessage;

    void resize(int n) {
        dstate.resize(n);
        hstate.preserve_resize(n);
        ff.resize(n);
        h.update(n);
    }
    int expected() const {return (int)ceil(h.max() * 1.1);}
};

namespace re {
void resize(RemoteHalo *r, int n) {
    r->dstate.resize(n);
    r->hstate.preserve_resize(n);
    r->ff.resize(n);
    r->h.update(n);
}

int expected(RemoteHalo *r) {
    return (int)ceil(r->h.max() * 1.1);
}

}
}
