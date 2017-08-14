namespace rex {
struct RemoteHalo {
    History h;
    DeviceBuffer<Particle> dstate;
    PinnedHostBuffer<Particle> hstate;
    PinnedHostBuffer<Force> ff;
    std::vector<Particle> pp;
};

namespace re {
void resize(RemoteHalo *r, int n) {
    r->dstate.resize(n);
    r->hstate.preserve_resize(n);
    r->ff.resize(n);
}

int expected(RemoteHalo *r) {
    return (int)ceil(r->h.max() * 1.1);
}

}
}
