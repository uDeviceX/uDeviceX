namespace rex {
struct RemoteHalo {
    Particle* dstate;
    PinnedHostBuffer<Particle> hstate;
    PinnedHostBuffer<Force> ff;
    std::vector<Particle> pp;
    int n;
};

namespace re {
void resize(RemoteHalo *r, int n) {
    r->n = n;
    r->hstate.preserve_resize(n);
    r->ff.resize(n);
}
}
}
