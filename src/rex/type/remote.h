namespace rex {
class RemoteHalo {
    History hist;
public:
    DeviceBuffer<Particle> dstate;
    PinnedHostBuffer<Particle> hstate;
    PinnedHostBuffer<Force> ff;
    std::vector<Particle> pmessage;

    void resize(int n) {
        dstate.resize(n);
        hstate.preserve_resize(n);
        ff.resize(n);
        hist.update(n);
    }
    int expected() const {return (int)ceil(hist.max() * 1.1);}
};
}
