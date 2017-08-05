namespace rex {
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
}
