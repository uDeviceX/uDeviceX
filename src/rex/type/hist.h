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
}
