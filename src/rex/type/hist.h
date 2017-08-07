namespace rex {
class History {
    static const int N = 200;
    int cnt, D[N];
public:
    History() : cnt(0) {}
    void update(int val) { D[cnt++ % N] = ::max(0, val); }
    int max() const {
        int i, rc = 0;
        for (i = 0; i < min(N, cnt); ++i) rc = ::max(D[i], rc);
        return rc;
    }
};
}
