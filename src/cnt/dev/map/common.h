/** TODO: should be merged with Map in fsi:: */
enum {FULL, EMPTY}; /* map status */
struct Map { /* helps to find remote particle */
    int  org0, org1, org2;
    int cnt0, cnt1, cnt2;
};

/* true if `i' is bigger than the number of remote particles */
static __device__ int endp(const Map m, int i) { return i >= m.cnt2; }

static __device__ int m2id(const Map m, int i) {
    /* return remote particle id */
    int m1, m2;
    int id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}
