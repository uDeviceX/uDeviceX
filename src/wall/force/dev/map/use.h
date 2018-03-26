/* true if `i' is bigger than the number of remote particles */
static __device__ int endp(const Map m, int i) { return i >= m.str2; }
static __device__ int m2id(const Map m, int i) {
    /* return remote particle id */
    int m1, m2, id;
    m1 = (i >= m.str0);
    m2 = (i >= m.str1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}
