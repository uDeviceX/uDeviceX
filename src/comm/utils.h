template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf) {
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
