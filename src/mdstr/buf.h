namespace mdstr {
namespace gen {

template <typename T>
struct pbuf { /* pack buffer */
    T *dd[27];
};

template <typename T>
void alloc_buf(int istart, int nmax, /**/ pbuf<T> *b) {
    for (int i = istart; i < 27; ++i) b->dd[i] = new T[nmax];
}

template <typename T>
void free_buf(int istart, /**/ pbuf<T> *b) {
    for (int i = istart; i < 27; ++i) delete[] b->dd[i];
}

} // gen
} // mdstr
