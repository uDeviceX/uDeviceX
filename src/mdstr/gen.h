namespace mdstr {
namespace gen {

enum DataLoc {Host, Device};

// https://stackoverflow.com/questions/19959637/double-templated-function-instantiation-fails
template <typename T, DataLoc LOC> struct cpy_pck {void operator() (T *dst, const T *src, int n);};
template <typename T, DataLoc LOC> struct cpy_upck{void operator() (T *dst, const T *src, int n);};
template <typename T> MPI_Datatype MType();

template <typename T, DataLoc LOC>
void pack(int *reord[27], const int counts[27],  const T *dd, int npd, /**/ pbuf<T> *b) {
    int fid, count, j, src;
    for (fid = 0; fid < 27; ++fid) {
        count = counts[fid];
        for (j = 0; j < count; ++j) {
            src = reord[fid][j];
            cpy_pck <T, LOC> (b->dd[fid] + j * npd, dd + src * npd, npd);
        }
    }
}

template <typename T>
void post_send(int npd, const int counts[27], const pbuf<T> *b, MPI_Comm cart, int bt, int rnk_ne[27],
               /**/ MPI_Request sreq[26]) {
    for (int i = 1; i < 27; ++i)
        MC(l::m::Isend(b->dd[i], npd * counts[i], MType<T>(), rnk_ne[i], bt + i, cart, sreq + i - 1));
}

template <typename T>
void post_recv(MPI_Comm cart, int nmax, int bt, int ank_ne[27], /**/ pbuf<T> *b, MPI_Request rreq[26]) {
    for (int i = 1; i < 27; ++i)
        MC(l::m::Irecv(b->dd[i], nmax, MType<T>, ank_ne[i], bt + i, cart, rreq + i - 1));
}

template <typename T, DataLoc LOC>
int unpack(int npd, const pbuf<T> *b, const int counts[27], /**/ T *dd) {
    int nm = 0;
    for (int i = 0; i < 27; ++i) {
        int c = counts[i];
        int n = c * npd;
        if (n) cpy_upck <T, LOC> (dd + nm * npd, b->dd[i], n);
        nm += c;
    }
    return nm;
}

/* shift() is not generic */

/* template specializations */

template <typename T>
struct cpy_pck <T, Host>   {void operator() (T *dst, const T *src, int n) {memcpy(dst, src, n*sizeof(T));}};

template <typename T>
struct cpy_pck <T, Device> {void operator() (T *dst, const T *src, int n) {CC(cudaMemcpyAsync(dst, src, n*sizeof(T), D2H));}};

template <typename T>
struct cpy_upck <T, Host>   {void operator() (T *dst, const T *src, int n) {memcpy(dst, src, n*sizeof(T));}};

template <typename T>
struct cpy_upck <T, Device> {void operator() (T *dst, const T *src, int n) { CC(cudaMemcpyAsync(dst, src, n*sizeof(T), H2D));}};

template <> MPI_Datatype MType<int>()      {return MPI_INT;}
template <> MPI_Datatype MType<Particle>() {return datatype::particle;}
template <> MPI_Datatype MType<Solid>()    {return datatype::solid;}

} // gen
} // mdstr
