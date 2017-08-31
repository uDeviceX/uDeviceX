namespace mdstr {
namespace gen {

enum DataLoc {Host, Device};

template <typename T> void cpy(T *dst, const T *src, int n, cudaMemcpyKind cmk) {
    CC(cudaMemcpyAsync(dst, src, n * sizeof(T), cmk));
}

template <typename T> MPI_Datatype MType();

template <typename T, DataLoc LOC>
void pack(int *reord[27], const int counts[27],  const T *dd, int npd, /**/ pbuf<T> *b) {
    int fid, count, j, src;
    for (fid = 0; fid < 27; ++fid) {
        count = counts[fid];
        for (j = 0; j < count; ++j) {
            src = reord[fid][j];
            cpy(b->dd[fid] + j * npd, dd + src * npd, npd, LOC == Host ? H2H : D2H);
        }
    }
}

template <typename T>
void post_send(int npd, const int counts[27], const pbuf<T> *b, MPI_Comm cart, int bt, int rnk_ne[27],
               /**/ MPI_Request sreq[26]) {
    for (int i = 1; i < 27; ++i)
        MC(m::Isend(b->dd[i], npd * counts[i], MType<T>(), rnk_ne[i], bt + i, cart, sreq + i - 1));
}

template <typename T>
void post_recv(MPI_Comm cart, int nmax, int bt, int ank_ne[27], /**/ pbuf<T> *b, MPI_Request rreq[26]) {
    for (int i = 1; i < 27; ++i)
        MC(m::Irecv(b->dd[i], nmax, MType<T>(), ank_ne[i], bt + i, cart, rreq + i - 1));
}

template <typename T, DataLoc LOC>
int unpack(int npd, const pbuf<T> *b, const int counts[27], /**/ T *dd) {
    int nm = 0;
    for (int i = 0; i < 27; ++i) {
        int c = counts[i];
        int n = c * npd;
        if (n) cpy(dd + nm * npd, b->dd[i], n, LOC == Host ? H2H : H2D);
        nm += c;
    }
    return nm;
}

/* shift() is not generic */



/* template specializations */

template <> MPI_Datatype MType<int>()      {return MPI_INT;}
template <> MPI_Datatype MType<Particle>() {return datatype::particle;}
template <> MPI_Datatype MType<Solid>()    {return datatype::solid;}

} // gen
} // mdstr
