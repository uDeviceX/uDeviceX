template<typename T>
__device__ __forceinline__
const T texo_fetch(const Texo<T> t, const int i) {
    return Tfetch(T, t.d, i);
}
