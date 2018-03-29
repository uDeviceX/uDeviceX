template<typename T>
__device__ __forceinline__
// tag::int[]
const T texo_fetch(const Texo<T> t, const int i)
// end::int[]
{
    return Tfetch(T, t.d, i);
}
