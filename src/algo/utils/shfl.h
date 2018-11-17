#define _I_ static __device__

template <typename T>
_I_ T shfl_down(T val, unsigned int width) {
    return __shfl_down(val, width);
}

#undef _I_
