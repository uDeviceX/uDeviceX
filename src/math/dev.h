/* use templates here because we might have mixed float/double
   see intersection routine                                    */

template <typename T1, typename T2, typename T3>
__device__ void diff(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2>
__device__ float dot(const T1 *a, const T2 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
__device__ void cross(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
__device__ void scalmult(const T1 *a, const T2 x, /**/ const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void apxb(const T1 *a, const T2 x, const T3 *b, /**/ T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

