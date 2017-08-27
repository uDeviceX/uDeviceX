namespace k_fsi {
struct Pa { /* local particle */
    float x, y, z;
    float vx, vy, vz;
};

struct Fo { float *x, *y, *z; }; /* force */
}
