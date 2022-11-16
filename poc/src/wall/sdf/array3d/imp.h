struct Array3d;
void array3d_ini(Array3d**, size_t x, size_t y, size_t z);                // <1>
void array3d_fin(Array3d*);                                               // <2>
void array3d_copy(size_t x, size_t y, size_t z, float *D, /**/ Array3d*); // <3>
