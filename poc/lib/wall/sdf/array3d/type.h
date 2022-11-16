struct Array3d {
    cudaArray_t a;  // <1>
    size_t x, y, z; // <2>
};
