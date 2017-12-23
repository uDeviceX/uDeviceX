struct Tex3d;
struct tex3Dca {
    Tex3d *t;
    cudaTextureObject_t to;
};

/* view */
struct Sdf_v { cudaTextureObject_t to; };
