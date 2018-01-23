struct Array3d;
struct Tform;
struct Sdf {
    Array3d   *arr;
    Tex3d     *tex;
    Tform     *t;
    float far_threshold;
};
