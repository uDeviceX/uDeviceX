void transpose(int n, /*io*/ Force *ff) {
    static_assert(sizeof(Force) == 3*sizeof(float),
                 "sizeof(Force) != 3*sizeof(float)");
    KL(transpose, (28, 1024), (n, (float*)ff));
}
