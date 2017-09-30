static void transpose0(int n, /*io*/ Force *ff) {
    assert(sizeof(Force) == 3*sizeof(float));
    KL(transpose, (28, 1024), (n, (float*)ff));
}
