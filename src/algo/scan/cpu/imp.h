void scan_work_ini(int, Work*) { }
void scan_work_fin(Work*) { }

void scan_apply(const int* input, int size, /**/ int* output, /*w*/ Work*) {
    int i, s;
    s = 0;
    for (i = 0; i < size; i++) {
        output[i] = s;
        s += input[i];
    }

}
