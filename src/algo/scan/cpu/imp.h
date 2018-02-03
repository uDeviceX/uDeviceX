void scan_work_ini(int, Scan**) { }
void scan_work_fin(Scan*) { }

void scan_apply(const int* input, int size, /**/ int* output, /*w*/ Scan*) {
    int i, s;
    s = 0;
    for (i = 0; i < size; i++) {
        output[i] = s;
        s += input[i];
    }

}
