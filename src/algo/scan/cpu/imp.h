void alloc_work(int, Work*) { }
void free_work(Work*) { }

void scan(const int* input, int size, /**/ int* output, /*w*/ Work*) {
    int i, s;
    s = 0;
    for (i = 0; i < size; i++) {
        output[i] = s;
        s += input[i];
    }

}
