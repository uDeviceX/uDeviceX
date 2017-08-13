void ini() { }

int Malloc(void **p, size_t size) {
    *p = malloc(size);
    return 0;
}
