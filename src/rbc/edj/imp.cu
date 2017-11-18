namespace edg {
void set(int md, int f, int x, int y,  /**/ int *hx, int *hy) {
    int j = f*md;
    while (hx[j] != -1) j++;
    hx[j] = x; hy[j] = y;
}

int get(int md, int i, int x, int *hx, int *hy) { /* next */
    i *= md;
    while (hx[i] != x) i++;
    return hy[i];
}
}
