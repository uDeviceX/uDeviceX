int get_id_inter(int a, int b) {
    int i, j;
    i = a > b ? a : b;
    j = a > b ? b : a;
    return j + i*(i+1)/2;
}

int contact_scan(int n, const int cc[], int ss[]) {
    int i, s;
    ss[0] = 0;
    for (i = s = 0; i < n; ++i)
        ss[i + 1] = (s += cc[i]);
    return ss[n];
}
