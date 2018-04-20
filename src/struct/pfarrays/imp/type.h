enum {
    MAX_SIZE = 10
};

struct PFarray {
    PaArray p;
    FoArray f;
    long n;
};

struct PFarrays {
    PFarray a[MAX_SIZE];
    int n;
};
