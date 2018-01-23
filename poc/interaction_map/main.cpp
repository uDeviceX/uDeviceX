#include <stdio.h>

static const char symbol[] = " +0";    

void print(int n, int x, int y, int z, int pred) {
    printf("%2d  %c  %+d %+d %+d\n", n, symbol[pred], x, y, z);
    //if (pred) printf("%2d  %+d %+d %+d\n", n, x, y, z);
}

bool smaller(int i, int j, int k,
             int i_, int j_, int k_) {
    return
        (i < i_)  ||
        (i == i_ && j < j_) ||
        (i == i_ && j == j_ && k < k_);    
}

int pred(int i, int j, int k) {
    int i_, j_, k_;
    i_ = 2-i;
    j_ = 2-j;
    k_ = 2-k;

    // always keep bulk
    if (i_ == i && j_ == j && k_ == k)
        return 2;

    if (smaller(i, j, k, i_, j_, k_))
        return 1;
    return 0;
}

void walk() {
    int n;
    int i, j, k;
    int x, y, z;
    n = 0;
    for (k = 0; k < 3; ++k) {
        z = k - 1;
        for (j = 0; j < 3; ++j) {
            y = j - 1;
            for (i = 0; i < 3; ++i) {
                x = i - 1;
                print(n++, x, y, z, pred(i, j, k));
            }
        }
    }
}

int main() {
    walk();
    return 0;
}
