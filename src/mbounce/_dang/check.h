namespace mbounce {
namespace sub {
namespace dev {

static _HD_ bool valid(float x, int L) {
    bool ok = x >= -3*L/2 && x < 3*L/2;
    if (!ok) printf(": BB : %g not in bounds %d\n", x, L);
    return ok; 
}

static _HD_ void check_pos(const float r[3]) {
    assert(valid(r[X], XS));
    assert(valid(r[Y], YS));
    assert(valid(r[Z], ZS));
}

static _HD_ void check_vel(const float v[3]) {
    assert(valid(v[X], XS / dt));
    assert(valid(v[Y], YS / dt));
    assert(valid(v[Z], ZS / dt));
}

} // dev
} // sub
} // mbounce
