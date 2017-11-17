__global__ void main() {
    using namespace map;

    int dx, dy, dz;
    dx = dy = dz = -1;

    do {
        if (valid(-1, 0, 1, dx, dy, dz))
            printf("%d %d %d\n", dx, dy, dz);
    } while (nxt_xyz0(&dx, &dy, &dz) != END);

}
