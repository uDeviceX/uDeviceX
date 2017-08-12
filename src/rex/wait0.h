namespace rex {

namespace s {
void wait0() {
    MPI_Status s[26];
    MC(l::m::Waitall(26, reqsendP.data(), s));
}
}

namespace r {
void wait0() {
    int i, count;
    MPI_Status s[26];
    MC(l::m::Waitall(26, reqrecvP.data(), s));
    for (i = 1; i < 26; i++) {
        MPI_Get_count(&s[i], MPI_FLOAT, &count);
        recv_counts[i] = count / (datatype::particle);
    }
}
}

}
