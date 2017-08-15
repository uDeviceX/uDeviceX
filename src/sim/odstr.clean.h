void odstr_clean() {
    using namespace odstr;
    post_recv_pp(/**/ &o::td);
    send_pp(/**/ &o::td);
    recv_pp(/**/ &o::td);
}

void odstr() {
    int i;
    for(i=0; ; ++i){
        odstr_clean();
        if (i%10000==0) MSG("i=%d", i);
    }
}
