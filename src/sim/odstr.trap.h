void odstr() {
    int i;
    for(i=0; ; ++i){
        odistr_trap::odstr();
        if (i%100==0) MSG("i=%d", i);
    }
}
