void fin() {
    delete indexes;
    scan::free_work(&g::ws);
    delete entries;
    delete g::rgen;

    Dfree(g::counts);
    Dfree(g::starts);
}
