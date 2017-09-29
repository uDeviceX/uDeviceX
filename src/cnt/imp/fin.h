void fin() {
    delete g::indexes;
    scan::free_work(&g::ws);
    delete g::entries;
    delete g::rgen;

    Dfree(g::counts);
    Dfree(g::starts);
}
