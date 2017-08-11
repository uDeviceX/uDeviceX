all: main0 main1 main2 main3 main4 main5 main6 main7

cc=mpicc
main0: main.c; $(cc) -DN=0 $< -o $@
main1: main.c; $(cc) -DN=1 $< -o $@
main2: main.c; $(cc) -DN=2 $< -o $@
main3: main.c; $(cc) -DN=3 $< -o $@
main4: main.c; $(cc) -DN=4 $< -o $@
main5: main.c; $(cc) -DN=5 $< -o $@
main6: main.c; $(cc) -DN=6 $< -o $@
main7: main.c; $(cc) -DN=7 $< -o $@

run: all
	mpiexec \
	-np 1 ./main0 : -np 1 ./main1 : \
	-np 1 ./main2 : -np 1 ./main3 : \
	-np 1 ./main4 : -np 1 ./main5 : \
	-np 1 ./main6 : -np 1 ./main7

clean:; rm -f main[0-7]
