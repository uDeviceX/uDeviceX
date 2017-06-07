#generate rbcs

./cph -v r=2.0 -v Lx=42 -v Ly=52 -v Lz=20 | ./cp0  -v s=0.25 > ../../src/rbcs-ic.txt 
