all: task2.cpp
	module load SpectrumMPI;
	mpixlC -O3 -std=c++11 task2.cpp -o task2

submit-polus:
	mpixlC -O3 -std=c++11 task2.cpp -o task2
	module load SpectrumMPI
	for eps in 3e-5 5e-6 1.5e-6 ; do \
		for p in 1 4 16 32 ; do \
			for i in {1..3} ; do \
				mpisubmit.pl -p $$p -w 00:10 --stdout /dev/null --stderr /dev/null ./task2 -- $$eps out\_$$p\_$$eps.txt ; \
			done \
		done \
	done
