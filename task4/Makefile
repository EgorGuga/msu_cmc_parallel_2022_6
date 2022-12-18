# module load SpectrumMPI
test:
        /usr/local/cuda/bin/nvcc -O3 -std=c++11 -arch=sm_35 --compiler-bindir /usr/bin/g++ -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm main.cu -o task4
        bsub -n 1 -gpu "num=2" -R "span[ptile=2]" -W 00:10 -o out.txt -e error.txt mpiexec ./task4 256 1 out\_1\_256\_1.txt

cuda:
        /usr/local/cuda/bin/nvcc -O3 -std=c++11 -arch=sm_35 --compiler-bindir /usr/bin/g++ -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm main.cu -o task4
        for N in 256 512 ; do \
                for p in 1 2 4 6 ; do \
                        bsub -n $$p -gpu "num=2" -R "span[ptile=2]" -W 00:10 -o /dev/null -e /dev/null mpiexec ./task4 $$N 1 c_out\_$$p\_$$N\_1.txt ; \
                        bsub -n $$p -gpu "num=2" -R "span[ptile=2]" -W 00:10 -o /dev/null -e /dev/null mpiexec ./task4 $$N c_out\_$$p\_$$N\_pi.txt ; \
                done \
        done

mpi-openmp:
        g++ -O3 -std=c++11 -fopenmp main_mpi.cpp -o task3 -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm
        for N in 256 512 ; do \
                for p in 1 2 4 6 ; do \
                        bsub -n $$p -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=128 mpiexec ./task3 $$N 1 mo_out\_$$p\_$$N\_1.txt ; \
                        bsub -n $$p -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=128 mpiexec ./task3 $$N mo_out\_$$p\_$$N\_pi.txt ; \
                done \
        done

