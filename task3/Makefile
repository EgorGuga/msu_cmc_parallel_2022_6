# module load SpectrumMPI/10.1.0
test:
        g++ -O3 -std=c++11 -fopenmp main.cpp -o task3
        bsub -n 1 -W 00:10 -o /dev/null -e /dev/null OMP_NUM_THREADS=1 ./task3 128 1 out\_1\_128\_1.txt
        g++ -O3 -std=c++11 -fopenmp main_mpi.cpp -o task3 -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm
        bsub -n 4 -W 00:10 -o /dev/null -e /dev/null OMP_NUM_THREADS=1 mpiexec ./task3 128 1 out\_4\_128\_1.txt

linear:
        g++ -O3 -std=c++11 -fopenmp main.cpp -o task3_linear
        for N in 128 256 512 ; do \
                bsub -n 1 -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=1 ./task3_linear $$N 1 l_out\_1\_$$N\_1.txt ; \
                bsub -n 1 -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=1 ./task3_linear $$N l_out\_1\_$$N\_pi.txt ; \
        done

mpi:
        g++ -O3 -std=c++11 -fopenmp main_mpi.cpp -o task3 -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm
        for N in 128 256 512 ; do \
                for p in 1 4 8 16 32 ; do \
                        bsub -n $$p -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=1 mpiexec ./task3 $$N 1 m_out\_$$p\_$$N\_1.txt ; \
                        bsub -n $$p -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=1 mpiexec ./task3 $$N m_out\_$$p\_$$N\_pi.txt ; \
                done \
        done

linear-openmp:
        g++ -O3 -std=c++11 -fopenmp main.cpp -o task3
        for N in 128 256 512 ; do \
                bsub -n 1 -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=4 ./task3 $$N 1 lo_out\_1\_$$N\_1.txt ; \
                bsub -n 1 -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=4 ./task3 $$N lo_out\_1\_$$N\_pi.txt ; \
        done

mpi-openmp:
        g++ -O3 -std=c++11 -fopenmp main_mpi.cpp -o task3 -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm
        for N in 128 256 512 ; do \
                for p in 1 4 8 16 32 ; do \
                        bsub -n $$p -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=4 mpiexec ./task3 $$N 1 mo_out\_$$p\_$$N\_1.txt ; \
                        bsub -n $$p -W 00:30 -o /dev/null -e /dev/null OMP_NUM_THREADS=4 mpiexec ./task3 $$N mo_out\_$$p\_$$N\_pi.txt ; \