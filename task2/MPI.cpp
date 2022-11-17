#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

double F(double x, double y, double z) {
    if ((x <= 1) && (y <= x) && (z >= 0) && (z <= (x * y)))
        return x * y * y * z * z * z;
    return 0;
}

void random_points(double *arr, int arr_len) {
    for (int i = 0; i < arr_len * 3; i += 3) {
        arr[i] = rand() / double(RAND_MAX);
        arr[i + 1] = rand() / double(RAND_MAX);
        arr[i + 2] = rand() / double(RAND_MAX);
    }
}

int main(int argc, char * argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    double eps = strtod(argv[1], NULL);
    char* filename = argv[2];

    //srand(time(NULL));
    int iter_per_step = 1;
    double *points = new double[3 * iter_per_step];   

    double res, err;
    double local_sum = 0;
    double sum = 0;
    double total_sum = 0;
    long total_points = 0;
  
    double start = MPI_Wtime();
    int _continue = 1;
    if (rank == 0) {
        double volume = 1.0;
        double true_res = 1.0 / 364;

        while (_continue) {
            for (int k = 1; k < size ; k++) {
                random_points(points, iter_per_step);
                MPI_Send(points, 3 * iter_per_step, MPI_DOUBLE, k, 1, MPI_COMM_WORLD);
            }
            MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            total_sum += sum;
            total_points += iter_per_step * (size - 1);
            res = volume * total_sum / total_points;
            err = fabs(true_res - res);
            _continue = err >= eps ? 1 : 0;
            MPI_Bcast(&_continue, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int local_sum;
        while (_continue) {
            MPI_Recv(points, 3 * iter_per_step, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            double local_sum = 0;
            for (int i = 0; i < 3 * iter_per_step; i += 3)
                local_sum += F(points[i], points[i + 1], points[i + 2]);         
            MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&_continue, 1, MPI_INT, 0, MPI_COMM_WORLD);

        }
    }
    double end = MPI_Wtime();
    double time = end - start;
    double total_time;
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 

    if (rank == 0) {
        std::ofstream fout(filename);
        fout << res << " " << err << " " << total_points << " " << total_time << std::endl;
        fout.close();
    }
    MPI_Finalize();
    return 0;
}
