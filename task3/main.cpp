#include <fstream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <chrono>

struct Grid {
    double L_x, L_y, L_z;
    double H_x, H_y, H_z;
    int N;
    double tau;
    double **u;

    Grid(double L, int N, double T, int K) {
        this->L_x = L;
        this->L_y = L;
        this->L_z = L;
        this->N = N;
        this->H_x = L / N;
        this->H_y = L / N;
        this->H_z = L / N;
        this->tau = T / K;
        u = new double*[3];
        u[0] = new double[(N + 1) * (N + 1) * (N + 1)];
        u[1] = new double[(N + 1) * (N + 1) * (N + 1)];
        u[2] = new double[(N + 1) * (N + 1) * (N + 1)];
    }
    ~Grid() {
        delete[] u[0];
        delete[] u[1];
        delete[] u[2];
        delete[] u;
    }
};

double u_analytical(double x, double y, double z, double t, const Grid &grid) {
    double a_t = M_PI * sqrt(1.0 / (grid.L_x * grid.L_x) + 1.0 / (grid.L_y * grid.L_y) + 1.0 / (grid.L_z * grid.L_z));
    return sin(M_PI * x / grid.L_x) * sin(M_PI * y / grid.L_y) * sin(M_PI * z / grid.L_z) * cos(a_t * t + 2 * M_PI);
}

double phi(double x, double y, double z, const Grid &grid) {
    return u_analytical(x, y, z, 0, grid);
}

int idx(int x, int y, int z, Grid &grid) {
    return (x * (grid.N + 1) + y) * (grid.N + 1) + z;
}


double laplace_operator(double *u_i, int x, int y, int z, Grid &grid) {
    double dx = (u_i[idx(x - 1, y, z, grid)] - 2 * u_i[idx(x, y, z, grid)] + u_i[idx(x + 1, y, z, grid)]) / (grid.H_x * grid.H_x);
    double dy = (u_i[idx(x, y - 1, z, grid)] - 2 * u_i[idx(x, y, z, grid)] + u_i[idx(x, y + 1, z, grid)]) / (grid.H_y * grid.H_y);
    double dz = (u_i[idx(x, y, z - 1, grid)] - 2 * u_i[idx(x, y, z, grid)] + u_i[idx(x, y, z + 1, grid)]) / (grid.H_z * grid.H_z);
    return dx + dy + dz;
}

void fill_borders(int u_idx, Grid &grid) {
    int N = grid.N;
    #pragma omp parallel for
    for (int i = 0; i <= N; i++) {
        #pragma omp parallel for
        for (int j = 0; j <= grid.N; j++) {
            grid.u[u_idx][idx(0, i, j, grid)] = 0;
            grid.u[u_idx][idx(N, i, j, grid)] = 0;
            grid.u[u_idx][idx(i, 0, j, grid)] = 0;
            grid.u[u_idx][idx(i, N, j, grid)] = 0;
            grid.u[u_idx][idx(i, j, 0, grid)] = 0;
            grid.u[u_idx][idx(i, j, N, grid)] = 0;
        }
    }
}

void fill_blocks(Grid &grid) {
    fill_borders(0, grid);
    fill_borders(1, grid);

    #pragma omp parallel for
    for (int x = 1; x < grid.N; x++)
        #pragma omp parallel for
        for (int y = 1; y < grid.N; y++)
            #pragma omp parallel for
            for (int z = 1; z < grid.N; z++)
                grid.u[0][idx(x, y, z, grid)] = phi(x * grid.H_x, y * grid.H_y, z * grid.H_z, grid);

    #pragma omp parallel for
    for (int x = 1; x < grid.N; x++)
        #pragma omp parallel for
        for (int y = 1; y < grid.N; y++)
            #pragma omp parallel for
            for (int z = 1; z < grid.N; z++)
                grid.u[1][idx(x, y, z, grid)] = grid.u[0][idx(x, y, z, grid)] + grid.tau * grid.tau / 2 * laplace_operator(grid.u[0], x, y, z, grid);
}

double eval_error(int u_idx, double t, Grid &grid) {
    double  error = 0;

    #pragma omp parallel for reduction(max: error)
    for (int x = 0; x <= grid.N; x++)
        #pragma omp parallel for
        for (int y = 0; y <= grid.N; y++)
            #pragma omp parallel for
            for (int z = 0; z <= grid.N; z++)
                error = std::max(error, fabs(grid.u[u_idx][idx(x, y, z, grid)] -
                                             u_analytical(x * grid.H_x, y * grid.H_y, z * grid.H_z, t, grid)));
    return error;
}

void save_layer(const double *layer, double tau, const char *filename, Grid &grid) {
    std::ofstream fout(filename);

    fout << "{" << std::endl;
    fout << "    \"Lx\": " << grid.L_x << ", " << std::endl;
    fout << "    \"Ly\": " << grid.L_y << ", " << std::endl;
    fout << "    \"Lz\": " << grid.L_z << ", " << std::endl;
    fout << "    \"N\": " << grid.N << ", " << std::endl;
    fout << "    \"t\": " << tau << ", " << std::endl;
    fout << "    \"u\": [" << std::endl;

    bool wasPrinted = false;

    for (int i = 0; i <= grid.N; i++) {
        for (int j = 0; j <= grid.N; j++) {
            for (int k = 0; k <= grid.N; k++) {
                if (wasPrinted) {
                    fout << ", " << std::endl;
                }
                else {
                    wasPrinted = true;
                }

                fout << "    " << layer[idx(i, j, k, grid)];
            }
        }
    }

    fout << std::endl;
    fout << "    ]" << std::endl;
    fout << "}" << std::endl;

    fout.close();
}

void save_u_analytical(double t, const char *filename, Grid &grid) {
    double *u_copy = new double[(grid.N + 1) * (grid.N + 1) * (grid.N + 1)];
    #pragma omp parallel for
    for (int i = 0; i <= grid.N; i++)
    #pragma omp parallel for
        for (int j = 0; j <= grid.N; j++)
            #pragma omp parallel for
            for (int k = 0; k <= grid.N; k++)
                u_copy[idx(i, j, k, grid)] = u_analytical(i * grid.H_x, j * grid.H_y, k * grid.H_z, t, grid);
    save_layer(u_copy, t, filename, grid);
}

void save_calculated(const double *u, double tau, const char *filename, Grid &grid) {
    double *u_copy = new double[(grid.N + 1) * (grid.N + 1) * (grid.N + 1)];
    #pragma omp parallel for
    for (int i = 0; i <= grid.N; i++)
        #pragma omp parallel for
        for (int j = 0; j <= grid.N; j++)
            #pragma omp parallel for
            for (int k = 0; k <= grid.N; k++)
                u_copy[idx(i, j, k, grid)] = fabs(u[idx(i, j, k, grid)] - u_analytical(i * grid.H_x, j * grid.H_y, k * grid.H_z, tau, grid));

    save_layer(u_copy, tau, filename, grid);
};

double solve_equation(Grid &grid, int steps, bool save) {
    fill_blocks(grid);

    for (int step = 2; step <= steps; step++) {
        #pragma omp parallel for
        for (int x = 1; x < grid.N; x++)
            #pragma omp parallel for
            for (int y = 1; y < grid.N; y++)
                #pragma omp parallel for
                for (int z = 1; z < grid.N; z++)
                    grid.u[step % 3][idx(x, y, z, grid)] = 2 * grid.u[(step + 2) % 3][idx(x, y, z, grid)] -
                                                           grid.u[(step + 1) % 3][idx(x, y, z, grid)] +
                                                     grid.tau * grid.tau * laplace_operator(grid.u[(step + 2) % 3], x, y, z, grid);
        fill_borders(step % 3, grid);
    }
    double error = eval_error(steps % 3, steps * grid.tau, grid);
    if (save) {
        save_layer(grid.u[steps % 3], steps * grid.tau, "numerical.json", grid);
        save_calculated(grid.u[steps % 3], steps * grid.tau, "difference.json", grid);
        save_u_analytical(steps * grid.tau, "analytical.json", grid);
    }
    return error;
}

int main(int argc, char** argv) {
    int steps = 20;
    int K = 1000;
    double T = 1.0;

    bool save = true;
    int N = atoi(argv[1]);
    double L = (argc == 4) ? strtod(argv[2], NULL) : M_PI;
    char *filename = (argc == 4) ? argv[3] : argv[2];

    auto start = std::chrono::high_resolution_clock::now();

    Grid grid = Grid(L, N, T, K);
    double error = solve_equation(grid, steps, save);

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double total_time = time.count() / 1000000.0;

    std::ofstream fout(filename);
    fout << N << " " << 1 << " " << error << " " << total_time << std::endl;
    fout.close();

    return 0;
}