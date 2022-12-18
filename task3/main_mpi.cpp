#include <fstream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <stdexcept>

enum Axis {
    X, Y, Z,
};

struct Block {
    int x_min, x_max;
    int y_min, y_max;
    int z_min, z_max;
    int x_size, y_size, z_size, size;

    Block(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max) : x_min(x_min), x_max(x_max), y_min(y_min), y_max(y_max), z_min(z_min), z_max(z_max) {
        x_size = x_max - x_min + 1;
        y_size = y_max - y_min + 1;
        z_size = z_max - z_min + 1;
        size = x_size * y_size * z_size;
    }
};

struct Grid {
    double L_x, L_y, L_z;
    double H_x, H_y, H_z;
    int N;
    double tau;
    int rank, size;
    std::vector< std::vector<double> > u;
    std::vector< std::pair<int, Block> > send_blocks;
    std::vector< std::pair<int, Block> > recv_blocks;

    Grid(double L, int N, double T, int K, int rank, int size) {
        this->L_x = L;
        this->L_y = L;
        this->L_z = L;
        this->N = N;
        this->H_x = L / N;
        this->H_y = L / N;
        this->H_z = L / N;
        this->tau = T / K;
        this->rank = rank;
        this->size = size;
    }
};

void split_block(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, int size, Axis axis, std::vector<Block> &blocks) {
    if (size == 1) {
        blocks.emplace_back(x_min, x_max, y_min, y_max, z_min, z_max);
        return;
    }

    if (size % 2 == 1) {
        if (axis == X) {
            int x = x_min + (x_max - x_min) / size;
            blocks.emplace_back(x_min, x, y_min, y_max, z_min, z_max);
            x_min = x + 1;
            axis = Y;
        } else if (axis == Y) {
            int y = y_min + (y_max - y_min) / size;
            blocks.emplace_back(x_min, x_max, y_min, y, z_min, z_max);
            y_min = y + 1;
            axis = Z;
        } else if (axis == Z) {
            int z = z_min + (z_max - z_min) / size;
            blocks.emplace_back(x_min, x_max, y_min, y_max, z_min, z);
            z_min = z + 1;
            axis = X;
        }
        size--;
    }

    if (axis == X) {
        int x = (x_min + x_max) / 2;
        split_block(x_min, x, y_min, y_max, z_min, z_max, size / 2, Y, blocks);
        split_block(x + 1, x_max, y_min, y_max, z_min, z_max, size / 2, Y, blocks);
    }
    else if (axis == Y) {
        int y = (y_min + y_max) / 2;
        split_block(x_min, x_max, y_min, y, z_min, z_max, size / 2, Z, blocks);
        split_block(x_min, x_max, y + 1, y_max, z_min, z_max, size / 2, Z, blocks);
    }
    else {
        int z = (z_min + z_max) / 2;
        split_block(x_min, x_max, y_min, y_max, z_min, z, size / 2, X, blocks);
        split_block(x_min, x_max, y_min, y_max, z + 1, z_max, size / 2, X, blocks);
    }
}

bool inside(int x_min1, int x_max1, int y_min1, int y_max1, int x_min2, int x_max2, int y_min2, int y_max2) {
    return x_min2 <= x_min1 && x_max1 <= x_max2 && y_min2 <= y_min1 && y_max1 <= y_max2;
}

void transfer_border_blocks(Grid &grid, const std::vector<Block> &blocks) {
    Block block = blocks[grid.rank];

    for (int i = 0; i < grid.size; i++) {
        if (i == grid.rank)
            continue;

        Block block2 = blocks[i];
        if (block.x_min == block2.x_max + 1 or block2.x_min == block.x_max + 1) {
            int x_send = block.x_min == block2.x_max + 1 ? block.x_min : block.x_max;
            int x_recv = block2.x_min == block.x_max + 1 ? block2.x_min : block2.x_max;
            int y_min, y_max, z_min, z_max;
            if (inside(block.y_min, block.y_max, block.z_min, block.z_max,
                       block2.y_min, block2.y_max, block2.z_min, block2.z_max)) {
                y_min = block.y_min; y_max = block.y_max; z_min = block.z_min; z_max = block.z_max;
            } else if (inside(block2.y_min, block2.y_max, block2.z_min, block2.z_max,
                              block.y_min, block.y_max, block.z_min, block.z_max)) {
                y_min = block2.y_min; y_max = block2.y_max; z_min = block2.z_min; z_max = block2.z_max;
            } else
                continue;

            grid.send_blocks.emplace_back(i, Block(x_send, x_send, y_min, y_max, z_min, z_max));
            grid.recv_blocks.emplace_back(i, Block(x_recv, x_recv, y_min, y_max, z_min, z_max));
            continue;
        }

        if (block.y_min == block2.y_max + 1 or block2.y_min == block.y_max + 1) {
            int y_send = block.y_min == block2.y_max + 1 ? block.y_min : block.y_max;
            int y_recv = block2.y_min == block.y_max + 1 ? block2.y_min : block2.y_max;
            int x_min, x_max, z_min, z_max;
            if (inside(block.x_min, block.x_max, block.z_min, block.z_max,
                       block2.x_min, block2.x_max, block2.z_min, block2.z_max)) {
                x_min = block.x_min; x_max = block.x_max; z_min = block.z_min; z_max = block.z_max;
            } else if (inside(block2.x_min, block2.x_max, block2.z_min, block2.z_max,
                              block.x_min, block.x_max, block.z_min, block.z_max)) {
                x_min = block2.x_min; x_max = block2.x_max; z_min = block2.z_min; z_max = block2.z_max;
            } else
                continue;

            grid.send_blocks.emplace_back(i, Block(x_min, x_max, y_send, y_send, z_min, z_max));
            grid.recv_blocks.emplace_back(i, Block(x_min, x_max, y_recv, y_recv, z_min, z_max));
            continue;
        }

        if (block.z_min == block2.z_max + 1 or block2.z_min == block.z_max + 1) {
            int z_send = block.z_min == block2.z_max + 1 ? block.z_min : block.z_max;
            int z_recv = block2.z_min == block.z_max + 1 ? block2.z_min : block2.z_max;
            int x_min, x_max, y_min, y_max;
            if (inside(block.x_min, block.x_max, block.y_min, block.y_max,
                       block2.x_min, block2.x_max, block2.y_min, block2.y_max)) {
                x_min = block.x_min; x_max = block.x_max; y_min = block.y_min; y_max = block.y_max;
            } else if (inside(block2.x_min, block2.x_max, block2.y_min, block2.y_max,
                              block.x_min, block.x_max, block.y_min, block.y_max)) {
                x_min = block2.x_min; x_max = block2.x_max; y_min = block2.y_min; y_max = block2.y_max;
            } else
                continue;

            grid.send_blocks.emplace_back(i, Block(x_min, x_max, y_min, y_max, z_send, z_send));
            grid.recv_blocks.emplace_back(i, Block(x_min, x_max, y_min, y_max, z_recv, z_recv));
            continue;
        }
    }
}

double u_analytical(double x, double y, double z, double t, const Grid &grid) {
    double a_t = M_PI * sqrt(1.0 / (grid.L_x * grid.L_x) + 1.0 / (grid.L_y * grid.L_y) + 1.0 / (grid.L_z * grid.L_z));
    return sin(M_PI * x / grid.L_x) * sin(M_PI * y / grid.L_y) * sin(M_PI * z / grid.L_z) * cos(a_t * t + 2 * M_PI);
}

double phi(double x, double y, double z, const Grid &grid) {
    return u_analytical(x, y, z, 0, grid);
}

int local_index(int x, int y, int z, const Block block) {
    return (x - block.x_min) * block.y_size * block.z_size + (y - block.y_min) * block.z_size + (z - block.z_min);
}

double find_u(int u_idx, int x, int y, int z, const std::vector< std::vector<double> > &recieved, Grid &grid, const Block block) {

    if (block.x_min <= x and x <= block.x_max and block.y_min <= y and y <= block.y_max and block.z_min <= z and z <= block.z_max)
        return grid.u[u_idx][local_index(x, y, z, block)];

    for (int r_i = 0; r_i < grid.recv_blocks.size(); r_i++) {
        Block block2 = grid.recv_blocks[r_i].second;
        if (x < block2.x_min or x > block2.x_max or
            y < block2.y_min or y > block2.y_max or
            z < block2.z_min or z > block2.z_max)
            continue;

        return recieved[r_i][local_index(x, y, z, block2)];
    }
    throw std::runtime_error("Cannot find u!");
}

double laplace_operator(int u_idx, int x, int y, int z, const std::vector< std::vector<double>> &recv, Grid &grid, const Block block) {
    double dx = (find_u(u_idx, x, y - 1, z, recv, grid, block) - 2 * grid.u[u_idx][local_index(x, y, z, block)] +
                 find_u(u_idx, x, y + 1, z, recv, grid, block)) / (grid.H_x * grid.H_x);
    double dy = (find_u(u_idx, x - 1, y, z, recv, grid, block) - 2 * grid.u[u_idx][local_index(x, y, z, block)] +
                 find_u(u_idx, x + 1, y, z, recv, grid, block)) / (grid.H_y * grid.H_y);
    double dz = (find_u(u_idx, x, y, z - 1, recv, grid, block) - 2 * grid.u[u_idx][local_index(x, y, z, block)] +
                 find_u(u_idx, x, y, z + 1, recv, grid, block)) / (grid.H_z * grid.H_z);
    return dx + dy + dz;
}

void fill_borders(int u_idx, Grid &grid, const Block block) {
    int N = grid.N;

    if (block.x_min == 0)
        #pragma omp parallel for
        for (int y = block.y_min; y <= block.y_max; y++)
            #pragma omp parallel for
            for (int z = block.z_min; z <= block.z_max; z++)
                grid.u[u_idx][local_index(block.x_min, y, z, block)] = 0;

    if (block.x_max == N)
        #pragma omp parallel for
        for (int y = block.y_min; y <= block.y_max; y++)
            #pragma omp parallel for
            for (int z = block.z_min; z <= block.z_max; z++)
                grid.u[u_idx][local_index(block.x_max, y, z, block)] = 0;

    if (block.y_min == 0)
        #pragma omp parallel for
        for (int x = block.x_min; x <= block.x_max; x++)
            #pragma omp parallel for
            for (int z = block.z_min; z <= block.z_max; z++)
                grid.u[u_idx][local_index(x, block.y_min, z, block)] = 0;

    if (block.y_max == N)
        #pragma omp parallel for
        for (int x = block.x_min; x <= block.x_max; x++)
            #pragma omp parallel for
            for (int z = block.z_min; z <= block.z_max; z++)
                grid.u[u_idx][local_index(x, block.y_max, z, block)] = 0;

    if (block.z_min == 0)
        #pragma omp parallel for
        for (int x = block.x_min; x <= block.x_max; x++)
            #pragma omp parallel for
            for (int y = block.y_min; y <= block.y_max; y++)
                grid.u[u_idx][local_index(x, y, block.z_min, block)] = 0;

    if (block.z_max == N)
        #pragma omp parallel for
        for (int x = block.x_min; x <= block.x_max; x++)
            #pragma omp parallel for
            for (int y = block.y_min; y <= block.y_max; y++)
                grid.u[u_idx][local_index(x, y, block.z_max, block)] = 0;
}

std::vector<double> to_send(int u_idx, const Block block, const Block block2, Grid &grid) {
    std::vector<double> send(block2.size);

    #pragma omp parallel for
    for (int x = block2.x_min; x <= block2.x_max; x++)
        #pragma omp parallel for
        for (int y = block2.y_min; y <= block2.y_max; y++)
            #pragma omp parallel for
            for (int z = block2.z_min; z <= block2.z_max; z++)
                send[local_index(x, y, z, block2)] = grid.u[u_idx][local_index(x, y, z, block)];
    return send;
}

std::vector< std::vector<double> > send_recv(int u_idx, Grid &grid, const Block block) {
    std::vector< std::vector<double> > recv(grid.recv_blocks.size());
    std::vector<MPI_Request> request(2);
    std::vector<MPI_Status> status(2);

    for (int i = 0; i < grid.recv_blocks.size(); i++) {
        std::vector<double> send = to_send(u_idx, block, grid.send_blocks[i].second, grid);
        recv[i] = std::vector<double>(grid.recv_blocks[i].second.size);
        MPI_Isend(send.data(), grid.send_blocks[i].second.size, MPI_DOUBLE, grid.send_blocks[i].first, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(recv[i].data(), grid.recv_blocks[i].second.size, MPI_DOUBLE, grid.recv_blocks[i].first, 0, MPI_COMM_WORLD, &request[1]);
        MPI_Waitall(2, request.data(), status.data());
    }
    return recv;
}

void fill_blocks(Grid &grid, const Block block) {
    fill_borders(0, grid, block);
    fill_borders(1, grid, block);

    int N = grid.N;
    int x_min = std::max(block.x_min, 1);
    int x_max = std::min(block.x_max, N - 1);
    int y_min = std::max(block.y_min, 1);
    int y_max = std::min(block.y_max, N - 1);
    int z_min = std::max(block.z_min, 1);
    int z_max = std::min(block.z_max, N - 1);

    #pragma omp parallel for
    for (int x = x_min; x <= x_max; x++)
        #pragma omp parallel for
        for (int y = y_min; y <= y_max; y++)
            #pragma omp parallel for
            for (int z = z_min; z <= z_max; z++)
                grid.u[0][local_index(x, y, z, block)] = phi(x * grid.H_x, y * grid.H_y, z * grid.H_z, grid);

    std::vector< std::vector<double> > recv = send_recv(0, grid, block);

    #pragma omp parallel for
    for (int x = x_min; x <= x_max; x++)
        #pragma omp parallel for
        for (int y = y_min; y <= y_max; y++)
            #pragma omp parallel for
            for (int z = z_min; z <= z_max; z++)
                grid.u[1][local_index(x, y, z, block)] = grid.u[0][local_index(x, y, z, block)] +
                                                         grid.tau * grid.tau / 2 * laplace_operator(0, x, y, z, recv, grid, block);
}

void calculate_u(int step, Grid &grid, const Block block) {
    int N = grid.N;
    int x_min = std::max(block.x_min, 1); int x_max = std::min(block.x_max, N - 1);
    int y_min = std::max(block.y_min, 1); int y_max = std::min(block.y_max, N - 1);
    int z_min = std::max(block.z_min, 1); int z_max = std::min(block.z_max, N - 1);

    std::vector< std::vector<double> > recv = send_recv((step + 2) % 3, grid, block);

    #pragma omp parallel for
    for (int x = x_min; x <= x_max; x++)
        #pragma omp parallel for
        for (int y = y_min; y <= y_max; y++)
            #pragma omp parallel for
            for (int z = z_min; z <= z_max; z++)
                grid.u[step % 3][local_index(x, y, z, block)] = 2 * grid.u[(step + 2) % 3][local_index(x, y, z, block)] -
                                                                grid.u[(step + 1) % 3][local_index(x, y, z, block)] +
                                                                grid.tau * grid.tau * laplace_operator((step + 2) % 3, x, y, z, recv, grid, block);
    fill_borders(step % 3, grid, block);
}

double eval_error(int u_idx, double t, Grid &grid, const Block block) {
    double localError = 0, error = 0;

    #pragma omp parallel for
    for (int x = block.x_min; x <= block.x_max; x++)
        #pragma omp parallel for
        for (int y = block.y_min; y <= block.y_max; y++)
            #pragma omp parallel for
            for (int z = block.z_min; z <= block.z_max; z++)
                #pragma omp critical
                    localError = std::max(localError, fabs(grid.u[u_idx][local_index(x, y, z, block)] -
                                                      u_analytical(x * grid.H_x, y * grid.H_y,z * grid.H_z, t, grid)));
    MPI_Reduce(&localError, &error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return error;
}

double solve_equation(Grid &grid, int steps) {
    std::vector<Block> blocks;
    split_block(0, grid.N, 0, grid.N, 0, grid.N, grid.size, X, blocks);
    Block block = blocks[grid.rank];

    grid.u.resize(3);
    for (int i = 0; i < 3; i++)
        grid.u[i].resize(block.size);

    transfer_border_blocks(grid, blocks);
    fill_blocks(grid, block);

    for (int step = 2; step <= steps; step++)
        calculate_u(step, grid, block);

    return eval_error(steps % 3, steps * grid.tau, grid, block);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int steps = 20;
    int K = 1000;
    double T = 1.0;

    int N = atoi(argv[1]);
    double L = (argc == 4) ? strtod(argv[2], NULL) : M_PI;
    char *filename = (argc == 4) ? argv[3] : argv[2];

    double start = MPI_Wtime();

    Grid grid = Grid(L, N, T, K, rank, size);
    double error = solve_equation(grid, steps);

    double end = MPI_Wtime();
    double time = end - start;
    double total_time;
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream fout(filename);
        fout << N << " " << size << " " << error << " " << total_time << std::endl;
        fout.close();
    }

    MPI_Finalize();
    return 0;
}