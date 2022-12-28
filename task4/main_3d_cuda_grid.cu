#include <fstream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

const int threads = 128;

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

    Grid(double L, int N, double T, int K) {
        this->L_x = L;
        this->L_y = L;
        this->L_z = L;
        this->N = N;
        this->H_x = L / (N - 1);
        this->H_y = L / (N - 1);
        this->H_z = L / (N - 1);
        this->tau = T / K;
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

void transfer_border_blocks(const std::vector<Block> &blocks, thrust::host_vector<Block> &send, thrust::host_vector<Block> &recv,
                            thrust::host_vector<int> &ranks, int rank, int size) {
    Block block = blocks[rank];

    for (int i = 0; i < size; i++) {
        if (i == rank)
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
            send.push_back(Block(x_send, x_send, y_min, y_max, z_min, z_max));
            recv.push_back(Block(x_recv, x_recv, y_min, y_max, z_min, z_max));
            ranks.push_back(i);
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

            send.push_back(Block(x_min, x_max, y_send, y_send, z_min, z_max));
            recv.push_back(Block(x_min, x_max, y_recv, y_recv, z_min, z_max));
            ranks.push_back(i);
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

            send.push_back(Block(x_min, x_max, y_min, y_max, z_send, z_send));
            recv.push_back(Block(x_min, x_max, y_min, y_max, z_recv, z_recv));
            ranks.push_back(i);
            continue;
        }
    }
}

__device__ double u_analytical(double x, double y, double z, double t, double a_t, const Grid &grid) {
    return sin(M_PI * x / grid.L_x) * sin(M_PI * y / grid.L_y) * sin(M_PI * z / grid.L_z) * cos(a_t * t + 2 * M_PI);
}

__device__ double phi(double x, double y, double z, double a_t, const Grid &grid) {
    return u_analytical(x, y, z, 0, a_t, grid);
}

__host__ __device__ int local_index(int x, int y, int z, const Block &block) {
    return (x - block.x_min) * block.y_size * block.z_size + (y - block.y_min) * block.z_size + (z - block.z_min);
}

__device__ double find_u(double *u, int x, int y, int z, const Block &block, double *to_recv, Block *recv, int d_size) {
    int idx = x * block.y_size * block.z_size + y * block.z_size + z;
    if (block.x_min <= x and x <= block.x_max and block.y_min <= y and y <= block.y_max and block.z_min <= z and z <= block.z_max)
        return u[idx];

    int offset = 0;

    for (int r_i = 0; r_i < d_size; r_i++) {
        Block block2 = recv[r_i];
        if (x < block2.x_min or x > block2.x_max or y < block2.y_min or y > block2.y_max or z < block2.z_min or z > block2.z_max) {
            offset += recv[r_i].size;
            continue;
        }
        return to_recv[offset + idx];
    }
    return 1;
}

__device__ double laplace_operator(double *u, int x, int y, int z, int idx, const Block &block, const Grid &grid, double *to_recv,
                                   Block *recv, int d_size) {
    double dx = (find_u(u, x, y - 1, z, block, to_recv, recv, d_size) - 2 * u[idx] + find_u(u, x, y + 1, z, block, to_recv, recv, d_size)) / (grid.H_y * grid.H_y);
    double dy = (find_u(u, x - 1, y, z, block, to_recv, recv, d_size) - 2 * u[idx] + find_u(u, x + 1, y, z, block, to_recv, recv, d_size)) / (grid.H_x * grid.H_x);
    double dz = (find_u(u, x, y, z - 1, block, to_recv, recv, d_size) - 2 * u[idx] + find_u(u, x, y, z + 1, block, to_recv, recv, d_size)) / (grid.H_z * grid.H_z);
    return dx + dy + dz;
}

__global__ void fill_first_kind_borders(double *u, const Block block, Axis axis, int i, int d1, int d2, int d1_size, int d2_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d1_size * d2_size)
        return;

    int i1 = d1 + idx / d2_size;
    int i2 = d2 + idx % d2_size;

    switch (axis) {
        case X:
            u[local_index(i, i1, i2, block)] = 0;
            break;
        case Y:
            u[local_index(i1, i, i2, block)] = 0;
            break;
        case Z:
            u[local_index(i1, i2, i, block)] = 0;
            break;
    }
}

void fill_borders(thrust::device_vector<double> &u_device, const Grid grid, const Block block) {
    if (block.x_min == 0)
        fill_first_kind_borders<<<((block.y_size * block.z_size + threads - 1) / threads), threads>>>(thrust::raw_pointer_cast(&u_device[0]), block, X, 0, block.y_min, block.z_min, block.y_size, block.z_size);

    if (block.x_max == grid.N - 1)
        fill_first_kind_borders<<<((block.y_size * block.z_size + threads - 1) / threads), threads>>>(thrust::raw_pointer_cast(&u_device[0]), block, X, grid.N - 1, block.y_min, block.z_min, block.y_size, block.z_size);

    if (block.y_min == 0)
        fill_first_kind_borders<<<((block.x_size * block.z_size + threads - 1) / threads), threads>>>(thrust::raw_pointer_cast(&u_device[0]), block, Y, 0, block.x_min, block.z_min, block.x_size, block.z_size);

    if (block.y_max == grid.N - 1)
        fill_first_kind_borders<<<((block.x_size * block.z_size + threads - 1) / threads), threads>>>(thrust::raw_pointer_cast(&u_device[0]), block, Y, grid.N - 1, block.x_min, block.z_min, block.x_size, block.z_size);

    if (block.z_min == 0)
        fill_first_kind_borders<<<((block.x_size * block.y_size + threads - 1) / threads), threads>>>(thrust::raw_pointer_cast(&u_device[0]), block, Z, 0, block.x_min, block.y_min, block.x_size, block.y_size);

    if (block.z_max == grid.N - 1)
        fill_first_kind_borders<<<((block.x_size * block.y_size + threads - 1) / threads), threads>>>(thrust::raw_pointer_cast(&u_device[0]), block, Z, grid.N - 1, block.x_min, block.y_min, block.x_size, block.y_size);
}

thrust::host_vector<double> collect_send(thrust::host_vector<double> &u, const Block block, const Block block2) {
    thrust::host_vector<double> to_send(block2.size);

    for (int x = block2.x_min; x <= block2.x_max; x++)
        for (int y = block2.y_min; y <= block2.y_max; y++)
            for (int z = block2.z_min; z <= block2.z_max; z++)
                to_send[local_index(x, y, z, block2)] = u[local_index(x, y, z, block)];
    return to_send;
}

thrust::host_vector<double> send_recv(thrust::host_vector<double> &u, const Block block,thrust::host_vector<Block> &send, thrust::host_vector<Block> &recv,thrust::host_vector<int> &ranks) {
    thrust::host_vector<double> to_recv;
    int offset = 0;
    thrust::host_vector<MPI_Request> request(2);
    thrust::host_vector<MPI_Status> status(2);
    for (int i = 0; i < ranks.size(); i++) {
        thrust::host_vector<double> to_send = collect_send(u, block, send[i]);
        to_recv.insert(to_recv.end(), recv[i].size, 0);
        MPI_Isend(to_send.data(), send[i].size, MPI_DOUBLE, ranks[i], 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(to_recv.data() + offset, recv[i].size, MPI_DOUBLE, ranks[i], 0, MPI_COMM_WORLD, &request[1]);
        MPI_Waitall(2, request.data(), status.data());
        offset += recv[i].size;
    }
    return to_recv;
}

__global__ void calculate_u0(double *u0, double a_t, const Grid grid, const Block block) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if ((x <= 0) or (x >= grid.N - 1) or (y <= 0) or (y >= grid.N - 1) or (z <= 0) or (z >= grid.N - 1))
        return;
    int idx = x * block.y_size * block.z_size + y * block.z_size + z;
    u0[idx] = phi(x * grid.H_x, y * grid.H_y, z * grid.H_z, a_t, grid);
}

__global__ void calculate_u1(double *u0, double *u1, double *to_recv, Block *recv, int d_size,
                             const Grid grid, const Block block) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if ((x <= 0) or (x >= grid.N - 1) or (y <= 0) or (y >= grid.N - 1) or (z <= 0) or (z >= grid.N - 1))
        return;
    int idx = x * block.y_size * block.z_size + y * block.z_size + z;
    u1[idx] = u0[idx] + grid.tau * grid.tau / 2 * laplace_operator(u0, x, y, z, idx, block, grid, to_recv, recv, d_size);
}

void fill_blocks(const Grid grid, const Block block, double a_t, thrust::device_vector<double> &u0_device, thrust::device_vector<double> &u1_device, thrust::host_vector<Block> &send, thrust::host_vector<Block> &recv, thrust::host_vector<int> &ranks) {
    fill_borders(u0_device, grid, block);
    fill_borders(u1_device, grid, block);

    dim3 grid_size(threads, threads, threads);
    dim3 block_size(block.x_size / threads, block.y_size / threads, block.z_size / threads);
    calculate_u0<<<grid_size, block_size>>>(thrust::raw_pointer_cast(&u0_device[0]), a_t, grid, block);

    thrust::host_vector<double> u0(u0_device);
    thrust::host_vector<double> to_recv = send_recv(u0, block, send, recv, ranks);
    thrust::device_vector<double> to_recv_device(to_recv);
    thrust::device_vector<Block> recv_device(recv);

    calculate_u1<<<grid_size, block_size>>>(thrust::raw_pointer_cast(&u0_device[0]),
                                                              thrust::raw_pointer_cast(&u1_device[0]),
                                                              thrust::raw_pointer_cast(&to_recv_device[0]),
                                                              thrust::raw_pointer_cast(&recv_device[0]), recv.size(),
                                                              grid, block);
}

__global__ void calculate_u_kernal(double *u, double *u0, double *u1, double *to_recv, Block *recv, int d_size,
                                   const Grid grid, const Block block) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if ((x <= 0) or (x >= grid.N - 1) or (y <= 0) or (y >= grid.N - 1) or (z <= 0) or (z >= grid.N - 1))
        return;
    int idx = x * block.y_size * block.z_size + y * block.z_size + z;

    u[idx] = 2 * u1[idx] - u0[idx] + grid.tau * grid.tau * laplace_operator(u1, x, y, z, idx, block, grid, to_recv, recv, d_size);
}

void calculate_u(int step, const Grid grid, const Block block, std::vector< thrust::device_vector<double> > &u, thrust::host_vector<Block> &send, thrust::host_vector<Block> &recv, thrust::host_vector<int> &ranks) {
    thrust::host_vector<double> u_device(u[(step + 2) % 3]);
    thrust::host_vector<double> to_recv = send_recv(u_device, block, send, recv, ranks);
    thrust::device_vector<double> to_recv_device(to_recv);
    thrust::device_vector<Block> recv_device(recv);

    dim3 grid_size(threads, threads, threads);
    dim3 block_size(block.x_size / threads, block.y_size / threads, block.z_size / threads);
    calculate_u_kernal<<<grid_size, block_size>>>(thrust::raw_pointer_cast(&u[step % 3][0]), thrust::raw_pointer_cast(&u[(step + 1) % 3][0]), thrust::raw_pointer_cast(&u[(step + 2) % 3][0]), thrust::raw_pointer_cast(&to_recv_device[0]),thrust::raw_pointer_cast(&recv_device[0]), recv.size(), grid, block);
    fill_borders(u[step % 3], grid, block);
}

__global__ void eval_error_kernel(double *u, double tau, double a_t, const Block block, const Grid grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = x * block.y_size * block.z_size + y * block.z_size + z;
    u[idx] = fabs(u[idx] - u_analytical(x * grid.H_x, y * grid.H_y, z * grid.H_z, tau, a_t, grid));
}

double eval_error(thrust::device_vector<double> &u_device, double tau, const Grid grid, const Block block, double a_t) {
    dim3 grid_size(threads, threads, threads);
    dim3 block_size(block.x_size / threads, block.y_size / threads, block.z_size / threads);
    eval_error_kernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(&u_device[0]), tau, a_t, block, grid);
    thrust::device_vector<double>::iterator iter = thrust::max_element(u_device.begin(), u_device.end());
    double local_error = u_device[iter - u_device.begin()];
    double error = 0;
    MPI_Reduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return error;
}

double solve_equation(Grid &grid, int steps) {
    double a_t = M_PI * sqrt(1.0 / (grid.L_x * grid.L_x) + 1.0 / (grid.L_y * grid.L_y) + 1.0 / (grid.L_z * grid.L_z));
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size = 1;
    std::vector<Block> blocks;
    split_block(0, grid.N - 1, 0, grid.N - 1, 0, grid.N - 1, size, X, blocks);
    Block block = blocks[rank];

    std::vector< thrust::device_vector<double> > u(3);
    for (int i = 0; i < 3; i++)
        u[i].resize(block.size);

    thrust::host_vector<Block> send_blocks, recv_blocks;
    thrust::host_vector<int> ranks;
    transfer_border_blocks(blocks, send_blocks, recv_blocks, ranks, rank, size);
    fill_blocks(grid, block, a_t, u[0], u[1], send_blocks, recv_blocks, ranks);

    for (int step = 2; step <= steps; step++)
        calculate_u(step, grid, block, u, send_blocks, recv_blocks, ranks);

    return eval_error(u[steps % 3], steps * grid.tau, grid, block, a_t);
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

    Grid grid = Grid(L, N, T, K);
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