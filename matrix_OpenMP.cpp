#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>

using namespace std;


void transpose(const vector<double> &matrix, vector<double> &transposed, size_t rows, size_t cols) {
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
}


void multiply(const vector<double> &A, const vector<double> &B, vector<double> &result, size_t rows, size_t cols, size_t common_dim) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0;
            for (size_t k = 0; k < common_dim; ++k) {
                sum += A[i * common_dim + k] * B[j * common_dim + k];
            }
            result[i * cols + j] = sum;
        }
    }
}


void initialize_matrix(vector<double> &matrix, size_t rows, size_t cols) {
    #pragma omp parallel for
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main() {
    vector<size_t> sizes = {3, 4, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    for (size_t size : sizes) {
        cout << "Matrix size: " << size << "x" << size << endl;


        vector<double> A(size * size);
        vector<double> B(size * size);
        vector<double> B_transposed(size * size);
        vector<double> result(size * size);

        initialize_matrix(A, size, size);
        initialize_matrix(B, size, size);

        transpose(B, B_transposed, size, size);

 
        auto start_multiply = chrono::high_resolution_clock::now();
        multiply(A, B_transposed, result, size, size, size);
        auto end_multiply = chrono::high_resolution_clock::now();
        chrono::duration<double> multiply_time = end_multiply - start_multiply;
        cout << "Multiplication time: " << multiply_time.count() << " seconds" << endl;

        cout << "-----------------------------------" << endl;
    }

    return 0;
}
