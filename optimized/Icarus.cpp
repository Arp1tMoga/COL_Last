#include <math.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int MATCH   =  2;
constexpr int MISMATCH = -1;
constexpr int GAP      = -2;

constexpr std::array<char, 4> ALPHABET = {'A', 'C', 'G', 'T'};

// Helper for max of multiple values
template<typename T>
inline T max4(T a, T b, T c, T d) {
    return std::max(std::max(a, b), std::max(c, d));
}

template<typename T>
inline T max3(T a, T b, T c) {
    return std::max(std::max(a, b), c);
}

void generate_sequence(char *seq, int n) {
    const char alphabet[] = "ACGT";
    for (int i = 0; i < n; i++)
        seq[i] = alphabet[std::rand() % 4];
    seq[n] = '\0';
}

// -----------------------------------------------------------------------------
// 1. Banded scalar implementation
//    - Only computes cells where |i - j| <= bandwidth
//    - If the true optimal path remains within this band, result == full DP
// -----------------------------------------------------------------------------
int smith_waterman_banded_scalar(const char *seq1, const char *seq2,
                                 int len1, int len2, int bandwidth) {
    if (len1 == 0 || len2 == 0 || bandwidth <= 0) return 0;

    const int band_width = 2 * bandwidth + 1;
    std::vector<int> H_prev(band_width, 0);
    std::vector<int> H_curr(band_width, 0);

    int max_score = 0;

    for (int i = 1; i <= len1; ++i) {
        std::fill(H_curr.begin(), H_curr.end(), 0);

        int j_min = std::max(1, i - bandwidth);
        int j_max = std::min(len2, i + bandwidth);

        for (int j = j_min; j <= j_max; ++j) {
            // Map (i,j) to band index: center (i=j) -> index = bandwidth
            int idx = j - i + bandwidth;
            if (idx < 0 || idx >= band_width) continue;

            int score = (seq1[i - 1] == seq2[j - 1]) ? MATCH : MISMATCH;

            // (i-1, j-1) diag -> index same
            int diag = 0;
            if (idx >= 0 && idx < band_width) {
                diag = H_prev[idx];
            }

            // (i-1, j) up -> index + 1
            int up = 0;
            if (idx + 1 >= 0 && idx + 1 < band_width) {
                up = H_prev[idx + 1];
            }

            // (i, j-1) left -> index - 1
            int left = 0;
            if (idx - 1 >= 0 && idx - 1 < band_width) {
                left = H_curr[idx - 1];
            }

            int v = max4(0,
                         diag + score,
                         up   + GAP,
                         left + GAP);
            H_curr[idx] = v;
            if (v > max_score) max_score = v;
        }

        std::swap(H_prev, H_curr);
    }

    return max_score;
}


} // namespace

// -----------------------------------------------------------------------------
// main: benchmarks various implementations
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
    int N = 32000;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    std::srand(42);

    char *seq1 = (char*)std::malloc((N + 1) * sizeof(char));
    char *seq2 = (char*)std::malloc((N + 1) * sizeof(char));

    generate_sequence(seq1, N);
    generate_sequence(seq2, N);

    std::cout << "=== Smith-Waterman Optimization Benchmark ===\n";
    std::cout << "Sequence length: " << N << "\n\n";

    int bandwidth = std::min(1000,N/10); // 50% of length

    {
        auto start = std::chrono::high_resolution_clock::now();
        int score = smith_waterman_banded_scalar(seq1, seq2, N, N, bandwidth);
        auto end   = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "=== BANDED SCALAR (bandwidth=" << bandwidth << ") ===\n";
        std::cout << "Score: " << score << '\n';
        std::cout << std::fixed << std::setprecision(6)
                  << "Time: " << elapsed << " seconds\n\n";
    }

    std::free(seq1);
    std::free(seq2);
    return 0;
}