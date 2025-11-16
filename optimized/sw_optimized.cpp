#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {
constexpr int MATCH = 2;
constexpr int MISMATCH = -1;
constexpr int GAP = -2;

constexpr std::array<int, 16> make_score_table() {
    std::array<int, 16> table{};
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            table[(a << 2) | b] = (a == b) ? MATCH : MISMATCH;
        }
    }
    return table;
}

constexpr std::array<int, 16> SCORE_TABLE = make_score_table();
constexpr std::array<char, 4> ALPHABET = {'A', 'C', 'G', 'T'};
constexpr std::array<uint8_t, 256> make_encode_table() {
    std::array<uint8_t, 256> table{};
    table['A'] = 0;
    table['C'] = 1;
    table['G'] = 2;
    table['T'] = 3;
    table['a'] = 0;
    table['c'] = 1;
    table['g'] = 2;
    table['t'] = 3;
    return table;
}

constexpr std::array<uint8_t, 256> ENCODE_TABLE = make_encode_table();

inline uint8_t encode_base(char base) {
    return ENCODE_TABLE[static_cast<unsigned char>(base)];
}

std::vector<char> generate_sequence(std::size_t n) {
    std::vector<char> seq(n);
    for (std::size_t i = 0; i < n; ++i) {
        seq[i] = ALPHABET[std::rand() % 4];
    }
    return seq;
}

std::vector<uint8_t> encode_sequence(const std::vector<char> &seq) {
    std::vector<uint8_t> encoded(seq.size());
    for (std::size_t i = 0; i < seq.size(); ++i) {
        encoded[i] = encode_base(seq[i]);
    }
    return encoded;
}

inline int match_score(uint8_t a, uint8_t b) {
    return SCORE_TABLE[(a << 2) | b];
}

std::array<std::vector<int16_t>, 4> build_score_lookup(const std::vector<uint8_t> &seq) {
    std::array<std::vector<int16_t>, 4> lookup = {
        std::vector<int16_t>(seq.size(), MISMATCH),
        std::vector<int16_t>(seq.size(), MISMATCH),
        std::vector<int16_t>(seq.size(), MISMATCH),
        std::vector<int16_t>(seq.size(), MISMATCH)};

    for (std::size_t j = 0; j < seq.size(); ++j) {
        lookup[seq[j]][j] = MATCH;
    }

    return lookup;
}

template <typename ScoreT>
int smith_waterman_rows(const std::vector<uint8_t> &seq1,
                        const std::vector<uint8_t> &seq2,
                        const std::array<std::vector<int16_t>, 4> &score_lookup) {
    const int len1 = static_cast<int>(seq1.size());
    const int len2 = static_cast<int>(seq2.size());

    std::vector<ScoreT> prev(len2 + 1, 0);
    std::vector<ScoreT> curr(len2 + 1, 0);

    int max_score = 0;

    for (int i = 1; i <= len1; ++i) {
        curr[0] = 0;
        const auto &scores = score_lookup[seq1[i - 1]];
        for (int j = 1; j <= len2; ++j) {
            int diag = static_cast<int>(prev[j - 1]) + static_cast<int>(scores[j - 1]);
            int up = static_cast<int>(prev[j]) + GAP;
            int left = static_cast<int>(curr[j - 1]) + GAP;

            int cell = diag;
            cell = (cell > up) ? cell : up;
            cell = (cell > left) ? cell : left;
            cell = (cell > 0) ? cell : 0;

            if constexpr (std::is_same_v<ScoreT, int16_t>) {
                if (cell > std::numeric_limits<int16_t>::max()) {
                    cell = std::numeric_limits<int16_t>::max();
                }
            }

            curr[j] = static_cast<ScoreT>(cell);
            max_score = (max_score > cell) ? max_score : cell;
        }
        std::swap(prev, curr);
    }

    return max_score;
}

int smith_waterman_optimized(const std::vector<uint8_t> &seq1,
                             const std::vector<uint8_t> &seq2) {
    if (seq1.empty() || seq2.empty()) {
        return 0;
    }

    const auto score_lookup = build_score_lookup(seq2);
    const long long max_possible = static_cast<long long>(MATCH) *
                                   static_cast<long long>(std::min(seq1.size(), seq2.size()));

    if (max_possible <= std::numeric_limits<int16_t>::max()) {
        return smith_waterman_rows<int16_t>(seq1, seq2, score_lookup);
    }

    return smith_waterman_rows<int32_t>(seq1, seq2, score_lookup);
}

} // namespace

int main(int argc, char **argv) {
    int N = 32000;
    if (argc > 1) {
        N = std::atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Sequence length must be positive.\n";
            return 1;
        }
    }

    std::srand(42);
    auto seq1 = generate_sequence(N);
    auto seq2 = generate_sequence(N);
    auto enc1 = encode_sequence(seq1);
    auto enc2 = encode_sequence(seq2);

    const auto start = std::chrono::high_resolution_clock::now();
    const int score = smith_waterman_optimized(enc1, enc2);
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Sequence length: " << N << '\n';
    std::cout << "Smith-Waterman optimized score: " << score << '\n';
    std::cout << std::fixed << std::setprecision(6)
              << "Execution time: " << elapsed << " seconds\n";

    return 0;
}
