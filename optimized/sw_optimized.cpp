#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#if defined(__GNUC__)
#define TARGET_AVX512_ATTR __attribute__((target("avx512f,avx512bw,avx512vl")))
#else
#define TARGET_AVX512_ATTR
#endif

namespace {

constexpr int MATCH = 2;
constexpr int MISMATCH = -1;
constexpr int GAP = -2;

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

using ScoreLookup = std::array<std::vector<int32_t>, 4>;

ScoreLookup build_score_lookup(const std::vector<uint8_t> &seq) {
    ScoreLookup lookup = {
        std::vector<int32_t>(seq.size(), MISMATCH),
        std::vector<int32_t>(seq.size(), MISMATCH),
        std::vector<int32_t>(seq.size(), MISMATCH),
        std::vector<int32_t>(seq.size(), MISMATCH)};

    for (std::size_t j = 0; j < seq.size(); ++j) {
        lookup[seq[j]][j] = MATCH;
    }

    return lookup;
}

struct CPUInfo {
    std::string vendor;
    std::string brand;
    bool has_sse41 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool os_supports_avx = false;
    bool os_supports_avx512 = false;
    int l1_kb = 0;
    int l2_kb = 0;
    int l3_kb = 0;
    unsigned logical_cores = std::max(1u, std::thread::hardware_concurrency());
};

#ifdef __x86_64__
uint64_t read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

std::string read_brand_string(unsigned max_extended) {
    if (max_extended < 0x80000004) {
        return "Unknown";
    }
    std::array<int, 12> brand{};
    unsigned int eax, ebx, ecx, edx;
    for (unsigned leaf = 0; leaf < 3; ++leaf) {
        __get_cpuid(0x80000002 + leaf, &eax, &ebx, &ecx, &edx);
        brand[leaf * 4 + 0] = static_cast<int>(eax);
        brand[leaf * 4 + 1] = static_cast<int>(ebx);
        brand[leaf * 4 + 2] = static_cast<int>(ecx);
        brand[leaf * 4 + 3] = static_cast<int>(edx);
    }
    return std::string(reinterpret_cast<char *>(brand.data()));
}

void populate_cache_sizes(CPUInfo &info, unsigned max_basic) {
    if (max_basic < 4) {
        return;
    }
    unsigned eax, ebx, ecx, edx;
    for (unsigned i = 0;; ++i) {
        __cpuid_count(4, i, eax, ebx, ecx, edx);
        unsigned cache_type = eax & 0x1F;
        if (cache_type == 0) {
            break;
        }
        unsigned level = (eax >> 5) & 0x7;
        unsigned ways = ((ebx >> 22) & 0x3FF) + 1;
        unsigned partitions = ((ebx >> 12) & 0x3FF) + 1;
        unsigned line_size = (ebx & 0xFFF) + 1;
        unsigned sets = ecx + 1;
        unsigned size_kb = ways * partitions * line_size * sets / 1024;
        if (level == 1) {
            info.l1_kb = static_cast<int>(size_kb);
        } else if (level == 2) {
            info.l2_kb = static_cast<int>(size_kb);
        } else if (level == 3) {
            info.l3_kb = static_cast<int>(size_kb);
        }
    }
}

CPUInfo query_cpu_info() {
    CPUInfo info{};
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned max_basic = __get_cpuid_max(0, nullptr);
    if (max_basic >= 0) {
        __get_cpuid(0, &eax, &ebx, &ecx, &edx);
        char vendor_str[13] = {};
        std::memcpy(vendor_str + 0, &ebx, 4);
        std::memcpy(vendor_str + 4, &edx, 4);
        std::memcpy(vendor_str + 8, &ecx, 4);
        info.vendor = vendor_str;
    }

    bool has_osxsave = false;
    if (max_basic >= 1) {
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        has_osxsave = (ecx & bit_OSXSAVE) != 0;
        info.has_sse41 = (ecx & bit_SSE4_1) != 0;
        info.has_avx = (ecx & bit_AVX) != 0;
    }

    if (has_osxsave) {
        const uint64_t xcr0 = read_xcr0();
        info.os_supports_avx = (xcr0 & 0x6) == 0x6;
        info.os_supports_avx512 = (xcr0 & 0xE0) == 0xE0;
    }

    if (!info.os_supports_avx) {
        info.has_avx = false;
    }

    if (max_basic >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        info.has_avx2 = ((ebx & bit_AVX2) != 0) && info.os_supports_avx;
        info.has_avx512f = ((ebx & bit_AVX512F) != 0) && info.os_supports_avx512;
    }

    unsigned max_extended = __get_cpuid_max(0x80000000, nullptr);
    info.brand = read_brand_string(max_extended);

    populate_cache_sizes(info, max_basic);

    return info;
}

#else

CPUInfo query_cpu_info() {
    return CPUInfo{};
}

#endif

enum class ISA { AVX512, AVX2, SSE41, SCALAR };

ISA pick_best_isa(const CPUInfo &info) {
    if (info.has_avx512f) {
        return ISA::AVX512;
    }
    if (info.has_avx2) {
        return ISA::AVX2;
    }
    if (info.has_sse41) {
        return ISA::SSE41;
    }
    return ISA::SCALAR;
}

const char *isa_name(ISA isa) {
    switch (isa) {
    case ISA::AVX512: return "AVX-512";
    case ISA::AVX2: return "AVX2";
    case ISA::SSE41: return "SSE4.1";
    default: return "Scalar";
    }
}

template <typename ScoreT>
int smith_waterman_rows_scalar(const std::vector<uint8_t> &seq1,
                               const std::vector<uint8_t> &seq2,
                               const ScoreLookup &score_lookup) {
    const int len1 = static_cast<int>(seq1.size());
    const int len2 = static_cast<int>(seq2.size());
    std::vector<ScoreT> prev(len2 + 1, 0);
    std::vector<ScoreT> curr(len2 + 1, 0);
    int max_score = 0;

    for (int i = 1; i <= len1; ++i) {
        curr[0] = 0;
        const auto &scores = score_lookup[seq1[i - 1]];
        for (int j = 1; j <= len2; ++j) {
            int diag = static_cast<int>(prev[j - 1]) + scores[j - 1];
            int up = static_cast<int>(prev[j]) + GAP;
            int left = static_cast<int>(curr[j - 1]) + GAP;
            int cell = std::max({diag, up, left, 0});
            curr[j] = static_cast<ScoreT>(cell);
            if (cell > max_score) {
                max_score = cell;
            }
        }
        std::swap(prev, curr);
    }

    return max_score;
}

template <typename Traits>
int smith_waterman_rows_simd_impl(const std::vector<uint8_t> &seq1,
                                  const std::vector<uint8_t> &seq2,
                                  const ScoreLookup &score_lookup) {
    const int len1 = static_cast<int>(seq1.size());
    const int len2 = static_cast<int>(seq2.size());
    if (len1 == 0 || len2 == 0) {
        return 0;
    }

    std::vector<int32_t> prev(len2 + 1, 0);
    std::vector<int32_t> curr(len2 + 1, 0);
    int max_score = 0;
    alignas(64) int32_t buffer[Traits::LANES];
    const int simd_limit = std::max(1, len2 - (Traits::LANES - 1));
    const auto gap_vec = Traits::set1(GAP);

    for (int i = 1; i <= len1; ++i) {
        curr[0] = 0;
        const auto &scores = score_lookup[seq1[i - 1]];
        int j = 1;
        for (; j <= simd_limit; j += Traits::LANES) {
            auto diag = Traits::load(prev.data() + j - 1);
            auto up = Traits::load(prev.data() + j);
            auto sc = Traits::load(scores.data() + j - 1);
            diag = Traits::add(diag, sc);
            up = Traits::add(up, gap_vec);
            auto best = Traits::max(diag, up);
            Traits::store(buffer, best);
            for (int lane = 0; lane < Traits::LANES; ++lane) {
                int idx = j + lane;
                if (idx > len2) {
                    break;
                }
                int left = curr[idx - 1] + GAP;
                int cell = buffer[lane];
                cell = (cell > left) ? cell : left;
                cell = (cell > 0) ? cell : 0;
                curr[idx] = cell;
                if (cell > max_score) {
                    max_score = cell;
                }
            }
        }
        for (; j <= len2; ++j) {
            int diag = prev[j - 1] + scores[j - 1];
            int up = prev[j] + GAP;
            int left = curr[j - 1] + GAP;
            int cell = std::max({diag, up, left, 0});
            curr[j] = cell;
            if (cell > max_score) {
                max_score = cell;
            }
        }
        std::swap(prev, curr);
    }

    return max_score;
}

struct TraitsSSE41 {
    using Reg = __m128i;
    static constexpr int LANES = 4;
    static Reg load(const int32_t *ptr) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
    }
    static void store(int32_t *dst, Reg value) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), value);
    }
    static Reg add(Reg a, Reg b) { return _mm_add_epi32(a, b); }
    static Reg max(Reg a, Reg b) { return _mm_max_epi32(a, b); }
    static Reg set1(int v) { return _mm_set1_epi32(v); }
};

struct TraitsAVX2 {
    using Reg = __m256i;
    static constexpr int LANES = 8;
    static Reg load(const int32_t *ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }
    static void store(int32_t *dst, Reg value) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), value);
    }
    static Reg add(Reg a, Reg b) { return _mm256_add_epi32(a, b); }
    static Reg max(Reg a, Reg b) { return _mm256_max_epi32(a, b); }
    static Reg set1(int v) { return _mm256_set1_epi32(v); }
};

struct TraitsAVX512 {
    using Reg = __m512i;
    static constexpr int LANES = 16;
    TARGET_AVX512_ATTR static Reg load(const int32_t *ptr) { return _mm512_loadu_si512(ptr); }
    TARGET_AVX512_ATTR static void store(int32_t *dst, Reg value) { _mm512_storeu_si512(dst, value); }
    TARGET_AVX512_ATTR static Reg add(Reg a, Reg b) { return _mm512_add_epi32(a, b); }
    TARGET_AVX512_ATTR static Reg max(Reg a, Reg b) { return _mm512_max_epi32(a, b); }
    TARGET_AVX512_ATTR static Reg set1(int v) { return _mm512_set1_epi32(v); }
};

int smith_waterman_rows_scalar_kernel(const std::vector<uint8_t> &seq1,
                                      const std::vector<uint8_t> &seq2,
                                      const ScoreLookup &lookup) {
    return smith_waterman_rows_scalar<int32_t>(seq1, seq2, lookup);
}

__attribute__((target("sse4.1")))
int smith_waterman_rows_sse41_kernel(const std::vector<uint8_t> &seq1,
                                     const std::vector<uint8_t> &seq2,
                                     const ScoreLookup &lookup) {
    return smith_waterman_rows_simd_impl<TraitsSSE41>(seq1, seq2, lookup);
}

__attribute__((target("avx2,fma")))
int smith_waterman_rows_avx2_kernel(const std::vector<uint8_t> &seq1,
                                    const std::vector<uint8_t> &seq2,
                                    const ScoreLookup &lookup) {
    return smith_waterman_rows_simd_impl<TraitsAVX2>(seq1, seq2, lookup);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
int smith_waterman_rows_avx512_kernel(const std::vector<uint8_t> &seq1,
                                      const std::vector<uint8_t> &seq2,
                                      const ScoreLookup &lookup) {
    return smith_waterman_rows_simd_impl<TraitsAVX512>(seq1, seq2, lookup);
}

int smith_waterman_wavefront_parallel(const std::vector<uint8_t> &seq1,
                                      const std::vector<uint8_t> &seq2,
                                      const ScoreLookup &lookup,
                                      unsigned thread_hint) {
    const int L1 = static_cast<int>(seq1.size());
    const int L2 = static_cast<int>(seq2.size());
    std::vector<int32_t> H((L1 + 1) * (L2 + 1), 0);
    std::atomic<int> global_max{0};
    const unsigned workers = std::max(1u, thread_hint);
    auto idx = [&](int i, int j) { return i * (L2 + 1) + j; };

    for (int k = 2; k <= L1 + L2; ++k) {
        const int i_start = std::max(1, k - L2);
        const int i_end = std::min(L1, k - 1);
        if (i_start > i_end) {
            continue;
        }
        const int span = i_end - i_start + 1;
        const int chunk = std::max(1, span / static_cast<int>(workers));
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) {
            const int local_begin = i_start + static_cast<int>(t) * chunk;
            int local_end = local_begin + chunk - 1;
            if (t == workers - 1) {
                local_end = i_end;
            }
            if (local_begin > i_end) {
                break;
            }
            threads.emplace_back([&, local_begin, local_end]() {
                int local_max = 0;
                for (int i = local_begin; i <= std::min(local_end, i_end); ++i) {
                    int j = k - i;
                    int diag = H[idx(i - 1, j - 1)] + lookup[seq1[i - 1]][j - 1];
                    int up = H[idx(i - 1, j)] + GAP;
                    int left = H[idx(i, j - 1)] + GAP;
                    int cell = std::max({diag, up, left, 0});
                    H[idx(i, j)] = cell;
                    if (cell > local_max) {
                        local_max = cell;
                    }
                }
                int prev = global_max.load(std::memory_order_relaxed);
                while (local_max > prev && !global_max.compare_exchange_weak(prev, local_max)) {}
            });
        }
        for (auto &th : threads) {
            th.join();
        }
    }

    return global_max.load();
}

int dispatch_row_kernel(const std::vector<uint8_t> &seq1,
                        const std::vector<uint8_t> &seq2,
                        const ScoreLookup &lookup,
                        ISA isa) {
    switch (isa) {
    case ISA::AVX512:
        return smith_waterman_rows_avx512_kernel(seq1, seq2, lookup);
    case ISA::AVX2:
        return smith_waterman_rows_avx2_kernel(seq1, seq2, lookup);
    case ISA::SSE41:
        return smith_waterman_rows_sse41_kernel(seq1, seq2, lookup);
    default:
        return smith_waterman_rows_scalar_kernel(seq1, seq2, lookup);
    }
}

struct ExecutionContext {
    CPUInfo cpu;
    ISA isa;
    std::string kernel_name;
};

ExecutionContext prepare_execution_context(const CPUInfo &info) {
    ExecutionContext ctx{info, pick_best_isa(info), isa_name(pick_best_isa(info))};
    return ctx;
}

int smith_waterman_optimized(const std::vector<uint8_t> &seq1,
                             const std::vector<uint8_t> &seq2,
                             const ExecutionContext &ctx) {
    if (seq1.empty() || seq2.empty()) {
        return 0;
    }

    const auto lookup = build_score_lookup(seq2);
    const long long max_possible = static_cast<long long>(MATCH) *
                                   static_cast<long long>(std::min(seq1.size(), seq2.size()));
    const std::size_t len1 = seq1.size();
    const std::size_t len2 = seq2.size();
    const unsigned long long cells = static_cast<unsigned long long>(len1 + 1) *
                                     static_cast<unsigned long long>(len2 + 1);
    constexpr unsigned long long max_cells_for_full = 150'000'000ULL; // ~600 MB @ int32

    if (cells <= max_cells_for_full && ctx.cpu.logical_cores > 1) {
        return smith_waterman_wavefront_parallel(seq1, seq2, lookup, ctx.cpu.logical_cores);
    }

    if (max_possible > std::numeric_limits<int32_t>::max()) {
        // extremely long sequences: fall back to int64 scalar implementation
        ScoreLookup wide_lookup = lookup;
        return smith_waterman_rows_scalar<int64_t>(seq1, seq2, wide_lookup);
    }

    return dispatch_row_kernel(seq1, seq2, lookup, ctx.isa);
}

void print_cpu_summary(const ExecutionContext &ctx) {
    std::cout << "CPU Vendor: " << ctx.cpu.vendor << '\n';
    if (!ctx.cpu.brand.empty()) {
        std::cout << "CPU Model : " << ctx.cpu.brand << '\n';
    }
    std::cout << "Cores     : " << ctx.cpu.logical_cores << '\n';
    std::cout << "Caches    : L1 " << ctx.cpu.l1_kb << " KB, L2 " << ctx.cpu.l2_kb
              << " KB, L3 " << ctx.cpu.l3_kb << " KB\n";
    std::cout << "ISA path  : " << ctx.kernel_name << "\n";
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

    CPUInfo cpu_info = query_cpu_info();
    ExecutionContext ctx = prepare_execution_context(cpu_info);
    print_cpu_summary(ctx);

    const auto start = std::chrono::high_resolution_clock::now();
    const int score = smith_waterman_optimized(enc1, enc2, ctx);
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Sequence length: " << N << '\n';
    std::cout << "Smith-Waterman score: " << score << '\n';
    std::cout << std::fixed << std::setprecision(6)
              << "Execution time: " << elapsed << " seconds\n";

    return 0;
}
