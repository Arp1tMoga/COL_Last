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
    table['A'] = 0; table['C'] = 1; table['G'] = 2; table['T'] = 3;
    table['a'] = 0; table['c'] = 1; table['g'] = 2; table['t'] = 3;
    return table;
}

constexpr std::array<uint8_t, 256> ENCODE_TABLE = make_encode_table();

inline uint8_t encode_base(char base) {
    return ENCODE_TABLE[static_cast<unsigned char>(base)];
}

void generate_sequence_fast(std::vector<char> &seq, std::size_t n) {
    seq.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        seq[i] = ALPHABET[std::rand() % 4];
    }
}

void encode_sequence_fast(std::vector<uint8_t> &encoded, const std::vector<char> &seq) {
    encoded.resize(seq.size());
    const std::size_t n = seq.size();
    // Unrolled encoding loop
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        encoded[i + 0] = encode_base(seq[i + 0]);
        encoded[i + 1] = encode_base(seq[i + 1]);
        encoded[i + 2] = encode_base(seq[i + 2]);
        encoded[i + 3] = encode_base(seq[i + 3]);
        encoded[i + 4] = encode_base(seq[i + 4]);
        encoded[i + 5] = encode_base(seq[i + 5]);
        encoded[i + 6] = encode_base(seq[i + 6]);
        encoded[i + 7] = encode_base(seq[i + 7]);
    }
    for (; i < n; ++i) {
        encoded[i] = encode_base(seq[i]);
    }
}

using ScoreLookup = std::array<std::vector<int32_t>, 4>;

ScoreLookup build_score_lookup(const std::vector<uint8_t> &seq) {
    ScoreLookup lookup = {
        std::vector<int32_t>(seq.size(), MISMATCH),
        std::vector<int32_t>(seq.size(), MISMATCH),
        std::vector<int32_t>(seq.size(), MISMATCH),
        std::vector<int32_t>(seq.size(), MISMATCH)};

    const std::size_t n = seq.size();
    // Unrolled lookup building
    std::size_t j = 0;
    for (; j + 7 < n; j += 8) {
        lookup[seq[j + 0]][j + 0] = MATCH;
        lookup[seq[j + 1]][j + 1] = MATCH;
        lookup[seq[j + 2]][j + 2] = MATCH;
        lookup[seq[j + 3]][j + 3] = MATCH;
        lookup[seq[j + 4]][j + 4] = MATCH;
        lookup[seq[j + 5]][j + 5] = MATCH;
        lookup[seq[j + 6]][j + 6] = MATCH;
        lookup[seq[j + 7]][j + 7] = MATCH;
    }
    for (; j < n; ++j) {
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
    if (max_extended < 0x80000004) return "Unknown";
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
    if (max_basic < 4) return;
    unsigned eax, ebx, ecx, edx;
    for (unsigned i = 0;; ++i) {
        __cpuid_count(4, i, eax, ebx, ecx, edx);
        unsigned cache_type = eax & 0x1F;
        if (cache_type == 0) break;
        unsigned level = (eax >> 5) & 0x7;
        unsigned ways = ((ebx >> 22) & 0x3FF) + 1;
        unsigned partitions = ((ebx >> 12) & 0x3FF) + 1;
        unsigned line_size = (ebx & 0xFFF) + 1;
        unsigned sets = ecx + 1;
        unsigned size_kb = ways * partitions * line_size * sets / 1024;
        if (level == 1) info.l1_kb = static_cast<int>(size_kb);
        else if (level == 2) info.l2_kb = static_cast<int>(size_kb);
        else if (level == 3) info.l3_kb = static_cast<int>(size_kb);
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

    if (!info.os_supports_avx) info.has_avx = false;

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
CPUInfo query_cpu_info() { return CPUInfo{}; }
#endif

enum class ISA { AVX512, AVX2, SSE41, SCALAR };

ISA pick_best_isa(const CPUInfo &info) {
    if (info.has_avx512f) return ISA::AVX512;
    if (info.has_avx2) return ISA::AVX2;
    if (info.has_sse41) return ISA::SSE41;
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

// Optimized scalar with better register usage
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
        const int32_t * __restrict__ scores_ptr = scores.data();
        const ScoreT * __restrict__ prev_ptr = prev.data();
        ScoreT * __restrict__ curr_ptr = curr.data();
        
        // Unrolled inner loop for better ILP
        int j = 1;
        for (; j + 3 <= len2; j += 4) {
            // Process 4 cells at once
            for (int k = 0; k < 4; ++k) {
                int idx = j + k;
                int diag = static_cast<int>(prev_ptr[idx - 1]) + scores_ptr[idx - 1];
                int up = static_cast<int>(prev_ptr[idx]) + GAP;
                int left = static_cast<int>(curr_ptr[idx - 1]) + GAP;
                int cell = std::max({diag, up, left, 0});
                curr_ptr[idx] = static_cast<ScoreT>(cell);
                max_score = (cell > max_score) ? cell : max_score;
            }
        }
        for (; j <= len2; ++j) {
            int diag = static_cast<int>(prev_ptr[j - 1]) + scores_ptr[j - 1];
            int up = static_cast<int>(prev_ptr[j]) + GAP;
            int left = static_cast<int>(curr_ptr[j - 1]) + GAP;
            int cell = std::max({diag, up, left, 0});
            curr_ptr[j] = static_cast<ScoreT>(cell);
            max_score = (cell > max_score) ? cell : max_score;
        }
        std::swap(prev, curr);
    }

    return max_score;
}

// Striped Smith-Waterman implementation based on SSW library
template <typename Traits>
int smith_waterman_rows_simd_impl(const std::vector<uint8_t> &seq1,
                                  const std::vector<uint8_t> &seq2,
                                  const ScoreLookup &score_lookup) {
    const int readLen = static_cast<int>(seq1.size());
    const int refLen = static_cast<int>(seq2.size());
    if (readLen == 0 || refLen == 0) return 0;

    // Calculate segment length - how many segments we need to cover the read
    const int segLen = (readLen + Traits::LANES - 1) / Traits::LANES;
    
    // Build striped profile: profile[base][segment]
    // Each SIMD element in segment j contains scores for positions: j, j+segLen, j+2*segLen, ...
    // profile[refBase][seg][lane] = score for matching refBase with seq1[readPos]
    std::vector<std::vector<typename Traits::Reg>> profile(4);
    for (int refBase = 0; refBase < 4; ++refBase) {
        profile[refBase].resize(segLen);
        alignas(64) int32_t temp[Traits::LANES];
        
        for (int seg = 0; seg < segLen; ++seg) {
            for (int lane = 0; lane < Traits::LANES; ++lane) {
                int readPos = seg + lane * segLen;
                if (readPos < readLen) {
                    // Score for matching refBase with seq1[readPos]
                    int readBase = seq1[readPos];
                    temp[lane] = (refBase == readBase) ? MATCH : MISMATCH;
                } else {
                    temp[lane] = 0;
                }
            }
            profile[refBase][seg] = Traits::load(temp);
        }
    }
    
    // Allocate H, E, F vectors
    std::vector<typename Traits::Reg> pvHStore(segLen);
    std::vector<typename Traits::Reg> pvHLoad(segLen);
    std::vector<typename Traits::Reg> pvE(segLen);
    
    const auto vZero = Traits::set1(0);
    const int gapOpen = -GAP;   // Positive value
    const int gapExt = -GAP;    // Positive value
    const auto vGapO = Traits::set1(gapOpen);
    const auto vGapE = Traits::set1(gapExt);
    
    // Initialize all to zero
    for (int j = 0; j < segLen; ++j) {
        pvHStore[j] = vZero;
        pvHLoad[j] = vZero;
        pvE[j] = vZero;
    }
    
    int max_score = 0;
    alignas(64) int32_t buffer[Traits::LANES];
    
    // Process each reference position (column)
    for (int i = 0; i < refLen; ++i) {
        const auto &vP = profile[seq2[i]];  // Get profile for current reference base
        
        // Initialize F to zero for this column
        auto vF = vZero;
        auto vMaxColumn = vZero;  // Track max for this column
        
        // Get last H from previous column and shift it (simulates diagonal access)
        auto vH = pvHStore[segLen - 1];
        vH = Traits::shift_left(vH, 4);  // Shift by 4 bytes (1 int32)
        
        // Swap H buffers
        std::swap(pvHLoad, pvHStore);
        
        // Inner loop: process each segment
        for (int j = 0; j < segLen; ++j) {
            // vH starts as H[i-1, j-1] (diagonal)
            // Add match/mismatch score
            vH = Traits::add(vH, vP[j]);
            
            // Load E[i-1, j] (insertion - was computed in previous ref position)
            auto vE = pvE[j];
            
            // H = max(H_diag + score, E, F, 0)
            vH = Traits::max(vH, vE);
            vH = Traits::max(vH, vF);
            vH = Traits::max(vH, vZero);
            
            // Track max for this column
            vMaxColumn = Traits::max(vMaxColumn, vH);
            
            // Store H[i, j]
            pvHStore[j] = vH;
            
            // Update E for next column: E = max(E - gapE, H - gapO)
            // E represents insertion (vertical gap in reference)
            vH = Traits::sub_sat(vH, vGapO);  // H - gapOpen
            vE = Traits::sub_sat(vE, vGapE);  // E - gapExtend
            vE = Traits::max(vE, vH);
            pvE[j] = vE;
            
            // Update F for next segment: F = max(F - gapE, H - gapO)
            // F represents deletion (horizontal gap in read)
            vF = Traits::sub_sat(vF, vGapE);  // F - gapExtend
            vF = Traits::max(vF, vH);
            
            // Load next H for next iteration (becomes diagonal in next segment)
            vH = pvHLoad[j];
        }
        
        // Lazy F loop: propagate F across segments
        // This handles deletions that span multiple segments
        for (int k = 0; k < Traits::LANES; ++k) {
            vF = Traits::shift_left(vF, 4);
            
            bool allDone = true;
            for (int j = 0; j < segLen; ++j) {
                vH = pvHStore[j];
                vH = Traits::max(vH, vF);
                
                // Update max after lazy F correction
                vMaxColumn = Traits::max(vMaxColumn, vH);
                
                pvHStore[j] = vH;
                
                vH = Traits::sub_sat(vH, vGapO);
                vF = Traits::sub_sat(vF, vGapE);
                
                // Check if F > H (more propagation needed)
                auto vTemp = Traits::sub_sat(vF, vH);
                if (!Traits::all_zero(vTemp)) {
                    allDone = false;
                }
            }
            
            if (allDone) break;
        }
        
        // Extract max from this column and update global max
        Traits::store(buffer, vMaxColumn);
        for (int lane = 0; lane < Traits::LANES; ++lane) {
            if (buffer[lane] > max_score) {
                max_score = buffer[lane];
            }
        }
    }

    return max_score;
}

struct TraitsSSE41 {
    using Reg = __m128i;
    static constexpr int LANES = 4;
    static inline Reg load(const int32_t *ptr) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
    }
    static inline void store(int32_t *dst, Reg value) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), value);
    }
    static inline Reg add(Reg a, Reg b) { return _mm_add_epi32(a, b); }
    static inline Reg max(Reg a, Reg b) { return _mm_max_epi32(a, b); }
    static inline Reg set1(int v) { return _mm_set1_epi32(v); }
    static inline Reg sub_sat(Reg a, Reg b) {
        // Saturated subtraction for signed 32-bit integers
        // Convert to signed, subtract, clamp to 0
        auto diff = _mm_sub_epi32(a, b);
        return _mm_max_epi32(diff, _mm_setzero_si128());
    }
    static inline Reg shift_left(Reg a, int bytes) {
        return _mm_slli_si128(a, bytes);
    }
    static inline bool all_zero(Reg a) {
        return _mm_testz_si128(a, a) != 0;
    }
};

struct TraitsAVX2 {
    using Reg = __m256i;
    static constexpr int LANES = 8;
    static inline Reg load(const int32_t *ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }
    static inline void store(int32_t *dst, Reg value) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), value);
    }
    static inline Reg add(Reg a, Reg b) { return _mm256_add_epi32(a, b); }
    static inline Reg max(Reg a, Reg b) { return _mm256_max_epi32(a, b); }
    static inline Reg set1(int v) { return _mm256_set1_epi32(v); }
    static inline Reg sub_sat(Reg a, Reg b) {
        auto diff = _mm256_sub_epi32(a, b);
        return _mm256_max_epi32(diff, _mm256_setzero_si256());
    }
    static inline Reg shift_left(Reg a, int bytes) {
        // Shift entire 256-bit register left by bytes
        // Need to use permute and blend since there's no direct shift across 128-bit lanes
        if (bytes == 0) return a;
        if (bytes >= 32) return _mm256_setzero_si256();
        
        // For 4 bytes (1 int32), shift each 128-bit lane and carry across
        if (bytes == 4) {
            auto shifted = _mm256_slli_si256(a, 4);
            auto perm = _mm256_permute2x128_si256(a, a, 0x08); // Swap lanes
            auto carry = _mm256_srli_si256(perm, 12);
            return _mm256_or_si256(shifted, carry);
        }
        
        // Generic fallback
        alignas(32) int32_t buffer[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), a);
        int shift_elems = bytes / 4;
        for (int i = 7; i >= shift_elems; --i) {
            buffer[i] = buffer[i - shift_elems];
        }
        for (int i = 0; i < shift_elems; ++i) {
            buffer[i] = 0;
        }
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(buffer));
    }
    static inline bool all_zero(Reg a) {
        return _mm256_testz_si256(a, a) != 0;
    }
};

struct TraitsAVX512 {
    using Reg = __m512i;
    static constexpr int LANES = 16;
    TARGET_AVX512_ATTR static inline Reg load(const int32_t *ptr) { 
        return _mm512_loadu_si512(ptr); 
    }
    TARGET_AVX512_ATTR static inline void store(int32_t *dst, Reg value) { 
        _mm512_storeu_si512(dst, value); 
    }
    TARGET_AVX512_ATTR static inline Reg add(Reg a, Reg b) { 
        return _mm512_add_epi32(a, b); 
    }
    TARGET_AVX512_ATTR static inline Reg max(Reg a, Reg b) { 
        return _mm512_max_epi32(a, b); 
    }
    TARGET_AVX512_ATTR static inline Reg set1(int v) { 
        return _mm512_set1_epi32(v); 
    }
    TARGET_AVX512_ATTR static inline Reg sub_sat(Reg a, Reg b) {
        auto diff = _mm512_sub_epi32(a, b);
        return _mm512_max_epi32(diff, _mm512_setzero_si512());
    }
    TARGET_AVX512_ATTR static inline Reg shift_left(Reg a, int bytes) {
        // For AVX512, use alignr across the full register
        if (bytes == 0) return a;
        if (bytes >= 64) return _mm512_setzero_si512();
        
        alignas(64) int32_t buffer[16];
        _mm512_store_si512(buffer, a);
        int shift_elems = bytes / 4;
        for (int i = 15; i >= shift_elems; --i) {
            buffer[i] = buffer[i - shift_elems];
        }
        for (int i = 0; i < shift_elems; ++i) {
            buffer[i] = 0;
        }
        return _mm512_load_si512(buffer);
    }
    TARGET_AVX512_ATTR static inline bool all_zero(Reg a) {
        return _mm512_test_epi32_mask(a, a) == 0;
    }
};

__attribute__((target("default")))
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
    if (seq1.empty() || seq2.empty()) return 0;

    const auto lookup = build_score_lookup(seq2);
    const long long max_possible = static_cast<long long>(MATCH) *
                                   static_cast<long long>(std::min(seq1.size(), seq2.size()));

    if (max_possible > std::numeric_limits<int32_t>::max()) {
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
    std::vector<char> seq1, seq2;
    std::vector<uint8_t> enc1, enc2;
    
    generate_sequence_fast(seq1, N);
    generate_sequence_fast(seq2, N);
    encode_sequence_fast(enc1, seq1);
    encode_sequence_fast(enc2, seq2);

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