import sys
import random
import time

MATCH = 2
MISMATCH = -1
GAP = -2

def generate_sequence(n):
    return ''.join(random.choice('ACGT') for _ in range(n))

def smith_waterman(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)
    H = [[0]*(len2+1) for _ in range(len1+1)]
    max_score = 0

    for i in range(1, len1+1):
        for j in range(1, len2+1):
            match = H[i-1][j-1] + (MATCH if seq1[i-1] == seq2[j-1] else MISMATCH)
            delete = H[i-1][j] + GAP
            insert = H[i][j-1] + GAP
            H[i][j] = max(0, match, delete, insert)
            if H[i][j] > max_score:
                max_score = H[i][j]
    return max_score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sw_baseline.py <sequence_length>")
        sys.exit(1)

    N = int(sys.argv[1])
    random.seed(42)
    seq1 = generate_sequence(N)
    seq2 = generate_sequence(N)

    start = time.time()
    score = smith_waterman(seq1, seq2)
    end = time.time()

    print(f"Sequence length: {N}")
    print(f"Smith-Waterman score: {score}")
    print(f"Execution time: {end - start:.6f} seconds")
