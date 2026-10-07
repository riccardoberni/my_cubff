/*
 * bff_tokens.c
 *
 * One epoch of the cubff "primordial soup" for the bff_noheads language, with
 * tracer tokens: besides the soup itself, an array of 64-bit "origin" ids is
 * carried along, one per byte, recording where each byte was created.
 *
 * The epoch is reproduced exactly as in cubff's common_language.h
 * (RunSimulation / MutateAndRunPrograms) and bff.inc.h (Evaluate), so that for
 * the same seed the soup stays byte-for-byte identical to the engine's:
 *   - pairing:   Fisher-Yates shuffle driven by SplitMix64(seed, epoch, i);
 *   - mutation:  each byte of each pair is replaced by a random byte with
 *                probability mutation_prob / 2^30;
 *   - execution: both heads and the instruction pointer start at 0, at most
 *                max_steps instructions.
 *
 * Token rules (as described in the paper): a byte created at initialization
 * or by a mutation gets a new origin id; the copy instructions '.' and ','
 * copy the origin id together with the byte; '+' and '-' change the byte but
 * keep its origin id.
 *
 * Build:  gcc -O3 -fopenmp -shared -fPIC -o libbff_tokens.so bff_tokens.c
 */

#include <stdint.h>
#include <string.h>

#define TAPE 64
#define PAIR (2 * TAPE)

/* Origin id of a byte created by a mutation: (epoch + 1) in the high bits,
 * soup position (program * 64 + offset) in the low POSITION_BITS bits.
 * Bytes created at initialization have epoch field 0. */
#define POSITION_BITS 32

uint64_t ts_splitmix64(uint64_t seed) {
  uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* seed(x) in RunSimulation. */
static inline uint64_t mix_seed(uint64_t run_seed, uint64_t x) {
  return ts_splitmix64(ts_splitmix64(run_seed) ^ ts_splitmix64(x));
}

/* Pairing of one epoch: s[2k] and s[2k+1] interact, s[2k] first. */
void ts_shuffle(uint32_t *s, uint64_t n, uint64_t run_seed, uint64_t epoch) {
  for (uint64_t i = 0; i < n; i++) s[i] = (uint32_t)i;
  for (uint64_t i = n; i-- > 0;) {
    uint64_t j = ts_splitmix64(mix_seed(run_seed, epoch * n + i)) % (i + 1);
    uint32_t tmp = s[i];
    s[i] = s[j];
    s[j] = tmp;
  }
}

/* Mutation step for the pair number `index`; `tok` may be NULL. */
static void mutate_pair(uint8_t *tape, uint64_t *tok, uint64_t n,
                        uint64_t epoch_seed, uint64_t index,
                        uint32_t mutation_prob, uint64_t epoch, uint32_t p1,
                        uint32_t p2) {
  for (uint64_t i = 0; i < PAIR; i++) {
    uint64_t rng = ts_splitmix64((n * epoch_seed + index) * TAPE * 2 + i);
    uint8_t repl = rng & 0xFF;
    uint64_t prob_rng = (rng >> 8) & ((1ULL << 30) - 1);
    if (prob_rng < mutation_prob) {
      tape[i] = repl;
      if (tok) {
        uint64_t pos = (i < TAPE) ? ((uint64_t)p1 * TAPE + i)
                                  : ((uint64_t)p2 * TAPE + (i - TAPE));
        tok[i] = ((epoch + 1) << POSITION_BITS) | pos;
      }
    }
  }
}

/* Evaluate() of bff.inc.h for bff_noheads, moving origin ids with the bytes.
 * Returns the number of steps taken. */
static uint64_t evaluate(uint8_t *tape, uint64_t *tok, uint64_t max_steps) {
  int pos = 0, head0 = 0, head1 = 0;
  uint64_t i = 0;
  for (; i < max_steps; i++) {
    head0 &= PAIR - 1;
    head1 &= PAIR - 1;
    switch (tape[pos]) {
      case '<': head0--; break;
      case '>': head0++; break;
      case '{': head1--; break;
      case '}': head1++; break;
      case '+': tape[head0]++; break;
      case '-': tape[head0]--; break;
      case '.':
        tape[head1] = tape[head0];
        tok[head1] = tok[head0];
        break;
      case ',':
        tape[head0] = tape[head1];
        tok[head0] = tok[head1];
        break;
      case '[':
        if (!tape[head0]) {
          int depth = 1;
          pos++;
          for (; pos < PAIR && depth > 0; pos++) {
            if (tape[pos] == ']') depth--;
            if (tape[pos] == '[') depth++;
          }
          pos--;
          if (depth != 0) pos = PAIR;
        }
        break;
      case ']':
        if (tape[head0]) {
          int depth = 1;
          pos--;
          for (; pos >= 0 && depth > 0; pos--) {
            if (tape[pos] == ']') depth++;
            if (tape[pos] == '[') depth--;
          }
          pos++;
          if (depth != 0) pos = -1;
        }
        break;
      default:
        break;
    }
    if (pos < 0) { i++; break; }
    pos++;
    if (pos >= PAIR) { i++; break; }
  }
  return i;
}

/* Run one epoch in place. `s` is the pairing produced by ts_shuffle. */
void ts_run_epoch(uint8_t *soup, uint64_t *origin, const uint32_t *s,
                  uint64_t n, uint64_t run_seed, uint64_t epoch,
                  uint32_t mutation_prob, uint64_t max_steps) {
  uint64_t epoch_seed = mix_seed(run_seed, epoch);
  int64_t num_pairs = (int64_t)(n / 2);
  #pragma omp parallel for schedule(dynamic, 256)
  for (int64_t index = 0; index < num_pairs; index++) {
    uint32_t p1 = s[2 * index], p2 = s[2 * index + 1];
    uint8_t tape[PAIR];
    uint64_t tok[PAIR];
    memcpy(tape, soup + (uint64_t)p1 * TAPE, TAPE);
    memcpy(tape + TAPE, soup + (uint64_t)p2 * TAPE, TAPE);
    memcpy(tok, origin + (uint64_t)p1 * TAPE, TAPE * sizeof(uint64_t));
    memcpy(tok + TAPE, origin + (uint64_t)p2 * TAPE, TAPE * sizeof(uint64_t));
    mutate_pair(tape, tok, n, epoch_seed, (uint64_t)index, mutation_prob,
                epoch, p1, p2);
    evaluate(tape, tok, max_steps);
    memcpy(soup + (uint64_t)p1 * TAPE, tape, TAPE);
    memcpy(soup + (uint64_t)p2 * TAPE, tape + TAPE, TAPE);
    memcpy(origin + (uint64_t)p1 * TAPE, tok, TAPE * sizeof(uint64_t));
    memcpy(origin + (uint64_t)p2 * TAPE, tok + TAPE, TAPE * sizeof(uint64_t));
  }
}

/* The 128-byte tape of the pair containing program `prog` at this epoch,
 * after the mutation step and before execution (what Evaluate() receives),
 * together with its origin ids. `soup`/`origin` are the state BEFORE the
 * epoch. Returns the pair's programs in p1/p2. */
void ts_pair_before_execution(const uint8_t *soup, const uint64_t *origin,
                              const uint32_t *s, uint64_t n,
                              uint64_t run_seed, uint64_t epoch,
                              uint32_t mutation_prob, uint32_t prog,
                              uint8_t *tape, uint64_t *tok, uint32_t *p1_out,
                              uint32_t *p2_out) {
  uint64_t slot = 0;
  while (slot < n && s[slot] != prog) slot++;
  uint64_t index = slot / 2;
  uint32_t p1 = s[2 * index], p2 = s[2 * index + 1];
  memcpy(tape, soup + (uint64_t)p1 * TAPE, TAPE);
  memcpy(tape + TAPE, soup + (uint64_t)p2 * TAPE, TAPE);
  memcpy(tok, origin + (uint64_t)p1 * TAPE, TAPE * sizeof(uint64_t));
  memcpy(tok + TAPE, origin + (uint64_t)p2 * TAPE, TAPE * sizeof(uint64_t));
  mutate_pair(tape, tok, n, mix_seed(run_seed, epoch), index, mutation_prob,
              epoch, p1, p2);
  *p1_out = p1;
  *p2_out = p2;
}