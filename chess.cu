// This is a skeleton of a GPU-accelerated chess engine using CUDA.
// Now the game only supports pawns and kings with very basic move generation and evaluation.
// The focus is on the GPU integration and multi-GPU threading structure.
// Build example:
//   nvcc -std=c++17 -O2 gpu_chess_skeleton.cu -o gpu_chess
//
// Run:
//   ./chess_gpu.exe
//
// Notes:
// - Host keeps the authoritative board for simplicity.
// - Each turn: board -> GPU, GPU proposes move, move -> host, host applies it.
// - Two host threads bind to two GPUs and alternate turns via condition_variable.

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "piece_moves.cuh"
#include "evaluate.cuh"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)                    \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";               \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// --------------------------------------------
// Device-side RNG helper (simple placeholder)
// --------------------------------------------
__device__ uint32_t lcg_next(uint32_t &state) {
  state = 1664525u * state + 1013904223u;
  return state;
}

// --------------------------------------------
// Device-side move generation and selection
// --------------------------------------------

__device__ int generate_moves_device(const Board* b, Move* out_moves, int max_moves) {
  int count = 0;
  int side = b->side_to_move; // +1 white, -1 black

  //all possible moves
  for (int i = 0; i < 64 && count < max_moves; i++) {
    int8_t p = b->squares[i];
    if (p == EMPTY) continue;

    if ((p > 0) != (side > 0)) continue; // not our piece

    switch (p) {
      case WP: case BP: {
        count = generate_pawn_moves(b, i, side, out_moves, count, max_moves);
        break;
      }
      case WK: case BK: {
        count = generate_king_moves(b, i, side, out_moves, count, max_moves);
        break;
      }
      //more pieces to be added
    }
  }   
  count = min(count, max_moves);
  return count;
} 


__device__ void apply_move_device(Board& b, const Move& m) {
  int8_t p = b.squares[(int)m.from];
  b.squares[(int)m.from] = EMPTY;
  b.squares[(int)m.to] = p;
}


// This kernel should pick one move for the current board.
__global__ void choose_move_kernel(const Board* d_board,
                                   Move* d_best_move,
                                   int* d_found,
                                   uint32_t seed) {
  __shared__ Move moves[256];
  __shared__ int  move_count;
  __shared__ int  scores[256];

  // Generate moves once
  if (threadIdx.x == 0) {
    move_count = generate_moves_device(d_board, moves, 256);
    if (move_count > 256) move_count = 256; 
    *d_found = (move_count > 0) ? 1 : 0;
  }
  __syncthreads();

  if (move_count <= 0) return;

  int i = threadIdx.x;

  // Only score valid moves
  if (i < move_count) {
    Board tmp = *d_board;
    apply_move_device(tmp, moves[i]);
    int s = eval_board_simplest(&tmp);

    // Higher is better for side to move
    s *= (int)d_board->side_to_move;

    scores[i] = s;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Find best score among valid moves
    int best_i = 0;
    for (int k = 1; k < move_count; k++) {
      if (scores[k] > scores[best_i]) best_i = k;
    }

    // Collect ties (indices), including k=0
    int best_score = scores[best_i];
    int best_idx[256];
    int best_count = 0;
    for (int k = 0; k < move_count; k++) {
      if (scores[k] == best_score) best_idx[best_count++] = k;
    }

    // Random pick among ties
    uint32_t st = seed ^ (uint32_t)clock64();
    uint32_t r  = lcg_next(st);
    int pick = best_idx[r % (uint32_t)best_count];

    *d_best_move = moves[pick];
    *d_found = 1;
  }
}


// ------------------------------------------------------
// Host-side chess hooks (apply move, end condition)
// ------------------------------------------------------

void apply_move_host(Board& b, const Move& m) {
  // PLACEHOLDER: move piece, no legality checks
  int8_t p = b.squares[(int)m.from];
  b.squares[(int)m.from] = EMPTY;
  b.squares[(int)m.to] = p;
  // b.side_to_move = -b.side_to_move;
}

//check game over conditions
GameResult check_game_over_host(const Board& b, int ply_count) {
  // PLACEHOLDER: stop after N plies
  if (ply_count >= 80) return DRAW;
  //find king
  bool found_K = false;
  for (int i = 0; i < 64; i++) {
    if (b.side_to_move > 0 && b.squares[i] == BK
    || b.side_to_move < 0 && b.squares[i] == WK) {
      found_K = true;
      break;
    }
  }
  if (!found_K) {
    if (b.side_to_move > 0) return WHITE_WINS;
    else return BLACK_WINS;
  }
  return ONGOING;
}

// Initialize a standard-ish position (you can replace with real chess init)
Board make_initial_board() {
  Board b{};
  std::memset(b.squares, 0, sizeof(b.squares));
  b.side_to_move = +1;

  // SUPER MINIMAL setup: pawns only (to keep your early debugging simpler)
  for (int file = 0; file < 8; file++) {
    b.squares[8 + file] = WP;       // white pawns on rank 2
    b.squares[48 + file] = BP;      // black pawns on rank 7
  }
  // Optional: add kings so "game over" can be meaningful later.
  b.squares[4]  = WK;
  b.squares[60] = BK;

  return b;
}

// Optional: pretty print board for debugging
void print_board(const Board& b, bool isStart=false) {
  auto piece_char = [](int8_t p) -> char {
    switch (p) {
      case WP: return 'P'; case WN: return 'N'; case WB: return 'B';
      case WR: return 'R'; case WQ: return 'Q'; case WK: return 'K';
      case BP: return 'p'; case BN: return 'n'; case BB: return 'b';
      case BR: return 'r'; case BQ: return 'q'; case BK: return 'k';
      default: return '.';
    }
  };
  if (!isStart) std::cout << (b.side_to_move > 0 ? "White" : "Black") << " to move\n";
  else std::cout << "Initial position\n";
  for (int r = 7; r >= 0; r--) {
    for (int f = 0; f < 8; f++) {
      std::cout << piece_char(b.squares[r * 8 + f]) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  std::cout << std::flush;
}

// ------------------------------------------------------
// Per-GPU "player" thread context + loop
// ------------------------------------------------------

struct PlayerGPUContext {
  int device_id;
  int side;

  // Device allocations (per GPU)
  Board* d_board = nullptr;
  Move*  d_best_move = nullptr;
  int*   d_found = nullptr;

  cudaStream_t stream = nullptr;

  void init() {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc(&d_board, sizeof(Board)));
    CUDA_CHECK(cudaMalloc(&d_best_move, sizeof(Move)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
  }

  void destroy() {
    CUDA_CHECK(cudaSetDevice(device_id));
    if (d_board) CUDA_CHECK(cudaFree(d_board));
    if (d_best_move) CUDA_CHECK(cudaFree(d_best_move));
    if (d_found) CUDA_CHECK(cudaFree(d_found));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
  }

  // Given a host board, ask this GPU to pick a move.
  // Returns true if a move was found.
  bool choose_move(const Board& h_board, Move& out_move, uint32_t seed) {
    CUDA_CHECK(cudaSetDevice(device_id));

    CUDA_CHECK(cudaMemcpyAsync(d_board, &h_board, sizeof(Board),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemsetAsync(d_found, 0, sizeof(int), stream));
    choose_move_kernel<<<1, 256, 0, stream>>>(d_board, d_best_move, d_found, seed);
    CUDA_CHECK(cudaGetLastError());
  
    int found = 0;
    CUDA_CHECK(cudaMemcpyAsync(&found, d_found, sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (found) {
      CUDA_CHECK(cudaMemcpy(&out_move, d_best_move, sizeof(Move),
                            cudaMemcpyDeviceToHost));
    }
    return found != 0;

  }
};

// Shared state between threads
struct SharedGameState {
  Board board;
  int ply_count = 0;
  GameResult result = ONGOING;

  std::mutex m;
  std::condition_variable cv;

  std::atomic<bool> stop{false};
};

// Thread function for one GPU player
void gpu_player_thread(PlayerGPUContext& ctx, SharedGameState* gs) {
  ctx.init();

  std::mt19937 rng((unsigned)std::hash<std::thread::id>{}(std::this_thread::get_id()));

  while (!gs->stop.load()) {
    Board local_board{};
    int my_side = ctx.side;

    {
      std::unique_lock<std::mutex> lk(gs->m);
      gs->cv.wait(lk, [&]{
        return gs->stop.load() || (gs->result == ONGOING && gs->board.side_to_move == ctx.side);
      });

      if (gs->stop.load() || gs->result != ONGOING) break;

      if (gs->board.side_to_move != my_side) {
        // Not my turn (shouldn’t happen if mapping & wait predicate align)
        continue;
      }
      local_board = gs->board;
    }

    // Choose move on GPU
    Move m{};
    uint32_t seed = (uint32_t)rng();

    bool found = ctx.choose_move(local_board, m, seed);

    {
      std::lock_guard<std::mutex> lk(gs->m);
      if (gs->result != ONGOING) break;

      if (!found) {
        // No moves found => treat as loss for side to move (placeholder)
        gs->result = (my_side > 0) ? BLACK_WINS : WHITE_WINS;
        gs->stop.store(true);
      } else {
        apply_move_host(gs->board, m);
        gs->ply_count++;
        print_board(gs->board);
        gs->result = check_game_over_host(gs->board, gs->ply_count);
        gs->board.side_to_move *= -1;

        if (gs->result != ONGOING) {
          gs->stop.store(true);
        }
      }
    }

    gs->cv.notify_all();
  }

  ctx.destroy();
}

// --------------------------------------------
// Main: spawn two GPU players, run game
// --------------------------------------------

int main() {

  freopen("log.txt", "w", stdout);
  // freopen("chess_error.txt", "w", stderr);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  bool single_gpu_emulation = (device_count < 2);
  if (single_gpu_emulation) {
    printf("Only %d CUDA device found. Running in single-GPU emulation mode.\n", device_count);
  }
  int dev0 = 0;
  int dev1 = single_gpu_emulation ? 0 : 1;  // both map to same physical GPU in emulation

  SharedGameState gs;
  gs.board = make_initial_board();

  std::cout << "Starting game on GPU0 (White) vs GPU1 (Black)\n";
  print_board(gs.board, true);

  PlayerGPUContext p0{dev0, +1}, p1{dev1, -1};

  std::thread t0(gpu_player_thread, std::ref(p0), &gs);
  std::thread t1(gpu_player_thread, std::ref(p1), &gs);

  // Kick off
  gs.cv.notify_all();

  t0.join();
  t1.join();

  // Final report
  std::cout << "Game ended after " << gs.ply_count << " plies.\n";
  switch (gs.result) {
    case WHITE_WINS: std::cout << "Result: White wins\n"; break;
    case BLACK_WINS: std::cout << "Result: Black wins\n"; break;
    case DRAW:       std::cout << "Result: Draw\n"; break;
    default:         std::cout << "Result: Ongoing? (unexpected)\n"; break;
  }

  return 0;
}
