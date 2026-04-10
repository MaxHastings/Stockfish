#include "quietprior.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "movegen.h"

namespace Stockfish::QuietPrior {

namespace {

constexpr char MAGIC[8] = {'C', 'H', 'F', 'I', 'S', 'H', 'R', '2'};
constexpr std::uint32_t FORMAT_VERSION = 4;
constexpr std::string_view EXPECTED_FEATURE_VERSION = "v2";
constexpr std::string_view EXPECTED_MODEL_KIND = "factorized_v3";
constexpr std::size_t kMaxCacheEntries = 32768;
constexpr std::uint32_t kMaxStringBytes = 256;

struct Layer {
    std::uint32_t     inDim = 0;
    std::uint32_t     outDim = 0;
    std::vector<float> weights;
    std::vector<float> bias;
};

struct Artifact {
    std::string         featureVersion;
    std::string         modelKind;
    std::uint32_t       positionDim = 0;
    std::uint32_t       moveDim = 0;
    std::uint32_t       globalDim = 0;
    std::uint32_t       hiddenDim = 0;
    std::vector<Layer>  positionLayers;
    std::vector<Layer>  moveLayers;
    std::vector<Layer>  globalLayers;
    std::vector<Layer>  positionHeadLayers;
    std::vector<Layer>  moveHeadLayers;
};

bool read_exact(std::istream& input, void* data, std::size_t size) {
    input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
    return input.good();
}

class Runtime {
   public:
    bool load(std::string_view path) {
        std::ifstream input(std::string(path), std::ios::binary);
        if (!input)
            return false;

        char magic[sizeof(MAGIC)];
        if (!read_exact(input, magic, sizeof(magic)))
            return false;
        if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0)
            return false;

        const auto version = read_u32(input);
        if (version != FORMAT_VERSION)
            return false;

        Artifact artifact;
        if (!read_string(input, artifact.featureVersion) || !read_string(input, artifact.modelKind))
            return false;
        if (artifact.featureVersion != EXPECTED_FEATURE_VERSION || artifact.modelKind != EXPECTED_MODEL_KIND)
            return false;

        artifact.positionDim = read_u32(input);
        artifact.moveDim     = read_u32(input);
        artifact.globalDim   = read_u32(input);
        artifact.hiddenDim   = read_u32(input);
        if (artifact.positionDim == 0 || artifact.moveDim == 0 || artifact.hiddenDim == 0)
            return false;

        if (!read_layer_stack(input, artifact.positionLayers) || !read_layer_stack(input, artifact.moveLayers)
            || !read_layer_stack(input, artifact.globalLayers) || !read_layer_stack(input, artifact.positionHeadLayers)
            || !read_layer_stack(input, artifact.moveHeadLayers))
            return false;

        if (!validate_layer_stack(artifact.positionLayers, artifact.positionDim, artifact.hiddenDim)
            || !validate_layer_stack(artifact.moveLayers, artifact.moveDim, artifact.hiddenDim)
            || !validate_head_stack(artifact.positionHeadLayers, artifact.hiddenDim)
            || !validate_head_stack(artifact.moveHeadLayers, artifact.hiddenDim))
            return false;

        if (artifact.globalDim == 0) {
            if (!artifact.globalLayers.empty())
                return false;
        } else if (!validate_layer_stack(artifact.globalLayers, artifact.globalDim, artifact.hiddenDim)) {
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        artifact_ = std::move(artifact);
        loaded_   = true;
        cache_.clear();
        cacheOrder_.clear();
        return true;
    }

    void unload() {
        std::lock_guard<std::mutex> lock(mutex_);
        artifact_ = Artifact{};
        loaded_   = false;
        cache_.clear();
        cacheOrder_.clear();
    }

    bool enabled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_;
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    void reset_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = Stats{};
    }

    void record_regime_gate(bool passed, int signal) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.gateChecks++;
        stats_.gateSignalSum += std::uint64_t(std::max(signal, 0));
        stats_.gateSignalMax = std::max(stats_.gateSignalMax, std::uint64_t(std::max(signal, 0)));
        if (passed)
            stats_.gatePasses++;
        else
            stats_.gateMisses++;
    }

    void record_bonus_application(int moveCount, bool changedTopMove, int totalBonus, int totalAbsBonus) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.bonusCalls++;
        stats_.bonusMoves += std::uint64_t(std::max(moveCount, 0));
        stats_.bonusSum += totalBonus;
        stats_.bonusAbsSum += std::uint64_t(std::max(totalAbsBonus, 0));
        if (changedTopMove)
            stats_.topMoveChanges++;
    }

    PositionResult evaluate(const Position& pos, int plyFromRoot) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_)
            return {};

        const auto start = std::chrono::steady_clock::now();
        const Key key = pos.key();
        const auto it = cache_.find(key);
        if (it != cache_.end())
        {
            stats_.evaluations++;
            stats_.cacheHits++;
            stats_.legalMoves += it->second.moves.size();
            stats_.evalNanos +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            return it->second;
        }

        PositionResult result;
        result.key = key;
        encode_and_score_position(pos, plyFromRoot, result);
        stats_.evaluations++;
        stats_.cacheMisses++;
        stats_.legalMoves += result.moves.size();
        stats_.evalNanos +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        store_result(result);
        return result;
    }

   private:
    static std::uint32_t read_u32(std::istream& input) {
        std::uint32_t value = 0;
        if (!read_exact(input, &value, sizeof(value)))
            return 0;
        return value;
    }

    static bool read_string(std::istream& input, std::string& out) {
        const std::uint32_t size = read_u32(input);
        if (size > kMaxStringBytes)
            return false;
        out.assign(size, '\0');
        if (size == 0)
            return true;
        return read_exact(input, out.data(), size);
    }

    static bool read_layer_stack(std::istream& input, std::vector<Layer>& layers) {
        const std::uint32_t count = read_u32(input);
        layers.clear();
        layers.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            Layer layer;
            layer.inDim  = read_u32(input);
            layer.outDim = read_u32(input);
            if (layer.inDim == 0 || layer.outDim == 0)
                return false;
            layer.weights.resize(std::size_t(layer.inDim) * std::size_t(layer.outDim));
            layer.bias.resize(layer.outDim);
            if (!read_exact(input, layer.weights.data(), layer.weights.size() * sizeof(float)))
                return false;
            if (!read_exact(input, layer.bias.data(), layer.bias.size() * sizeof(float)))
                return false;
            layers.push_back(std::move(layer));
        }
        return true;
    }

    static bool validate_layer_stack(const std::vector<Layer>& layers, std::uint32_t inputDim, std::uint32_t outputDim) {
        if (layers.empty())
            return false;
        if (layers.front().inDim != inputDim || layers.back().outDim != outputDim)
            return false;
        for (std::size_t i = 1; i < layers.size(); ++i)
            if (layers[i - 1].outDim != layers[i].inDim)
                return false;
        return true;
    }

    static bool validate_head_stack(const std::vector<Layer>& layers, std::uint32_t hiddenDim) {
        if (layers.empty())
            return false;
        if (layers.front().inDim != hiddenDim || layers.back().outDim != 1)
            return false;
        for (std::size_t i = 1; i < layers.size(); ++i)
            if (layers[i - 1].outDim != layers[i].inDim)
                return false;
        return true;
    }

    static int piece_type_index(PieceType pt) {
        switch (pt)
        {
        case PAWN: return 0;
        case KNIGHT: return 1;
        case BISHOP: return 2;
        case ROOK: return 3;
        case QUEEN: return 4;
        case KING: return 5;
        default: return -1;
        }
    }

    static Square canonical_square(Color us, Square sq) {
        return us == WHITE ? sq : flip_rank(sq);
    }

    static int piece_symbol_index(Piece pc, Color us) {
        if (pc == NO_PIECE)
            return -1;
        const Color relativeColor = color_of(pc) == us ? WHITE : BLACK;
        const int offset = relativeColor == WHITE ? 0 : 6;
        return offset + piece_type_index(type_of(pc));
    }

    static int promotion_index(Move move) {
        if (move.type_of() != PROMOTION)
            return 0;
        switch (move.promotion_type())
        {
        case KNIGHT: return 1;
        case BISHOP: return 2;
        case ROOK: return 3;
        case QUEEN: return 4;
        default: return 0;
        }
    }

    static Square feature_to_sq(const Position& pos, Move move) {
        Square to = move.to_sq();
        if (move.type_of() == CASTLING && !pos.is_chess960())
            to = make_square(to > move.from_sq() ? FILE_G : FILE_C, rank_of(move.from_sq()));
        return to;
    }

    static int captured_piece_index(const Position& pos, Move move) {
        Piece captured = pos.piece_on(feature_to_sq(pos, move));
        if (move.type_of() == EN_PASSANT)
            captured = make_piece(~pos.side_to_move(), PAWN);
        return captured == NO_PIECE ? 0 : piece_type_index(type_of(captured)) + 1;
    }

    std::vector<float> encode_position(const Position& pos) const {
        const Color us = pos.side_to_move();
        std::vector<float> out;
        out.reserve(artifact_.positionDim);
        out.push_back(1.0f);
        out.push_back(pos.can_castle(us == WHITE ? WHITE_OO : BLACK_OO) ? 1.0f : 0.0f);
        out.push_back(pos.can_castle(us == WHITE ? WHITE_OOO : BLACK_OOO) ? 1.0f : 0.0f);
        out.push_back(pos.can_castle(us == WHITE ? BLACK_OO : WHITE_OO) ? 1.0f : 0.0f);
        out.push_back(pos.can_castle(us == WHITE ? BLACK_OOO : WHITE_OOO) ? 1.0f : 0.0f);
        out.push_back(pos.ep_square() != SQ_NONE ? 1.0f : 0.0f);

        const auto count_piece = [&](PieceType pt) {
            return popcount(pos.pieces(WHITE, pt) | pos.pieces(BLACK, pt));
        };
        const int pawns   = count_piece(PAWN);
        const int knights = count_piece(KNIGHT);
        const int bishops = count_piece(BISHOP);
        const int rooks   = count_piece(ROOK);
        const int queens  = count_piece(QUEEN);
        const int nonKingPieces = pawns + knights + bishops + rooks + queens;
        const int remainingMaterial = pawns + 3 * knights + 3 * bishops + 5 * rooks + 9 * queens;
        int phase = 1;
        if (pos.game_ply() < 12 && remainingMaterial >= 40)
            phase = 0;
        else if (remainingMaterial <= 18 || nonKingPieces <= 8)
            phase = 2;
        out.push_back(phase == 0 ? 1.0f : 0.0f);
        out.push_back(phase == 1 ? 1.0f : 0.0f);
        out.push_back(phase == 2 ? 1.0f : 0.0f);

        std::vector<float> pieceVector(12 * 64, 0.0f);
        const auto& board = pos.piece_array();
        for (int sq = 0; sq < 64; ++sq)
        {
            const Square canonicalSq = canonical_square(us, static_cast<Square>(sq));
            const int pieceIndex = piece_symbol_index(board[static_cast<Square>(sq)], us);
            if (pieceIndex >= 0)
                pieceVector[pieceIndex * 64 + canonicalSq] = 1.0f;
        }
        out.insert(out.end(), pieceVector.begin(), pieceVector.end());

        for (Color c : {us, ~us})
            for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN})
                out.push_back(float(popcount(pos.pieces(c, pt))));

        return out;
    }

    std::vector<float> encode_global(const Position& pos) const {
        std::vector<float> out;
        out.reserve(artifact_.globalDim);
        const int pawns = popcount(pos.pieces(PAWN));
        const int knights = popcount(pos.pieces(KNIGHT));
        const int bishops = popcount(pos.pieces(BISHOP));
        const int rooks = popcount(pos.pieces(ROOK));
        const int queens = popcount(pos.pieces(QUEEN));
        const int minors = knights + bishops;
        out.push_back(float(pawns + minors + rooks + queens));
        out.push_back(float(pawns));
        out.push_back(float(minors));
        out.push_back(float(rooks));
        out.push_back(float(queens));
        out.push_back(float(pos.game_ply()));
        return out;
    }

    std::vector<float> encode_move(const Position& pos, Move move) const {
        const Color us = pos.side_to_move();
        std::vector<float> out(artifact_.moveDim, 0.0f);
        std::size_t offset = 0;
        out[offset + int(canonical_square(us, move.from_sq()))] = 1.0f;
        offset += 64;
        out[offset + int(canonical_square(us, feature_to_sq(pos, move)))] = 1.0f;
        offset += 64;

        const Piece moved = pos.moved_piece(move);
        const int movingIndex = piece_type_index(type_of(moved));
        if (movingIndex >= 0)
            out[offset + movingIndex] = 1.0f;
        offset += 6;

        out[offset + captured_piece_index(pos, move)] = 1.0f;
        offset += 7;

        out[offset + promotion_index(move)] = 1.0f;
        offset += 5;

        out[offset++] = pos.capture(move) ? 1.0f : 0.0f;
        out[offset++] = move.type_of() == PROMOTION ? 1.0f : 0.0f;
        out[offset++] = move.type_of() == CASTLING ? 1.0f : 0.0f;
        out[offset++] = move.type_of() == EN_PASSANT ? 1.0f : 0.0f;
        out[offset++] = pos.gives_check(move) ? 1.0f : 0.0f;
        out[offset++] = 0.0f;
        return out;
    }

    static void dense_forward(const Layer& layer, const std::vector<float>& input, std::vector<float>& output, bool reluLast) {
        output.assign(layer.outDim, 0.0f);
        for (std::uint32_t outIdx = 0; outIdx < layer.outDim; ++outIdx)
        {
            float sum = 0.0f;
            const std::size_t rowOffset = std::size_t(outIdx) * std::size_t(layer.inDim);
            for (std::uint32_t inIdx = 0; inIdx < layer.inDim; ++inIdx)
                sum += layer.weights[rowOffset + inIdx] * input[inIdx];
            output[outIdx] = sum;
        }
        for (std::uint32_t outIdx = 0; outIdx < layer.outDim; ++outIdx)
        {
            output[outIdx] += layer.bias[outIdx];
            if (reluLast)
                output[outIdx] = std::max(0.0f, output[outIdx]);
        }
    }

    static std::vector<float> forward_stack(const std::vector<Layer>& layers, const std::vector<float>& input, bool reluLast) {
        std::vector<float> current = input;
        std::vector<float> next;
        for (std::size_t i = 0; i < layers.size(); ++i)
        {
            const bool applyRelu = reluLast || i + 1 < layers.size();
            dense_forward(layers[i], current, next, applyRelu);
            current.swap(next);
        }
        return current;
    }

    static std::vector<float> finalize_move_probabilities(std::vector<MoveInfo>& moves, const std::vector<float>& logits) {
        if (moves.empty() || logits.empty())
            return {};
        const float maxLogit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(logits.size(), 0.0f);
        float sum = 0.0f;
        for (std::size_t i = 0; i < logits.size(); ++i)
        {
            const float expv = std::exp(logits[i] - maxLogit);
            probs[i] = expv;
            sum += expv;
        }
        const float invSum = 1.0f / std::max(sum, 1.0e-20f);
        for (std::size_t i = 0; i < moves.size(); ++i)
        {
            moves[i].prob  = probs[i] * invSum;
            moves[i].rank  = 0;
            moves[i].logit = logits[i];
        }
        std::vector<std::size_t> order(moves.size());
        for (std::size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return moves[a].prob > moves[b].prob;
        });
        for (std::size_t rank = 0; rank < order.size(); ++rank)
            moves[order[rank]].rank = int(rank) + 1;
        return probs;
    }

    void encode_and_score_position(const Position& pos, int plyFromRoot, PositionResult& result) const {
        (void) plyFromRoot;
        const MoveList<LEGAL> legalMoves(pos);
        result.ok = true;
        if (legalMoves.size() == 0)
            return;

        const std::vector<float> positionInput = encode_position(pos);
        std::vector<float> positionContext = forward_stack(artifact_.positionLayers, positionInput, true);
        if (artifact_.globalDim > 0)
        {
            const std::vector<float> globalInput = encode_global(pos);
            std::vector<float> globalContext = forward_stack(artifact_.globalLayers, globalInput, true);
            for (std::size_t i = 0; i < positionContext.size() && i < globalContext.size(); ++i)
                positionContext[i] += globalContext[i];
        }
        const float positionBias = forward_stack(artifact_.positionHeadLayers, positionContext, false).front();

        std::vector<float> logits;
        logits.reserve(legalMoves.size());
        result.moves.reserve(legalMoves.size());
        for (Move move : legalMoves)
        {
            const std::vector<float> moveInput = encode_move(pos, move);
            const std::vector<float> moveContext = forward_stack(artifact_.moveLayers, moveInput, true);
            float interaction = 0.0f;
            for (std::size_t i = 0; i < positionContext.size() && i < moveContext.size(); ++i)
                interaction += positionContext[i] * moveContext[i];
            const float moveBias = forward_stack(artifact_.moveHeadLayers, moveContext, false).front();
            const float logit = interaction + positionBias + moveBias;
            logits.push_back(logit);
            result.moves.push_back({move, logit, 0.0f, 0});
        }

        const auto probs = finalize_move_probabilities(result.moves, logits);
        if (!probs.empty())
        {
            result.confidence = *std::max_element(probs.begin(), probs.end());
            result.debugMappedMoves = int(result.moves.size());
        }
    }

    void store_result(const PositionResult& result) {
        cache_[result.key] = result;
        cacheOrder_.push_back(result.key);
        if (cacheOrder_.size() <= kMaxCacheEntries)
            return;

        const std::size_t trimCount = kMaxCacheEntries / 4;
        for (std::size_t i = 0; i < trimCount && !cacheOrder_.empty(); ++i)
        {
            cache_.erase(cacheOrder_.front());
            cacheOrder_.pop_front();
        }
    }

    mutable std::mutex                     mutex_;
    bool                                   loaded_ = false;
    Artifact                               artifact_;
    std::unordered_map<Key, PositionResult> cache_;
    std::deque<Key>                        cacheOrder_;
    Stats                                  stats_;
};

Runtime& runtime() {
    static Runtime instance;
    return instance;
}

}  // namespace

bool enabled() { return runtime().enabled(); }
bool load(std::string_view path) { return runtime().load(path); }
void unload() { runtime().unload(); }
PositionResult evaluate(const Position& pos, int plyFromRoot) { return runtime().evaluate(pos, plyFromRoot); }
Stats stats() { return runtime().stats(); }
void reset_stats() { runtime().reset_stats(); }
void record_regime_gate(bool passed, int signal) { runtime().record_regime_gate(passed, signal); }
void record_bonus_application(int moveCount, bool changedTopMove, int totalBonus, int totalAbsBonus) {
    runtime().record_bonus_application(moveCount, changedTopMove, totalBonus, totalAbsBonus);
}

const MoveInfo* find_move(const PositionResult& result, Move move) {
    for (const auto& info : result.moves)
        if (info.move == move)
            return &info;
    return nullptr;
}

int quiet_bonus_for_move(
  const PositionResult& result,
  Move                  move,
  int                   strength,
  float                 confidenceScale) {
    const MoveInfo* info = find_move(result, move);
    if (!info || !result.ok || strength <= 0 || confidenceScale <= 0.0f)
        return 0;

    const float centered = std::log(std::max(info->prob, 1.0e-20f))
                         - std::log(1.0f / float(std::max<std::size_t>(1, result.moves.size())));
    const float bonus = centered * float(strength) * confidenceScale * 128.0f;
    return int(std::lround(bonus));
}

}  // namespace Stockfish::QuietPrior
