/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "movepick.h"

#include <array>
#include <cassert>
#include <limits>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

namespace {

enum Stages {
    // generate main search moves
    MAIN_TT,
    CAPTURE_INIT,
    GOOD_CAPTURE,
    QUIET_INIT,
    GOOD_QUIET,
    BAD_CAPTURE,
    BAD_QUIET,

    // generate evasion moves
    EVASION_TT,
    EVASION_INIT,
    EVASION,

    // generate probcut moves
    PROBCUT_TT,
    PROBCUT_INIT,
    PROBCUT,

    // generate qsearch moves
    QSEARCH_TT,
    QCAPTURE_INIT,
    QCAPTURE
};


// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
}

int quiet_prior_phase(const Position& pos) {
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
    if (pos.game_ply() < 12 && remainingMaterial >= 40)
        return 0;
    if (remainingMaterial <= 18 || nonKingPieces <= 8)
        return 2;
    return 1;
}

bool quiet_prior_low_tactical_pressure(const Position& pos) {
    int legalCount = 0;
    int tacticalCount = 0;
    for (const Move move : MoveList<LEGAL>(pos))
    {
        ++legalCount;
        if (pos.capture_stage(move) || move.type_of() == PROMOTION || pos.gives_check(move))
            ++tacticalCount;
    }
    return legalCount > 0 && tacticalCount * 5 <= legalCount;
}

}  // namespace


// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const SharedHistories*       sh,
                       int                          pl,
                       int                          quietStrengthValue,
                       int                          quietPlyFromRootValue,
                       int                          quietCountMinValue) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    sharedHistory(sh),
    ttMove(ttm),
    depth(d),
    ply(pl),
    quietStrength(quietStrengthValue),
    quietPlyFromRoot(quietPlyFromRootValue),
    quietCountMin(quietCountMinValue) {

    if (pos.checkers())
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

    else
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th) {
    assert(!pos.checkers());

    stage = PROBCUT_TT + !(ttm && pos.capture_stage(ttm) && pos.pseudo_legal(ttm));
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
ExtMove* MovePicker::score(const MoveList<Type>& ml) {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    Color us = pos.side_to_move();

    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
    if constexpr (Type == QUIETS)
    {
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
        threatByLesser[KING]  = 0;
    }

    ExtMove* it = cur;
    if constexpr (Type == QUIETS)
    {
        int       topQuietHistory    = std::numeric_limits<int>::min();
        int       secondQuietHistory = std::numeric_limits<int>::min();
        int quietCount = 0;
        ExtMove* quietBegin = it;
        ExtMove* historyBest = quietBegin;
        ExtMove* historySecond = quietBegin;
        for (auto move : ml)
        {
            ExtMove& m = *it++;
            m          = move;

            const Square    from = m.from_sq();
            const Square    to   = m.to_sq();
            const Piece     pc   = pos.moved_piece(m);
            const PieceType pt   = type_of(pc);

            int value = 2 * (*mainHistory)[us][move.raw()];
            value += 2 * sharedHistory->pawn_entry(pos)[pc][to];
            value += (*continuationHistory[0])[pc][to];
            value += (*continuationHistory[1])[pc][to];
            value += (*continuationHistory[2])[pc][to];
            value += (*continuationHistory[3])[pc][to];
            value += (*continuationHistory[5])[pc][to];
            value += (bool(pos.check_squares(pt) & to) && pos.see_ge(move, -75)) * 16384;
            int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
            value += PieceValue[pt] * v;
            if (ply < LOW_PLY_HISTORY_SIZE)
                value += 8 * (*lowPlyHistory)[ply][move.raw()] / (1 + ply);

            m.value = value;
            if (value > topQuietHistory)
            {
                secondQuietHistory = topQuietHistory;
                topQuietHistory    = value;
                historySecond      = historyBest;
                historyBest        = &m;
            }
            else if (value > secondQuietHistory)
            {
                secondQuietHistory = value;
                historySecond      = &m;
            }
            ++quietCount;
        }

        const bool calmPhase = quiet_prior_phase(pos) <= 1;
        const bool regimeCandidate = quietStrength > 0 && quietCountMin > 0 && quietCount >= quietCountMin
                                   && quietPlyFromRoot <= 1 && calmPhase && QuietPrior::enabled();
        const bool applyQuietPrior = regimeCandidate && quiet_prior_low_tactical_pressure(pos);
        QuietPrior::record_regime_gate(applyQuietPrior, quietCount);
        if (applyQuietPrior)
            quietPrior = QuietPrior::evaluate(pos, quietPlyFromRoot);

        if (applyQuietPrior && quietPrior.ok)
        {
            const QuietPrior::MoveInfo* bestInfo  = QuietPrior::find_move(quietPrior, Move(*historyBest));
            const QuietPrior::MoveInfo* secondInfo = QuietPrior::find_move(quietPrior, Move(*historySecond));
            const bool swapped = bestInfo && secondInfo && secondInfo->prob > bestInfo->prob;
            if (swapped)
                std::swap(historyBest->value, historySecond->value);
            QuietPrior::record_bonus_application(2, swapped, 0, 0);
        }

        return it;
    }

    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

        const Square to   = m.to_sq();
        const Piece  pc   = pos.moved_piece(m);

        if constexpr (Type == CAPTURES)
        {
            const Piece capturedPiece = pos.piece_on(to);
            m.value = (*captureHistory)[pc][to][type_of(capturedPiece)] + 7 * int(PieceValue[capturedPiece]);
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
            {
                const Piece capturedPiece = pos.piece_on(to);
                m.value = PieceValue[capturedPiece] + (1 << 28);
            }
            else
                m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
        }
    }

    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.
template<typename Pred>
Move MovePicker::select(Pred filter) {

    for (; cur < endCur; ++cur)
        if (*cur != ttMove && filter())
            return *cur++;

    return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move() {

    constexpr int goodQuietThreshold = -14000;
top:
    switch (stage)
    {

    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT : {
        MoveList<CAPTURES> ml(pos);

        cur = endBadCaptures = moves;
        endCur = endCaptures = score<CAPTURES>(ml);

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
        ++stage;
        goto top;
    }

    case GOOD_CAPTURE :
        if (select([&]() {
                if (pos.see_ge(*cur, -cur->value / 18))
                    return true;
                std::swap(*endBadCaptures++, *cur);
                return false;
            }))
            return *(cur - 1);

        ++stage;
        [[fallthrough]];

    case QUIET_INIT :
        if (!skipQuiets)
        {
            MoveList<QUIETS> ml(pos);

            endCur = endGenerated = score<QUIETS>(ml);

            partial_insertion_sort(cur, endCur, -3560 * depth);
        }

        ++stage;
        [[fallthrough]];

    case GOOD_QUIET :
        if (!skipQuiets && select([&]() { return cur->value > goodQuietThreshold; }))
            return *(cur - 1);

        // Prepare the pointers to loop over the bad captures
        cur    = moves;
        endCur = endBadCaptures;

        ++stage;
        [[fallthrough]];

    case BAD_CAPTURE :
        if (select([]() { return true; }))
            return *(cur - 1);

        // Prepare the pointers to loop over quiets again
        cur    = endCaptures;
        endCur = endGenerated;

        ++stage;
        [[fallthrough]];

    case BAD_QUIET :
        if (!skipQuiets)
            return select([&]() { return cur->value <= goodQuietThreshold; });

        return Move::none();

    case EVASION_INIT : {
        MoveList<EVASIONS> ml(pos);

        cur    = moves;
        endCur = endGenerated = score<EVASIONS>(ml);

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
        ++stage;
        [[fallthrough]];
    }

    case EVASION :
    case QCAPTURE :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });
    }

    assert(false);
    return Move::none();  // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

}  // namespace Stockfish
