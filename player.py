from __future__ import annotations
import random
from typing import Optional
from abc import ABC, abstractmethod

import chess
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class Player(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, fen: str) -> Optional[str]:
        pass


# simple piece values (pawn = 1)
_PIECE_VALUE = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0,
}

def _rule_adjustment(board: chess.Board, move: chess.Move) -> float:
    board_after = board.copy()
    board_after.push(move)
    adj = 0.0

    if board_after.is_checkmate():
        return 100.0  # play mate if available

    if board_after.is_stalemate():
        return -50.0  # avoid stalemate

    if board_after.is_check():
        adj += 0.5

    # reward captures based on piece value
    if board.is_capture(move):
        target = board.piece_at(move.to_square)
        if target is not None:
            adj += _PIECE_VALUE.get(target.piece_type, 1.0) * 0.3

    # penalize leaving a piece where it can be taken
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is not None:
        if board_after.is_attacked_by(not board.turn, move.to_square):
            adj -= _PIECE_VALUE.get(moving_piece.piece_type, 1.0) * 0.4

    return adj


class TransformerPlayer(Player):

    MODEL_NAME = "Mnadoba/GPT2-fine-tuned-chess-model"

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{name}] Loading {self.MODEL_NAME} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _score_move(self, prompt: str, move_uci: str) -> float:
        
        full_text = prompt + move_uci
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        n_prompt = prompt_ids.shape[1]
        n_full = full_ids.shape[1]

        if n_full - n_prompt <= 0:
            return float("-inf")

        with torch.no_grad():
            logits = self.model(full_ids).logits

        log_probs = F.log_softmax(logits, dim=-1)

        # logit i predicts token i+1, so shift by one
        move_ids = full_ids[0, n_prompt:]
        pred_lp = log_probs[0, n_prompt - 1: n_full - 1, :]

        gathered = pred_lp.gather(1, move_ids.unsqueeze(1)).squeeze(1)
        return gathered.mean().item()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        prompt = f"FEN: {fen} Move: "

        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            uci = move.uci()
            rule_adj = _rule_adjustment(board, move)

            if rule_adj >= 100.0:
                return uci  # mate found

            try:
                lm_score = self._score_move(prompt, uci)
            except Exception:
                lm_score = float("-inf")

            total = lm_score + rule_adj

            if total > best_score:
                best_score = total
                best_move = uci

        return best_move if best_move is not None else random.choice(legal_moves).uci()
