"""Evaluation cadence and early stopping policy."""
from dataclasses import dataclass
import math


def evaluation_step(args):
    if args.max_iter == 50:
        step = 40
    elif args.max_iter == 120:
        step = 90
    elif args.max_iter == 40:
        step = 30
    else:
        lr_steps = args.lr_step if isinstance(args.lr_step, (list, tuple)) else [int(args.lr_step)]
        step = int(lr_steps[0])
    return step


@dataclass
class EarlyStopping:
    patience: int = 10
    min_iter: int = 40
    best: float = 0.0
    no_improve: int = 0
    has_score: bool = False

    def observe(self, score):
        if not math.isfinite(score):
            self.no_improve += 1
            return False
        improved = not self.has_score or score > self.best
        if improved:
            self.has_score = True
            self.best = score
            self.no_improve = 0
        else:
            self.no_improve += 1
        return improved

    def should_stop(self, iteration):
        return self.patience > 0 and iteration >= self.min_iter and self.no_improve >= self.patience
