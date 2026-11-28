import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from board import Board

class AfterstateDQN(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

class CUSTOM_AI_MODEL:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AfterstateDQN().to(self.device)

        model_path = Path(__file__).resolve().parent.parent / "dqn_tetris.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def extract_afterstate_features(self, board, lines_cleared):
        width = board.width
        height = board.height
        heights = board.heights

        agg_height = sum(heights)
        max_height = max(heights) if heights else 0

        holes = 0
        for x in range(width):
            for y in range(heights[x]):
                if not board.board[y][x]:
                    holes += 1

        bumpiness = sum(abs(heights[x] - heights[x + 1]) for x in range(width - 1))

        wells = 0
        for x in range(width):
            left_h = heights[x - 1] if x > 0 else height
            right_h = heights[x + 1] if x < width - 1 else height
            min_neighbor = min(left_h, right_h)
            if min_neighbor > heights[x]:
                wells += (min_neighbor - heights[x])

        return np.array([
            agg_height / 200.0,
            max_height / 20.0,
            holes / 200.0,
            bumpiness / 180.0,
            wells / 200.0,
            lines_cleared / 4.0
        ], dtype=np.float32)

    def get_best_move(self, board, piece, depth=1):
        candidates = []
        current = piece
        seen_bodies = set()

        for _ in range(4):
            body_key = tuple(sorted(current.body))
            if body_key not in seen_bodies:
                seen_bodies.add(body_key)

                for x in range(board.width):
                    if x + len(current.skirt) > board.width:
                        continue

                    y = board.drop_height(current, x)
                    if y < 0 or y + max(pos[1] for pos in current.body) >= board.height + 4:
                        continue

                    sim_board = Board()
                    sim_board.board = [row[:] for row in board.board]
                    sim_board.widths = board.widths[:]
                    sim_board.heights = board.heights[:]

                    res = sim_board.place(x, y, current)
                    if isinstance(res, Exception):
                        continue

                    lines = sim_board.clear_rows()
                    done = sim_board.top_filled()
                    features = self.extract_afterstate_features(sim_board, lines)

                    candidates.append((x, current, features, lines, done))

            current = current.get_next_rotation()

        if not candidates:
            # Fallback if game over is unavoidable on next drop
            return 0, piece

        feats = np.array([c[2] for c in candidates], dtype=np.float32)
        feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            values = self.model(feats_t)
            imm_rewards = torch.tensor(
                [-10.0 if c[4] else (1.0 + (c[3] ** 2) * 10.0) for c in candidates],
                dtype=torch.float32,
                device=self.device
            )
            not_done = torch.tensor(
                [0.0 if c[4] else 1.0 for c in candidates],
                dtype=torch.float32,
                device=self.device
            )
            scores = imm_rewards + 0.99 * values * not_done
            best_idx = int(scores.argmax().item())

        return candidates[best_idx][0], candidates[best_idx][1]