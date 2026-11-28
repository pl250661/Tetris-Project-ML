import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(279, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 40)
        )

    def forward(self, state):
        return self.network(state)


class CUSTOM_AI_MODEL:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DQN().to(self.device)

        model_path = (Path(__file__).resolve().parent.parent/ "dqn_tetris.pth")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
        )

        self.model.eval()

    def get_state(self, board, piece):

        board_state = np.array(
            board.board, dtype=np.float32
        ).flatten()

        piece_state = np.zeros((4, 4), dtype=np.float32)

        for x, y in piece.body:
            piece_state[y, x] = 1

        heights = np.array(
            board.heights, dtype=np.float32
        ) / board.height

        holes = np.zeros(board.width, dtype=np.float32)

        for x in range(board.width):
            for y in range(board.heights[x]):
                if not board.board[y][x]:
                    holes[x] += 1

        holes /= board.height

        aggregate_height = sum(board.heights)

        total_holes = 0

        for x in range(board.width):
            for y in range(board.heights[x]):
                if not board.board[y][x]:
                    total_holes += 1

        bumpiness = sum(
            abs(board.heights[x] - board.heights[x + 1])
            for x in range(board.width - 1)
        )

        extra_features = np.array([
            aggregate_height / (board.width * board.height),
            total_holes / (board.width * board.height),
            bumpiness / (board.width * board.height)
        ], dtype=np.float32)

        state = np.concatenate([
            board_state,
            piece_state.flatten(),
            heights,
            holes,
            extra_features
        ])

        return state

    def get_rotations(self, piece):

        rotations = []
        current = piece

        for _ in range(4):
            rotations.append(current)
            current = current.get_next_rotation()

        return rotations

    def valid_actions(self, board, rotations):

        valid = np.zeros(40, dtype=bool)

        for r, current_piece in enumerate(rotations):

            for x in range(board.width):

                if x + len(current_piece.skirt) > board.width:
                    continue

                y = board.drop_height(current_piece, x)

                if y < 0:
                    continue

                if (
                    y + max(pos[1] for pos in current_piece.body)
                    >= board.height + 4
                ):
                    continue

                collision = False

                for px, py in current_piece.body:

                    bx = x + px
                    by = y + py

                    if board.board[by][bx]:
                        collision = True
                        break

                if not collision:
                    valid[r * 10 + x] = True

        return valid

    def get_best_move(self, board, piece, depth=1):

        rotations = self.get_rotations(piece)

        state = self.get_state(board, piece)

        valid = self.valid_actions(board, rotations)

        possible_actions = np.flatnonzero(valid)

        if len(possible_actions) == 0:
            raise RuntimeError("No legal Tetris moves available")

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(state_tensor)

            mask = torch.tensor(
                valid,
                dtype=torch.bool,
                device=self.device
            )

            q_values[0][~mask] = -float("inf")

            action = q_values.argmax(dim=1).item()

        rotation = action // 10
        x = action % 10

        selected_piece = rotations[rotation]

        return x, selected_piece
