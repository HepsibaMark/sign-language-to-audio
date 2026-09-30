"""
Canonical Landmark Dataset Generator (High Distinction Version)
Synthesizes accurate hand landmark geometry for standard ASL alphabet and key social phrases,
ensuring clean class separability and robust ML training.
"""

import os
import pickle
import numpy as np


def generate_hand_pose(
    thumb_state="folded",     # "folded", "extended", "up", "ring"
    index_state="folded",     # "folded", "extended", "bent"
    middle_state="folded",    # "folded", "extended", "bent"
    ring_state="folded",      # "folded", "extended", "bent"
    pinky_state="folded",     # "folded", "extended", "bent"
    spread=0.0
):
    """Generates 21 3D landmark points for a specific hand gesture configuration."""
    lm = np.zeros((21, 3), dtype=np.float32)

    # 0: Wrist at origin
    lm[0] = [0.0, 0.0, 0.0]

    # Thumb (1: CMC, 2: MCP, 3: IP, 4: Tip)
    if thumb_state == "extended":
        lm[1] = [-0.25, -0.15, 0.0]
        lm[2] = [-0.50, -0.30, 0.0]
        lm[3] = [-0.70, -0.45, 0.0]
        lm[4] = [-0.85, -0.55, 0.0]
    elif thumb_state == "up":
        lm[1] = [-0.20, -0.15, 0.0]
        lm[2] = [-0.30, -0.35, 0.0]
        lm[3] = [-0.35, -0.60, 0.0]
        lm[4] = [-0.38, -0.85, 0.0]  # Pointing straight up
    elif thumb_state == "ring":
        # Touching index tip
        lm[1] = [-0.15, -0.15, 0.0]
        lm[2] = [-0.25, -0.35, 0.0]
        lm[3] = [-0.20, -0.55, 0.05]
        lm[4] = [-0.10, -0.65, 0.05]
    else:  # folded
        lm[1] = [-0.15, -0.15, 0.0]
        lm[2] = [-0.25, -0.30, 0.0]
        lm[3] = [-0.18, -0.42, 0.06]
        lm[4] = [-0.12, -0.48, 0.08]

    # 4 Main fingers: Index, Middle, Ring, Pinky
    bases = [
        (-0.20, -0.65),  # Index MCP (5)
        (0.00, -0.70),   # Middle MCP (9)
        (0.20, -0.65),   # Ring MCP (13)
        (0.38, -0.58)    # Pinky MCP (17)
    ]

    finger_states = [index_state, middle_state, ring_state, pinky_state]
    finger_ids = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20)
    ]

    for idx, (mcp_id, pip_id, dip_id, tip_id) in enumerate(finger_ids):
        bx, by = bases[idx]
        state = finger_states[idx]
        lm[mcp_id] = [bx, by, 0.0]

        # Calculate spread offset
        spread_offset = (idx - 1.5) * spread

        if state == "extended":
            lm[pip_id] = [bx + spread_offset * 0.3, by - 0.25, 0.0]
            lm[dip_id] = [bx + spread_offset * 0.6, by - 0.48, 0.0]
            lm[tip_id] = [bx + spread_offset * 1.0, by - 0.70, 0.0]
        elif state == "bent":
            lm[pip_id] = [bx, by - 0.20, 0.05]
            lm[dip_id] = [bx, by - 0.25, 0.15]
            lm[tip_id] = [bx, by - 0.15, 0.20]
        else:  # folded into fist
            lm[pip_id] = [bx, by - 0.15, 0.08]
            lm[dip_id] = [bx, by - 0.05, 0.16]
            lm[tip_id] = [bx, by + 0.05, 0.12]

    # Normalize: origin to wrist, scale by distance to middle MCP
    lm = lm - lm[0]
    scale = np.linalg.norm(lm[9])
    if scale > 1e-4:
        lm = lm / scale

    return lm.flatten()


def generate_variations(base_coords: np.ndarray, num_samples: int = 150, noise_level: float = 0.025):
    """Produces diverse variations around base gesture with noise and rotations."""
    samples = []
    base_3d = base_coords.reshape(21, 3)

    for _ in range(num_samples):
        noise = np.random.normal(0, noise_level, base_3d.shape)
        noisy = base_3d + noise

        # 3D rotation jitter
        theta_z = np.radians(np.random.uniform(-10, 10))
        cz, sz = np.cos(theta_z), np.sin(theta_z)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        theta_x = np.radians(np.random.uniform(-6, 6))
        cx, sx = np.cos(theta_x), np.sin(theta_x)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

        rotated = np.dot(np.dot(noisy, rz.T), rx.T)

        # Scale jitter
        scale_fact = np.random.uniform(0.95, 1.05)
        scaled = rotated * scale_fact

        # Re-normalize
        ref_norm = np.linalg.norm(scaled[9] - scaled[0])
        if ref_norm > 1e-4:
            scaled = scaled / ref_norm

        samples.append(scaled.flatten())

    return samples


def generate_all():
    """Generates distinct, high-accuracy training set."""
    gesture_definitions = {
        "A": {"thumb_state": "folded", "index_state": "folded", "middle_state": "folded", "ring_state": "folded", "pinky_state": "folded"},
        "B": {"thumb_state": "folded", "index_state": "extended", "middle_state": "extended", "ring_state": "extended", "pinky_state": "extended", "spread": 0.0},
        "C": {"thumb_state": "extended", "index_state": "bent", "middle_state": "bent", "ring_state": "bent", "pinky_state": "bent"},
        "D": {"thumb_state": "ring", "index_state": "extended", "middle_state": "folded", "ring_state": "folded", "pinky_state": "folded"},
        "HELLO": {"thumb_state": "extended", "index_state": "extended", "middle_state": "extended", "ring_state": "extended", "pinky_state": "extended", "spread": 0.15},
        "HELP": {"thumb_state": "up", "index_state": "folded", "middle_state": "folded", "ring_state": "folded", "pinky_state": "folded"},
        "I": {"thumb_state": "folded", "index_state": "folded", "middle_state": "folded", "ring_state": "folded", "pinky_state": "extended"},
        "I LOVE YOU": {"thumb_state": "extended", "index_state": "extended", "middle_state": "folded", "ring_state": "folded", "pinky_state": "extended"},
        "L": {"thumb_state": "extended", "index_state": "extended", "middle_state": "folded", "ring_state": "folded", "pinky_state": "folded"},
        "OK": {"thumb_state": "ring", "index_state": "bent", "middle_state": "extended", "ring_state": "extended", "pinky_state": "extended", "spread": 0.05},
        "PEACE": {"thumb_state": "folded", "index_state": "extended", "middle_state": "extended", "ring_state": "folded", "pinky_state": "folded", "spread": 0.20},
        "WATER": {"thumb_state": "folded", "index_state": "extended", "middle_state": "extended", "ring_state": "extended", "pinky_state": "folded", "spread": 0.08},
        "Y": {"thumb_state": "extended", "index_state": "folded", "middle_state": "folded", "ring_state": "folded", "pinky_state": "extended"}
    }

    data = []
    labels = []

    print("[Dataset Generator] Building distinct class definitions...")
    for label, config in gesture_definitions.items():
        base = generate_hand_pose(**config)
        samples = generate_variations(base, num_samples=180, noise_level=0.02)
        for s in samples:
            data.append(s)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
    dataset_file = os.path.join(os.path.dirname(__file__), "..", "data", "landmarks_dataset.pickle")

    with open(dataset_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"[Dataset Generator] Successfully generated {len(data)} samples for {len(gesture_definitions)} classes.")


if __name__ == "__main__":
    generate_all()
