from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import snntorch as snn
import torch
import torch.nn.functional as F

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)


def make_raw_inputs(
    input_strength: float,
    steps: int = 30,
    input_start: int = 5,
    input_duration: int = 4,
) -> torch.Tensor:
    """Create a 1D time series with a constant input pulse."""
    raw_inputs = torch.zeros(steps)
    raw_inputs[input_start:input_start + input_duration] = input_strength
    return raw_inputs


def apply_sum_buffer(raw_inputs: torch.Tensor, buffer_size: int) -> torch.Tensor:
    """Replace each input with the sum of the most recent buffer_size inputs."""
    buffered_inputs = torch.zeros_like(raw_inputs)

    for t in range(len(raw_inputs)):
        start = max(0, t - buffer_size + 1)
        buffered_inputs[t] = raw_inputs[start:t + 1].sum()

    return buffered_inputs


def apply_truncated_decay_buffer(
    raw_inputs: torch.Tensor,
    buffer_size: int = 4,
    decay: float = 0.8,
) -> torch.Tensor:
    """Apply a finite memory buffer where older inputs decay geometrically."""
    buffered_inputs = torch.zeros_like(raw_inputs)

    for t in range(len(raw_inputs)):
        total = 0.0

        for k in range(buffer_size):
            idx = t - k
            if idx < 0:
                break

            total += (decay ** k) * raw_inputs[idx]

        buffered_inputs[t] = total

    return buffered_inputs


def apply_buffer(
    raw_inputs: torch.Tensor,
    buffer_type: int | str | None,
    decay: float = 0.8,
) -> torch.Tensor:
    """Dispatch to no buffer, an integer sum buffer, or a decayed buffer."""
    if buffer_type is None:
        return raw_inputs.clone()

    if isinstance(buffer_type, int):
        return apply_sum_buffer(raw_inputs, buffer_type)

    if buffer_type in {"decay", "decay4"}:
        return apply_truncated_decay_buffer(raw_inputs, buffer_size=4, decay=decay)

    raise ValueError(f"Unknown buffer type: {buffer_type}")


def did_spike(
    input_strength: float,
    buffer_type: int | str | None,
    beta: float = 0.9,
    steps: int = 30,
    input_start: int = 5,
    input_duration: int = 4,
    decay: float = 0.8,
) -> int:
    """Return 1 if a single LIF neuron spikes for this input setup, else 0."""
    lif = snn.Leaky(beta=beta)
    mem = torch.zeros(1)
    raw_inputs = make_raw_inputs(input_strength, steps, input_start, input_duration)
    lif_inputs = apply_buffer(raw_inputs, buffer_type, decay=decay)

    for t in range(len(lif_inputs)):
        current = lif_inputs[t].reshape(1)
        spk, mem = lif(current, mem)
        if spk.item() == 1:
            return 1

    return 0


def first_spike_threshold(
    input_strengths: torch.Tensor,
    buffer_type: int | str | None,
    **did_spike_kwargs: Any,
) -> float | None:
    """Return the first input strength that causes a spike, or None if none do."""
    for input_strength in input_strengths:
        spike = did_spike(float(input_strength), buffer_type, **did_spike_kwargs)
        if spike == 1:
            return float(input_strength)

    return None


def make_moving_1d_object(
    length: int = 12,
    object_width: int = 3,
    steps: int = 6,
    speed: int = 2,
    intensity: float = 1.0,
    start_pos: int = 0,
) -> torch.Tensor:
    """Create frames for a bright 1D object moving across spatial positions."""
    frames = torch.zeros(steps, length)

    for t in range(steps):
        pos = start_pos + speed * t
        end = min(pos + object_width, length)

        if pos < length:
            frames[t, pos:end] = intensity

    return frames


def make_moving_1d_object_from_edge_fully_visible(
    length: int = 24,
    object_width: int = 3,
    steps: int = 8,
    edge: str = "left",
    speed: int = 2,
    intensity: float = 0.30,
) -> torch.Tensor:
    """Create a moving 1D object that starts fully visible at either edge."""
    frames = torch.zeros(steps, length)

    if edge == "left":
        start_pos = 0
        signed_speed = abs(speed)
    elif edge == "right":
        start_pos = length - object_width
        signed_speed = -abs(speed)
    else:
        raise ValueError("edge must be 'left' or 'right'")

    for t in range(steps):
        pos = int(start_pos + signed_speed * t)

        start = max(0, pos)
        end = min(length, pos + object_width)

        if end > start:
            frames[t, start:end] = intensity

    return frames


def make_moving_1d_pattern(
    pattern: torch.Tensor | list[float] | list[int],
    length: int = 32,
    steps: int = 10,
    start_pos: int = 0,
    speed: int = 6,
    intensity: float = 0.30,
) -> torch.Tensor:
    """Create frames for a moving 1D binary pattern."""
    frames = torch.zeros(steps, length)
    pattern = torch.as_tensor(pattern, dtype=torch.float32)

    for t in range(steps):
        pos = int(start_pos + speed * t)

        for i, value in enumerate(pattern):
            x = pos + i

            if 0 <= x < length and value > 0:
                frames[t, x] = intensity * value

    return frames


def make_moving_1d_pattern_from_edge_fully_visible(
    pattern: torch.Tensor | list[float] | list[int],
    length: int = 32,
    steps: int = 10,
    edge: str = "left",
    speed: int = 6,
    intensity: float = 0.30,
) -> torch.Tensor:
    """Create a moving 1D pattern that starts fully visible at either edge."""
    pattern_width = len(pattern)

    if edge == "left":
        start_pos = 0
        signed_speed = abs(speed)
    elif edge == "right":
        start_pos = length - pattern_width
        signed_speed = -abs(speed)
    else:
        raise ValueError("edge must be 'left' or 'right'")

    return make_moving_1d_pattern(
        pattern=pattern,
        length=length,
        steps=steps,
        start_pos=start_pos,
        speed=signed_speed,
        intensity=intensity,
    )


def make_toy_2d_shape_templates() -> dict[str, torch.Tensor]:
    """Return small binary 2D templates for toy moving-shape experiments."""
    return {
        "seven": torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ], dtype=torch.float32),
        "ell": torch.tensor([
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ], dtype=torch.float32),
        "tee": torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32),
        "box": torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ], dtype=torch.float32),
    }


def place_shape_on_canvas(
    shape: torch.Tensor,
    canvas_height: int = 32,
    canvas_width: int = 32,
    top: int = 0,
    left: int = 0,
    intensity: float = 1.0,
) -> torch.Tensor:
    """Place a binary 2D shape on a canvas, clipping outside the field of view."""
    canvas = torch.zeros(canvas_height, canvas_width)
    shape = torch.as_tensor(shape)
    shape_h, shape_w = shape.shape

    for y in range(shape_h):
        for x in range(shape_w):
            if shape[y, x] > 0:
                canvas_y = top + y
                canvas_x = left + x

                if 0 <= canvas_y < canvas_height and 0 <= canvas_x < canvas_width:
                    canvas[canvas_y, canvas_x] = intensity

    return canvas


def make_moving_2d_shape(
    shape: torch.Tensor,
    canvas_height: int = 32,
    canvas_width: int = 32,
    steps: int = 8,
    start_top: int = 12,
    start_left: int = 0,
    velocity_y: float = 0,
    velocity_x: float = 6,
    intensity: float = 1.0,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Generate frames for a 2D shape moving at a constant velocity."""
    frames = []
    origins = []

    for t in range(steps):
        top = int(round(start_top + velocity_y * t))
        left = int(round(start_left + velocity_x * t))

        frame = place_shape_on_canvas(
            shape=shape,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            top=top,
            left=left,
            intensity=intensity,
        )

        frames.append(frame)
        origins.append((top, left))

    return torch.stack(frames), origins


def apply_frame_buffer(frames: torch.Tensor, buffer_size: int) -> torch.Tensor:
    """Sum recent frames independently at each spatial position."""
    buffered = torch.zeros_like(frames)

    for t in range(frames.shape[0]):
        start = max(0, t - buffer_size + 1)
        buffered[t] = frames[start:t + 1].sum(dim=0)

    return buffered


def apply_decay_frame_buffer(
    frames: torch.Tensor,
    buffer_size: int = 4,
    decay: float = 0.7,
) -> torch.Tensor:
    """Apply a finite decaying temporal buffer to each spatial position."""
    buffered = torch.zeros_like(frames)

    for t in range(frames.shape[0]):
        total = torch.zeros(frames.shape[1])

        for k in range(buffer_size):
            idx = t - k
            if idx < 0:
                break

            total += (decay ** k) * frames[idx]

        buffered[t] = total

    return buffered


def shift_1d(frame: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift a 1D frame; values shifted outside the frame are discarded."""
    shifted = torch.zeros_like(frame)

    if shift > 0:
        shifted[shift:] = frame[:-shift]
    elif shift < 0:
        shifted[:shift] = frame[-shift:]
    else:
        shifted = frame.clone()

    return shifted


def apply_motion_compensated_buffer(
    frames: torch.Tensor,
    buffer_size: int = 4,
    assumed_speed: int = 3,
    decay: float | None = None,
) -> torch.Tensor:
    """Buffer past frames after shifting them to the expected current position."""
    buffered = torch.zeros_like(frames)

    for t in range(frames.shape[0]):
        total = torch.zeros_like(frames[0])

        for k in range(buffer_size):
            idx = t - k
            if idx < 0:
                break

            shift = assumed_speed * k
            aligned_frame = shift_1d(frames[idx], shift)
            weight = 1.0 if decay is None else decay ** k
            total += weight * aligned_frame

        buffered[t] = total

    return buffered


def plot_spikes(spk_rec: torch.Tensor, title: str) -> None:
    """Plot a [time, position] binary spike raster as an image."""
    plt.figure(figsize=(10, 3))
    plt.imshow(spk_rec.detach().cpu().numpy(), aspect="auto", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Time")
    plt.colorbar(label="Spike")
    plt.show()


def plot_frame_matrix(
    matrix: torch.Tensor,
    title: str,
    colorbar_label: str = "Value",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a [time, position] matrix with time on the y-axis."""
    plt.figure(figsize=(10, 3))
    plt.imshow(
        matrix.detach().cpu().numpy() if torch.is_tensor(matrix) else matrix,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Time")
    plt.colorbar(label=colorbar_label)
    plt.show()


def plot_image(
    image: torch.Tensor,
    title: str = "",
    colorbar_label: str = "Value",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot one 2D image with x/y axes."""
    plt.figure(figsize=(4, 4))
    plt.imshow(
        image.detach().cpu().numpy() if torch.is_tensor(image) else image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label=colorbar_label)
    plt.show()


def plot_frame_sequence(
    frames: torch.Tensor,
    title: str = "",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Plot a sequence of 2D frames side-by-side."""
    num_frames = frames.shape[0]

    fig, axes = plt.subplots(
        1,
        num_frames,
        figsize=(2.2 * num_frames, 3),
    )

    if num_frames == 1:
        axes = [axes]

    for t, ax in enumerate(axes):
        ax.imshow(
            frames[t].detach().cpu().numpy(),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"t={t}")
        ax.axis("off")

    fig.suptitle(title)
    plt.show()


def run_lif_layer(input_frames: torch.Tensor, beta: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one LIF neuron per spatial position over a [time, position] input."""
    lif = snn.Leaky(beta=beta)
    mem = torch.zeros(input_frames.shape[1])

    spk_rec = []
    mem_rec = []

    for t in range(input_frames.shape[0]):
        current = input_frames[t]
        spk, mem = lif(current, mem)

        spk_rec.append(spk.clone())
        mem_rec.append(mem.clone())

    return torch.stack(spk_rec), torch.stack(mem_rec)


def summarize_spikes(frames: torch.Tensor, spk_rec: torch.Tensor, name: str) -> dict[str, Any]:
    """Compute simple spike-count, stale-spike, and duplicate-position metrics."""
    total_spikes = int(spk_rec.sum().item())
    stale_spikes = int(((spk_rec == 1) & (frames == 0)).sum().item())
    spikes_per_position = spk_rec.sum(dim=0)
    duplicate_positions = int((spikes_per_position > 1).sum().item())

    spike_times = torch.where(spk_rec == 1)[0]
    if len(spike_times) > 0:
        first_spike_time = int(spike_times.min().item())
    else:
        first_spike_time = None

    return {
        "method": name,
        "total_spikes": total_spikes,
        "stale_spikes": stale_spikes,
        "duplicate_positions": duplicate_positions,
        "first_spike_time": first_spike_time,
    }


def summarize_local_spike_delays(
    frames: torch.Tensor,
    spk_rec: torch.Tensor,
    name: str,
) -> dict[str, Any]:
    """Summarize how long after raw evidence each spike occurs."""
    delays = []
    events = []

    for t in range(spk_rec.shape[0]):
        for pos in range(spk_rec.shape[1]):
            if spk_rec[t, pos] == 1:
                active_times = torch.where(frames[:, pos] > 0)[0]
                past_active_times = active_times[active_times <= t]

                if len(past_active_times) == 0:
                    delay = None
                else:
                    last_active_time = int(past_active_times.max().item())
                    delay = t - last_active_time
                    delays.append(delay)

                events.append({
                    "time": t,
                    "position": pos,
                    "delay": delay,
                })

    return {
        "method": name,
        "num_spikes": len(events),
        "mean_delay": sum(delays) / len(delays) if delays else None,
        "max_delay": max(delays) if delays else None,
        "events": events,
    }


def summarize_spatial_error(
    frames: torch.Tensor,
    spk_rec: torch.Tensor,
    name: str,
) -> dict[str, Any]:
    """Summarize how far each spike is from the current raw object position."""
    errors = []
    events = []

    for t in range(spk_rec.shape[0]):
        current_object_positions = torch.where(frames[t] > 0)[0]

        for pos in torch.where(spk_rec[t] == 1)[0]:
            pos = int(pos.item())

            if len(current_object_positions) == 0:
                spatial_error = None
            else:
                spatial_error = int(torch.min(torch.abs(current_object_positions - pos)).item())
                errors.append(spatial_error)

            events.append({
                "time": t,
                "position": pos,
                "spatial_error": spatial_error,
            })

    return {
        "method": name,
        "num_spikes": len(events),
        "mean_spatial_error": sum(errors) / len(errors) if errors else None,
        "max_spatial_error": max(errors) if errors else None,
        "events": events,
    }


def evaluate_predicted_shapes(
    frames: torch.Tensor,
    predicted_shapes: list[list[int]],
) -> dict[str, Any]:
    """Evaluate predicted object positions against every frame."""
    rows = []

    for t in range(frames.shape[0]):
        true_positions = set(torch.where(frames[t] > 0)[0].tolist())
        pred_positions = set(predicted_shapes[t])

        if len(true_positions) == 0 and len(pred_positions) == 0:
            coverage = None
            precision = None
            exact_match = True
        elif len(true_positions) == 0:
            coverage = None
            precision = 0.0
            exact_match = False
        else:
            correct = true_positions & pred_positions
            coverage = len(correct) / len(true_positions)
            precision = 0.0 if len(pred_positions) == 0 else len(correct) / len(pred_positions)
            exact_match = true_positions == pred_positions

        rows.append({
            "time": t,
            "true_positions": sorted(list(true_positions)),
            "predicted_positions": sorted(list(pred_positions)),
            "coverage": coverage,
            "precision": precision,
            "exact_match": exact_match,
        })

    valid_coverages = [
        row["coverage"]
        for row in rows
        if row["coverage"] is not None
    ]
    valid_precisions = [
        row["precision"]
        for row in rows
        if row["precision"] is not None
    ]

    return {
        "mean_coverage": sum(valid_coverages) / len(valid_coverages) if valid_coverages else None,
        "mean_precision": sum(valid_precisions) / len(valid_precisions) if valid_precisions else None,
        "num_exact_matches": sum(row["exact_match"] for row in rows),
        "events": rows,
    }


def evaluate_predicted_shapes_visible_only(
    frames: torch.Tensor,
    predicted_shapes: list[list[int]],
    expected_width: int | None = None,
) -> dict[str, Any]:
    """Evaluate predictions only on frames where the object is visible or full-width."""
    rows = []

    for t in range(frames.shape[0]):
        true_positions = set(torch.where(frames[t] > 0)[0].tolist())
        pred_positions = set(predicted_shapes[t])

        if expected_width is not None:
            true_fully_visible = len(true_positions) == expected_width
        else:
            true_fully_visible = len(true_positions) > 0

        if not true_fully_visible:
            continue

        correct = true_positions & pred_positions
        coverage = len(correct) / len(true_positions)
        precision = 0.0 if len(pred_positions) == 0 else len(correct) / len(pred_positions)
        exact_match = true_positions == pred_positions

        rows.append({
            "time": t,
            "true_positions": sorted(list(true_positions)),
            "predicted_positions": sorted(list(pred_positions)),
            "coverage": coverage,
            "precision": precision,
            "exact_match": exact_match,
        })

    if len(rows) == 0:
        return {
            "mean_coverage": None,
            "mean_precision": None,
            "num_exact_matches": 0,
            "num_evaluated_frames": 0,
            "events": rows,
        }

    return {
        "mean_coverage": sum(row["coverage"] for row in rows) / len(rows),
        "mean_precision": sum(row["precision"] for row in rows) / len(rows),
        "num_exact_matches": sum(row["exact_match"] for row in rows),
        "num_evaluated_frames": len(rows),
        "events": rows,
    }


def summarize_frame_shape_recovery(
    frames: torch.Tensor,
    spk_rec: torch.Tensor,
    name: str,
) -> dict[str, Any]:
    """Summarize whether spikes recover the visible object shape in each frame."""
    frame_events = []

    for t in range(frames.shape[0]):
        true_positions = set(torch.where(frames[t] > 0)[0].tolist())
        spike_positions = set(torch.where(spk_rec[t] == 1)[0].tolist())

        if len(true_positions) == 0:
            coverage = None
            missed_positions = []
            correct_positions = []
        else:
            correct_positions = sorted(list(true_positions & spike_positions))
            missed_positions = sorted(list(true_positions - spike_positions))
            coverage = len(correct_positions) / len(true_positions)

        extra_positions = sorted(list(spike_positions - true_positions))

        frame_events.append({
            "time": t,
            "true_positions": sorted(list(true_positions)),
            "spike_positions": sorted(list(spike_positions)),
            "correct_positions": correct_positions,
            "missed_positions": missed_positions,
            "extra_positions": extra_positions,
            "coverage": coverage,
            "num_extra_spikes": len(extra_positions),
        })

    valid_coverages = [
        event["coverage"]
        for event in frame_events
        if event["coverage"] is not None
    ]

    max_coverage = max(valid_coverages) if valid_coverages else None
    frame_shape_success = any(
        (event["coverage"] == 1.0) and (event["num_extra_spikes"] == 0)
        for event in frame_events
        if event["coverage"] is not None
    )

    return {
        "method": name,
        "max_frame_coverage": max_coverage,
        "frame_shape_success": frame_shape_success,
        "events": frame_events,
    }


def run_method_and_summarize(
    frames: torch.Tensor,
    method_name: str,
    speed: int,
    buffer_size: int = 4,
    beta: float = 0.9,
    decay: float = 0.9,
) -> dict[str, Any]:
    """Run one buffering method and return spatial and shape-recovery metrics."""
    if method_name == "raw":
        input_frames = frames
    elif method_name == "sum4":
        input_frames = apply_frame_buffer(frames, buffer_size=buffer_size)
    elif method_name == "decay4":
        input_frames = apply_decay_frame_buffer(
            frames,
            buffer_size=buffer_size,
            decay=decay,
        )
    elif method_name == "motion_sum4":
        input_frames = apply_motion_compensated_buffer(
            frames,
            buffer_size=buffer_size,
            assumed_speed=speed,
            decay=None,
        )
    elif method_name == "motion_decay4":
        input_frames = apply_motion_compensated_buffer(
            frames,
            buffer_size=buffer_size,
            assumed_speed=speed,
            decay=decay,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    spk_rec, _ = run_lif_layer(input_frames, beta=beta)
    spatial_summary = summarize_spatial_error(frames, spk_rec, name=method_name)
    shape_summary = summarize_frame_shape_recovery(frames, spk_rec, name=method_name)

    return {
        "method": method_name,
        "num_spikes": spatial_summary["num_spikes"],
        "mean_spatial_error": spatial_summary["mean_spatial_error"],
        "max_spatial_error": spatial_summary["max_spatial_error"],
        "max_frame_coverage": shape_summary["max_frame_coverage"],
        "frame_shape_success": shape_summary["frame_shape_success"],
    }


def extract_spike_shapes(spk_rec: torch.Tensor) -> list[list[int]]:
    """Return sorted spike positions at each timestep."""
    shapes = []

    for t in range(spk_rec.shape[0]):
        positions = torch.where(spk_rec[t] == 1)[0].tolist()
        shapes.append(sorted(positions))

    return shapes


def extract_spike_positions_by_time(spk_rec: torch.Tensor) -> list[list[int]]:
    """Return spike positions at each timestep."""
    return extract_spike_shapes(spk_rec)


def shift_positions_int(positions: list[int], shift: int, length: int) -> list[int]:
    """Shift integer positions while dropping values outside [0, length)."""
    shifted = []

    for pos in positions:
        new_pos = int(round(pos + shift))

        if 0 <= new_pos < length:
            shifted.append(new_pos)

    return sorted(list(set(shifted)))


def shift_positions(positions: list[int], shift: float, length: int) -> list[int]:
    """Shift positions by a possibly non-integer amount, rounding to integers."""
    shifted = []

    for pos in positions:
        new_pos = int(round(pos + shift))

        if 0 <= new_pos < length:
            shifted.append(new_pos)

    return sorted(list(set(shifted)))


def shapes_to_frame_matrix(shapes: list[list[int]], length: int) -> torch.Tensor:
    """Convert a list of position lists into a [time, position] binary matrix."""
    matrix = torch.zeros(len(shapes), length)

    for t, positions in enumerate(shapes):
        for pos in positions:
            if 0 <= pos < length:
                matrix[t, pos] = 1.0

    return matrix


def positions_by_time_to_matrix(
    positions_by_time: list[list[int]],
    length: int,
) -> torch.Tensor:
    """Convert position lists by time into a [time, position] binary matrix."""
    return shapes_to_frame_matrix(positions_by_time, length=length)


def compact_pattern_from_positions(positions: list[int]) -> list[int] | None:
    """Convert active positions into a compact binary pattern."""
    positions = sorted(list(set(positions)))

    if len(positions) == 0:
        return None

    start = min(positions)
    end = max(positions)

    return [
        1 if x in positions else 0
        for x in range(start, end + 1)
    ]


def safe_hamming_similarity(
    a: list[int] | torch.Tensor | None,
    b: list[int] | torch.Tensor,
) -> float | None:
    """Return Hamming similarity, or None when patterns cannot be compared."""
    if a is None:
        return None

    if len(a) != len(b):
        return None

    a_tensor = torch.as_tensor(a).int()
    b_tensor = torch.as_tensor(b).int()

    return (a_tensor == b_tensor).float().mean().item()


def classify_compact_pattern_safe(
    recovered_pattern: list[int] | torch.Tensor | None,
    templates: dict[str, torch.Tensor],
    min_score: float = 0.75,
) -> tuple[str, dict[str, float]]:
    """Classify a compact pattern without forcing weak or partial evidence."""
    if recovered_pattern is None:
        return "unknown", {}

    scores = {}

    for name, template in templates.items():
        score = safe_hamming_similarity(recovered_pattern, template)

        if score is not None:
            scores[name] = score

    if len(scores) == 0:
        return "unknown", scores

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]

    if best_score < min_score:
        return "unknown", scores

    return best_name, scores


def full_pattern_observations_from_spikes(
    spike_positions_by_time: list[list[int]],
    pattern_templates: dict[str, torch.Tensor],
    min_score: float = 0.99,
) -> list[dict[str, Any]]:
    """Extract rows where spikes contain a full recognized pattern."""
    observations = []

    for t, positions in enumerate(spike_positions_by_time):
        compact = compact_pattern_from_positions(positions)

        label, scores = classify_compact_pattern_safe(
            compact,
            pattern_templates,
            min_score=min_score,
        )

        if label == "unknown":
            continue

        template = pattern_templates[label]

        if compact is None or len(compact) != len(template):
            continue

        observations.append({
            "time": t,
            "label": label,
            "origin": min(positions),
            "positions": sorted(positions),
            "compact": compact,
            "score": scores[label],
        })

    return observations


def estimate_speed_from_full_pattern_observations(
    observations: list[dict[str, Any]],
) -> int | None:
    """Estimate integer pattern speed from full delayed pattern observations."""
    if len(observations) < 2:
        return None

    speeds = []

    for obs0, obs1 in zip(observations[:-1], observations[1:]):
        if obs0["label"] != obs1["label"]:
            continue

        dt = obs1["time"] - obs0["time"]

        if dt <= 0:
            continue

        dx = obs1["origin"] - obs0["origin"]
        speeds.append(dx / dt)

    if len(speeds) == 0:
        return None

    speeds_tensor = torch.tensor(speeds)
    return int(round(float(torch.median(speeds_tensor).item())))


def predict_current_positions_from_full_spike_patterns(
    spike_positions_by_time: list[list[int]],
    estimated_speed: int | None,
    delay: int,
    length: int,
    pattern_templates: dict[str, torch.Tensor],
    min_score: float = 0.99,
) -> list[list[int]]:
    """Predict current positions from delayed rows with full recognized patterns."""
    predicted_positions_by_time = []

    for positions in spike_positions_by_time:
        compact = compact_pattern_from_positions(positions)

        label, _ = classify_compact_pattern_safe(
            compact,
            pattern_templates,
            min_score=min_score,
        )

        if label == "unknown" or estimated_speed is None:
            predicted_positions_by_time.append([])
            continue

        template = pattern_templates[label]

        if compact is None or len(compact) != len(template):
            predicted_positions_by_time.append([])
            continue

        predicted = shift_positions(
            positions,
            shift=estimated_speed * delay,
            length=length,
        )

        predicted_positions_by_time.append(predicted)

    return predicted_positions_by_time


def estimate_integer_speed_from_spike_shapes(
    spike_shapes: list[list[int]],
    length: int,
    max_speed: int = 10,
) -> int | None:
    """Estimate constant integer speed by matching shifted nonempty spike rows."""
    candidates = list(range(-max_speed, max_speed + 1))
    scores = {v: 0 for v in candidates}

    nonempty = [
        (t, positions)
        for t, positions in enumerate(spike_shapes)
        if len(positions) > 0
    ]

    if len(nonempty) < 2:
        return None

    for (t0, p0), (t1, p1) in zip(nonempty[:-1], nonempty[1:]):
        dt = t1 - t0
        p1_set = set(p1)

        for v in candidates:
            shifted = shift_positions_int(
                p0,
                shift=v * dt,
                length=length,
            )

            scores[v] += len(set(shifted) & p1_set)

    best_speed = max(scores, key=scores.get)

    if scores[best_speed] == 0:
        return None

    return best_speed


def reconstruct_delayed_shapes_overlap_aware(
    spike_shapes: list[list[int]],
    length: int,
    speed: int | None,
) -> tuple[list[list[int]], str]:
    """Reconstruct delayed shapes from hard-buffer spike rows with overlap handling."""
    reconstructed_shapes = [[] for _ in spike_shapes]

    if speed is None:
        return spike_shapes.copy(), "unknown_speed"

    nonempty_indices = [
        t for t, positions in enumerate(spike_shapes)
        if len(positions) > 0
    ]

    if len(nonempty_indices) == 0:
        return reconstructed_shapes, "no_spikes"

    if len(nonempty_indices) == 1:
        t0 = nonempty_indices[0]
        reconstructed_shapes[t0] = spike_shapes[t0]
        return reconstructed_shapes, "single_spike_row"

    first_t = nonempty_indices[0]
    second_t = nonempty_indices[1]

    first_shape = spike_shapes[first_t]
    second_shape = spike_shapes[second_t]

    dt = second_t - first_t

    shifted_first = shift_positions_int(
        first_shape,
        shift=speed * dt,
        length=length,
    )

    shifted_first_set = set(shifted_first)
    second_set = set(second_shape)

    overlap_with_shifted = shifted_first_set & second_set
    residual_second = sorted(list(second_set - shifted_first_set))

    overlap_mode = (
        len(first_shape) > 0
        and len(overlap_with_shifted) > 0
        and len(residual_second) > 0
    )

    if not overlap_mode:
        for t in nonempty_indices:
            reconstructed_shapes[t] = spike_shapes[t]

        return reconstructed_shapes, "non_overlap_complete_rows"

    carried_early_fragment = first_shape

    for t in nonempty_indices[1:]:
        current_shape = spike_shapes[t]
        current_set = set(current_shape)

        expected_next_early_fragment = shift_positions_int(
            carried_early_fragment,
            shift=speed,
            length=length,
        )

        expected_next_set = set(expected_next_early_fragment)
        late_fragment = sorted(list(current_set - expected_next_set))

        if len(late_fragment) == 0:
            reconstructed_shapes[t] = current_shape
            carried_early_fragment = current_shape
        else:
            reconstructed_shapes[t] = sorted(
                list(set(carried_early_fragment + late_fragment))
            )
            carried_early_fragment = expected_next_early_fragment

    return reconstructed_shapes, "overlap_fragmented_rows"


def delayed_shape_tracker_overlap_aware(
    frames: torch.Tensor,
    buffer_size: int = 4,
    beta: float = 0.9,
    max_speed: int = 10,
) -> dict[str, Any]:
    """Track current shape/location from delayed hard-buffer spikes."""
    delay = buffer_size - 1
    length = frames.shape[1]

    buffered = apply_frame_buffer(frames, buffer_size=buffer_size)
    spk_rec, mem_rec = run_lif_layer(buffered, beta=beta)

    raw_spike_shapes = extract_spike_shapes(spk_rec)

    estimated_speed = estimate_integer_speed_from_spike_shapes(
        raw_spike_shapes,
        length=length,
        max_speed=max_speed,
    )

    reconstructed_shapes, reconstruction_mode = reconstruct_delayed_shapes_overlap_aware(
        raw_spike_shapes,
        length=length,
        speed=estimated_speed,
    )

    predicted_shapes = []

    for positions in reconstructed_shapes:
        if estimated_speed is None or len(positions) == 0:
            predicted_shapes.append([])
        else:
            predicted_shapes.append(
                shift_positions_int(
                    positions,
                    shift=estimated_speed * delay,
                    length=length,
                )
            )

    predicted_frame_matrix = shapes_to_frame_matrix(predicted_shapes, length=length)
    reconstructed_frame_matrix = shapes_to_frame_matrix(
        reconstructed_shapes,
        length=length,
    )

    return {
        "buffered": buffered,
        "spk_rec": spk_rec,
        "mem_rec": mem_rec,
        "raw_spike_shapes": raw_spike_shapes,
        "estimated_speed": estimated_speed,
        "reconstructed_shapes": reconstructed_shapes,
        "reconstructed_frame_matrix": reconstructed_frame_matrix,
        "reconstruction_mode": reconstruction_mode,
        "predicted_shapes": predicted_shapes,
        "predicted_frame_matrix": predicted_frame_matrix,
        "delay": delay,
    }


def run_pattern_pipeline(
    pattern_name: str,
    pattern: torch.Tensor,
    patterns: dict[str, torch.Tensor],
    length: int = 32,
    steps: int = 10,
    edge: str = "left",
    speed: int = 6,
    intensity: float = 0.30,
    buffer_size: int = 4,
    beta: float = 0.9,
    min_score: float = 0.99,
) -> dict[str, Any]:
    """Run the moving-pattern buffer/spike/prediction workflow."""
    delay = buffer_size - 1

    frames = make_moving_1d_pattern_from_edge_fully_visible(
        pattern=pattern,
        length=length,
        steps=steps,
        edge=edge,
        speed=speed,
        intensity=intensity,
    )

    buffered = apply_frame_buffer(frames, buffer_size=buffer_size)
    spk_rec, mem_rec = run_lif_layer(buffered, beta=beta)
    spike_positions_by_time = extract_spike_positions_by_time(spk_rec)

    full_observations = full_pattern_observations_from_spikes(
        spike_positions_by_time,
        pattern_templates=patterns,
        min_score=min_score,
    )

    estimated_speed = estimate_speed_from_full_pattern_observations(
        full_observations
    )

    predicted_positions_by_time = predict_current_positions_from_full_spike_patterns(
        spike_positions_by_time,
        estimated_speed=estimated_speed,
        delay=delay,
        length=length,
        pattern_templates=patterns,
        min_score=min_score,
    )

    predicted_matrix = positions_by_time_to_matrix(
        predicted_positions_by_time,
        length=length,
    )

    return {
        "pattern_name": pattern_name,
        "pattern": pattern,
        "frames": frames,
        "buffered": buffered,
        "spk_rec": spk_rec,
        "mem_rec": mem_rec,
        "spike_positions_by_time": spike_positions_by_time,
        "full_observations": full_observations,
        "estimated_speed": estimated_speed,
        "predicted_positions_by_time": predicted_positions_by_time,
        "predicted_matrix": predicted_matrix,
        "delay": delay,
    }


def print_pattern_position_comparison(
    result: dict[str, Any],
    patterns: dict[str, torch.Tensor],
) -> None:
    """Print true, spiking, and predicted positions for a pattern result."""
    frames = result["frames"]
    spike_positions_by_time = result["spike_positions_by_time"]
    predicted_positions_by_time = result["predicted_positions_by_time"]

    print("=" * 90)
    print("PATTERN:", result["pattern_name"])
    print("template:", result["pattern"].tolist())
    print("estimated speed:", result["estimated_speed"])
    print("delay:", result["delay"])
    print("=" * 90)

    for t in range(frames.shape[0]):
        true_positions = torch.where(frames[t] > 0)[0].tolist()
        spike_positions = spike_positions_by_time[t]
        predicted_positions = predicted_positions_by_time[t]

        true_compact = compact_pattern_from_positions(true_positions)
        spike_compact = compact_pattern_from_positions(spike_positions)
        pred_compact = compact_pattern_from_positions(predicted_positions)

        spike_label, spike_scores = classify_compact_pattern_safe(
            spike_compact,
            patterns,
            min_score=0.75,
        )

        pred_label, pred_scores = classify_compact_pattern_safe(
            pred_compact,
            patterns,
            min_score=0.75,
        )

        exact_match = set(true_positions) == set(predicted_positions)

        print(f"t={t}")
        print("true positions:      ", true_positions)
        print("spike positions:     ", spike_positions)
        print("predicted positions: ", predicted_positions)
        print("exact match:         ", exact_match)
        print("true compact:        ", true_compact)
        print("spike compact:       ", spike_compact)
        print("pred compact:        ", pred_compact)
        print("spike classified as: ", spike_label, spike_scores)
        print("pred classified as:  ", pred_label, pred_scores)
        print("-" * 90)


def evaluate_pattern_result(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate predicted pattern positions against the true frame positions."""
    frames = result["frames"]
    predicted_positions_by_time = result["predicted_positions_by_time"]
    rows = []

    for t in range(frames.shape[0]):
        true_positions = set(torch.where(frames[t] > 0)[0].tolist())
        pred_positions = set(predicted_positions_by_time[t])
        exact_match = true_positions == pred_positions

        if len(true_positions) == 0 and len(pred_positions) == 0:
            coverage = None
            precision = None
        elif len(true_positions) == 0:
            coverage = None
            precision = 0.0
        else:
            correct = true_positions & pred_positions
            coverage = len(correct) / len(true_positions)
            precision = 0.0 if len(pred_positions) == 0 else len(correct) / len(pred_positions)

        rows.append({
            "time": t,
            "exact_match": exact_match,
            "coverage": coverage,
            "precision": precision,
            "true_count": len(true_positions),
            "pred_count": len(pred_positions),
        })

    valid_coverage = [
        row["coverage"]
        for row in rows
        if row["coverage"] is not None
    ]

    valid_precision = [
        row["precision"]
        for row in rows
        if row["precision"] is not None
    ]

    return {
        "num_exact_matches": sum(row["exact_match"] for row in rows),
        "mean_coverage": sum(valid_coverage) / len(valid_coverage) if valid_coverage else None,
        "mean_precision": sum(valid_precision) / len(valid_precision) if valid_precision else None,
        "events": rows,
    }


def apply_consecutive_repetition_penalty_buffer(
    frames,
    buffer_size=4,
    repeat_penalty=0.5,
):
    """
    Hard buffer, but if the same position was active in the previous frame,
    reduce the current contribution at that position.

    This is less aggressive than penalizing all repeats in the whole buffer.
    """
    adjusted_frames = torch.zeros_like(frames)

    for t in range(frames.shape[0]):
        if t == 0:
            adjusted_frames[t] = frames[t]
            continue

        repeated = (frames[t] > 0) & (frames[t - 1] > 0)

        adjusted_frames[t] = frames[t].clone()
        adjusted_frames[t][repeated] *= repeat_penalty

    buffered = apply_frame_buffer(
        adjusted_frames,
        buffer_size=buffer_size,
    )

    return buffered


def directional_kernel_2d(
    length: int = 7,
    sigma_along: float = 1.5,
    sigma_across: float = 0.35,
    direction: str = "horizontal",
) -> torch.Tensor:
    """Create an anisotropic Gaussian kernel for horizontal or vertical blur."""
    ax = torch.arange(length) - length // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")

    if direction == "horizontal":
        kernel = torch.exp(
            -(xx ** 2) / (2 * sigma_along ** 2)
            - (yy ** 2) / (2 * sigma_across ** 2)
        )
    elif direction == "vertical":
        kernel = torch.exp(
            -(yy ** 2) / (2 * sigma_along ** 2)
            - (xx ** 2) / (2 * sigma_across ** 2)
        )
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    return kernel / kernel.sum()


def apply_directional_blur_to_frames(
    frames: torch.Tensor,
    velocity_y: float = 0,
    velocity_x: float = 6,
    kernel_length: int = 7,
    sigma_along: float = 1.6,
    sigma_across: float = 0.35,
    gain: float = 1.0,
) -> torch.Tensor:
    """Blur frames mostly along the dominant motion axis."""
    if abs(velocity_x) >= abs(velocity_y):
        direction = "horizontal"
    else:
        direction = "vertical"

    kernel = directional_kernel_2d(
        length=kernel_length,
        sigma_along=sigma_along,
        sigma_across=sigma_across,
        direction=direction,
    )
    kernel = kernel.to(device=frames.device, dtype=frames.dtype)
    kernel = kernel.view(1, 1, kernel_length, kernel_length)

    frames_4d = frames.unsqueeze(1)
    blurred = F.conv2d(
        frames_4d,
        kernel,
        padding=kernel_length // 2,
    )

    return gain * blurred.squeeze(1)

def run_lif_2d_layer(
    input_frames,
    beta=0.9,
    threshold=0.75,
):
    """
    Run one LIF neuron per pixel.

    input_frames: [time, height, width]
    """
    mem = torch.zeros_like(input_frames[0])

    spk_rec = []
    mem_rec = []

    for t in range(input_frames.shape[0]):
        mem = beta * mem + input_frames[t]

        spk = (mem >= threshold).float()

        # Subtractive reset.
        mem = mem - spk * threshold

        spk_rec.append(spk)
        mem_rec.append(mem.clone())

    return torch.stack(spk_rec), torch.stack(mem_rec)


def apply_2d_frame_buffer(frames: torch.Tensor, buffer_size: int = 4) -> torch.Tensor:
    """Sum recent 2D frames independently at each pixel."""
    buffered = torch.zeros_like(frames)

    for t in range(frames.shape[0]):
        start = max(0, t - buffer_size + 1)
        buffered[t] = frames[start:t + 1].sum(dim=0)

    return buffered


def shift_2d_frame(
    frame: torch.Tensor,
    shift_y: float = 0,
    shift_x: float = 0,
) -> torch.Tensor:
    """Shift a 2D frame by rounded pixel offsets, dropping values outside view."""
    shift_y = int(round(shift_y))
    shift_x = int(round(shift_x))
    height, width = frame.shape
    shifted = torch.zeros_like(frame)

    src_y_start = max(0, -shift_y)
    src_y_end = min(height, height - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = min(height, height + shift_y)

    src_x_start = max(0, -shift_x)
    src_x_end = min(width, width - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(width, width + shift_x)

    if src_y_end <= src_y_start or src_x_end <= src_x_start:
        return shifted

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = frame[
        src_y_start:src_y_end,
        src_x_start:src_x_end,
    ]

    return shifted


def apply_motion_compensated_2d_buffer(
    frames: torch.Tensor,
    buffer_size: int = 4,
    velocity_y: float = 0,
    velocity_x: float = 6,
) -> torch.Tensor:
    """Sum recent 2D frames after shifting them to the current object location."""
    compensated = torch.zeros_like(frames)

    for t in range(frames.shape[0]):
        acc = torch.zeros_like(frames[t])

        for k in range(buffer_size):
            source_t = t - k

            if source_t < 0:
                continue

            acc += shift_2d_frame(
                frames[source_t],
                shift_y=velocity_y * k,
                shift_x=velocity_x * k,
            )

        compensated[t] = acc

    return compensated


def crop_patch(
    frame: torch.Tensor,
    top: int,
    left: int,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Crop a fixed-size patch, zero-filling regions outside the frame."""
    patch = torch.zeros(
        patch_height,
        patch_width,
        dtype=frame.dtype,
        device=frame.device,
    )

    for y in range(patch_height):
        for x in range(patch_width):
            source_y = top + y
            source_x = left + x

            if 0 <= source_y < frame.shape[0] and 0 <= source_x < frame.shape[1]:
                patch[y, x] = frame[source_y, source_x]

    return patch


def template_energy_stats(
    patch: torch.Tensor,
    template: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Measure how much patch energy lands on template versus background pixels."""
    patch = torch.clamp(patch, min=0)
    template = template.to(device=patch.device, dtype=patch.dtype)

    object_energy = patch[template > 0].sum()
    background_energy = patch[template == 0].sum()
    total_energy = patch.sum()
    energy_fraction = object_energy / (total_energy + eps)

    return {
        "object_energy": object_energy.item(),
        "background_energy": background_energy.item(),
        "total_energy": total_energy.item(),
        "energy_fraction": energy_fraction.item(),
    }


def template_energy_fraction_with_mass(
    patch: torch.Tensor,
    template: torch.Tensor,
    reference_energy: float,
    eps: float = 1e-8,
) -> float:
    """Score template alignment while penalizing patches with too little mass."""
    stats = template_energy_stats(patch, template, eps=eps)
    energy_fraction = stats["energy_fraction"]
    total_energy = stats["total_energy"]
    mass_factor = min(total_energy / (reference_energy + eps), 1.0)

    return energy_fraction * mass_factor


def score_template_at_location_with_mass(
    frame: torch.Tensor,
    top: int,
    left: int,
    template: torch.Tensor,
    reference_energy: float,
) -> float:
    """Score template evidence at one top-left location in a frame."""
    patch_h, patch_w = template.shape
    patch = crop_patch(
        frame,
        top=top,
        left=left,
        patch_height=patch_h,
        patch_width=patch_w,
    )

    return template_energy_fraction_with_mass(
        patch,
        template,
        reference_energy=reference_energy,
    )


def template_score_map_offsets_with_mass(
    frame: torch.Tensor,
    true_top: int,
    true_left: int,
    template: torch.Tensor,
    offsets: list[int],
    reference_energy: float,
    axis: str = "x",
) -> pd.DataFrame:
    """Score template matches at offsets from the true object location."""
    rows = []

    for offset in offsets:
        if axis == "x":
            top = true_top
            left = true_left + offset
        elif axis == "y":
            top = true_top + offset
            left = true_left
        else:
            raise ValueError("axis must be 'x' or 'y'")

        score = score_template_at_location_with_mass(
            frame,
            top=top,
            left=left,
            template=template,
            reference_energy=reference_energy,
        )

        patch = crop_patch(
            frame,
            top=top,
            left=left,
            patch_height=template.shape[0],
            patch_width=template.shape[1],
        )

        stats = template_energy_stats(patch, template)

        rows.append({
            "offset": offset,
            "score": score,
            "total_energy": stats["total_energy"],
            "energy_fraction": stats["energy_fraction"],
            "object_energy": stats["object_energy"],
            "background_energy": stats["background_energy"],
        })

    return pd.DataFrame(rows)


def template_score_map_x_offsets_with_mass(
    frame: torch.Tensor,
    true_top: int,
    true_left: int,
    template: torch.Tensor,
    x_offsets: list[int],
    reference_energy: float,
) -> pd.DataFrame:
    """Compatibility wrapper for scoring horizontal offsets only."""
    scores = template_score_map_offsets_with_mass(
        frame,
        true_top=true_top,
        true_left=true_left,
        template=template,
        offsets=x_offsets,
        reference_energy=reference_energy,
        axis="x",
    )

    return scores.rename(columns={"offset": "x_offset"})


def localization_summary_generic(
    offset_scores: pd.DataFrame,
    offset_column: str = "offset",
) -> dict[str, float | int]:
    """Summarize whether the best localization score is at offset zero."""
    best_row = offset_scores.loc[offset_scores["score"].idxmax()]
    score_at_zero = offset_scores[
        offset_scores[offset_column] == 0
    ]["score"].iloc[0]
    nonzero_scores = offset_scores[
        offset_scores[offset_column] != 0
    ]["score"]
    max_off_target = nonzero_scores.max()
    dominance = score_at_zero / (max_off_target + 1e-8)

    return {
        "best_offset": int(best_row[offset_column]),
        "best_score": float(best_row["score"]),
        "score_at_zero": float(score_at_zero),
        "max_off_target": float(max_off_target),
        "dominance": float(dominance),
    }


def localization_summary(offset_scores: pd.DataFrame) -> dict[str, float | int]:
    """Summarize horizontal localization scores with an x_offset column."""
    return localization_summary_generic(offset_scores, offset_column="x_offset")


def make_default_2d_motion_conditions(
    canvas_height: int,
    canvas_width: int,
    shape: torch.Tensor,
    speed: int = 6,
    start_top: int = 12,
    start_left: int = 12,
) -> dict[str, dict[str, int]]:
    """Create right/left/down/up motion-condition configs for one shape."""
    shape_height, shape_width = shape.shape

    return {
        "right_fast": {
            "velocity_y": 0,
            "velocity_x": speed,
            "start_top": start_top,
            "start_left": 0,
        },
        "left_fast": {
            "velocity_y": 0,
            "velocity_x": -speed,
            "start_top": start_top,
            "start_left": canvas_width - shape_width,
        },
        "down_fast": {
            "velocity_y": speed,
            "velocity_x": 0,
            "start_top": 0,
            "start_left": start_left,
        },
        "up_fast": {
            "velocity_y": -speed,
            "velocity_x": 0,
            "start_top": canvas_height - shape_height,
            "start_left": start_left,
        },
    }


def run_motion_condition(
    condition_name: str,
    velocity_y: float,
    velocity_x: float,
    start_top: int,
    start_left: int,
    shape: torch.Tensor,
    shape_name: str,
    canvas_height: int = 32,
    canvas_width: int = 64,
    steps: int = 5,
    intensity: float = 1.0,
    blur_kernel_length: int = 9,
    blur_sigma_along: float = 1.3,
    blur_sigma_across: float = 0.1,
    blur_gain: float = 1.2,
    lif_beta: float = 0.9,
    lif_threshold: float = 0.75,
    buffer_size: int = 4,
    t_plot: int = 4,
    offsets: list[int] | None = None,
    show_plots: bool = False,
) -> dict[str, Any]:
    """Run one 2D motion condition through blur, LIF, buffers, and scoring."""
    if offsets is None:
        offsets = list(range(-24, 25))

    frames, origins = make_moving_2d_shape(
        shape=shape,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        steps=steps,
        start_top=start_top,
        start_left=start_left,
        velocity_y=velocity_y,
        velocity_x=velocity_x,
        intensity=intensity,
    )

    input_current = apply_directional_blur_to_frames(
        frames,
        velocity_y=velocity_y,
        velocity_x=velocity_x,
        kernel_length=blur_kernel_length,
        sigma_along=blur_sigma_along,
        sigma_across=blur_sigma_across,
        gain=blur_gain,
    )

    spk_rec_2d, mem_rec_2d = run_lif_2d_layer(
        input_current,
        beta=lif_beta,
        threshold=lif_threshold,
    )

    mem_buffered = apply_2d_frame_buffer(mem_rec_2d, buffer_size=buffer_size)
    mem_motion_compensated = apply_motion_compensated_2d_buffer(
        mem_rec_2d,
        buffer_size=buffer_size,
        velocity_y=velocity_y,
        velocity_x=velocity_x,
    )

    true_top, true_left = origins[t_plot]
    true_patch_motion = crop_patch(
        mem_motion_compensated[t_plot],
        top=true_top,
        left=true_left,
        patch_height=shape.shape[0],
        patch_width=shape.shape[1],
    )
    reference_energy = true_patch_motion.sum().item()

    if abs(velocity_x) >= abs(velocity_y):
        axis = "x"
    else:
        axis = "y"

    naive_scores = template_score_map_offsets_with_mass(
        mem_buffered[t_plot],
        true_top=true_top,
        true_left=true_left,
        template=shape,
        offsets=offsets,
        reference_energy=reference_energy,
        axis=axis,
    )

    motion_comp_scores = template_score_map_offsets_with_mass(
        mem_motion_compensated[t_plot],
        true_top=true_top,
        true_left=true_left,
        template=shape,
        offsets=offsets,
        reference_energy=reference_energy,
        axis=axis,
    )

    result = {
        "condition": condition_name,
        "shape_name": shape_name,
        "shape": shape,
        "axis": axis,
        "velocity_y": velocity_y,
        "velocity_x": velocity_x,
        "origins": origins,
        "frames": frames,
        "input_current": input_current,
        "spikes": spk_rec_2d,
        "membrane": mem_rec_2d,
        "naive_buffer": mem_buffered,
        "motion_comp_buffer": mem_motion_compensated,
        "naive_scores": naive_scores,
        "motion_comp_scores": motion_comp_scores,
        "naive_summary": localization_summary_generic(naive_scores),
        "motion_comp_summary": localization_summary_generic(motion_comp_scores),
        "reference_energy": reference_energy,
        "buffer_size": buffer_size,
        "t_plot": t_plot,
    }

    if show_plots:
        plot_motion_condition_panel(result, t_plot=t_plot)
        plot_localization_curve(result)

    return result


def run_motion_conditions(
    motion_conditions: dict[str, dict[str, int]],
    shape: torch.Tensor,
    shape_name: str,
    verbose: bool = False,
    **run_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run all named motion conditions for a single shape."""
    results = {}

    for condition_name, params in motion_conditions.items():
        result = run_motion_condition(
            condition_name=condition_name,
            velocity_y=params["velocity_y"],
            velocity_x=params["velocity_x"],
            start_top=params["start_top"],
            start_left=params["start_left"],
            shape=shape,
            shape_name=shape_name,
            **run_kwargs,
        )

        results[condition_name] = result

        if verbose:
            print_motion_result_summary(result)

    return results


def motion_results_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a summary table from motion-condition results."""
    rows = []

    for condition_name, result in results.items():
        naive = result["naive_summary"]
        motion = result["motion_comp_summary"]

        rows.append({
            "condition": condition_name,
            "axis": result["axis"],
            "velocity_y": result["velocity_y"],
            "velocity_x": result["velocity_x"],
            "naive_best_offset": naive["best_offset"],
            "naive_score_at_zero": naive["score_at_zero"],
            "naive_dominance": naive["dominance"],
            "motion_best_offset": motion["best_offset"],
            "motion_score_at_zero": motion["score_at_zero"],
            "motion_dominance": motion["dominance"],
        })

    return pd.DataFrame(rows)


def get_motion_result(
    shape_templates: dict[str, torch.Tensor],
    motion_conditions: dict[str, dict[str, int]],
    shape_name: str = "seven",
    condition_name: str = "right_fast",
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Run one shape and one named motion condition."""
    shape = shape_templates[shape_name]
    params = motion_conditions[condition_name]

    return run_motion_condition(
        condition_name=condition_name,
        velocity_y=params["velocity_y"],
        velocity_x=params["velocity_x"],
        start_top=params["start_top"],
        start_left=params["start_left"],
        shape=shape,
        shape_name=shape_name,
        **run_kwargs,
    )


def plot_motion_condition_panel(
    result: dict[str, Any],
    t_plot: int | None = None,
    title: str | None = None,
) -> None:
    """Plot clean input, blurred input, spikes, membrane, and both buffers."""
    if t_plot is None:
        t_plot = result.get("t_plot", 4)

    frames = result["frames"]
    input_current = result["input_current"]
    spikes = result["spikes"]
    membrane = result["membrane"]
    naive_buffer = result["naive_buffer"]
    motion_comp_buffer = result["motion_comp_buffer"]

    condition = result["condition"]

    if title is None:
        title = f"{condition}, t={t_plot}"

    images = [
        (frames[t_plot], "clean shape"),
        (input_current[t_plot], "blurred input"),
        (spikes[t_plot], "pixel spikes"),
        (membrane[t_plot], "membrane"),
        (naive_buffer[t_plot], "naive mem buffer"),
        (motion_comp_buffer[t_plot], "motion-comp mem buffer"),
    ]

    vmax_candidates = [
        image.max().item()
        for image, _ in images
        if image.max().item() > 0
    ]
    vmax = max(vmax_candidates) if vmax_candidates else 1.0

    fig, axes = plt.subplots(1, len(images), figsize=(18, 4))

    for ax, (image, label) in zip(axes, images):
        ax.imshow(
            image.detach().cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    plt.show()


def plot_localization_curve(
    result: dict[str, Any],
    title: str | None = None,
) -> None:
    """Plot naive versus motion-compensated localization scores."""
    naive_scores = result["naive_scores"]
    motion_comp_scores = result["motion_comp_scores"]
    condition = result["condition"]
    axis = result["axis"]

    if title is None:
        title = f"Mass-weighted localization: {condition}"

    plt.figure(figsize=(8, 4))
    plt.plot(
        naive_scores["offset"],
        naive_scores["score"],
        marker="o",
        label="naive membrane buffer",
    )
    plt.plot(
        motion_comp_scores["offset"],
        motion_comp_scores["score"],
        marker="o",
        label="motion-comp membrane buffer",
    )
    plt.axvline(0, linestyle="--", label="true location")

    if axis == "x":
        plt.axvline(
            result["velocity_x"],
            linestyle=":",
            label="next-frame location",
        )
        plt.xlabel("x offset from true object location")
    else:
        plt.axvline(
            result["velocity_y"],
            linestyle=":",
            label="next-frame location",
        )
        plt.xlabel("y offset from true object location")

    plt.ylabel("Template score with energy mass")
    plt.title(title)
    plt.legend()
    plt.show()


def spike_count_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize total and per-frame spike counts for a motion result."""
    spikes = result["spikes"]
    spike_counts = spikes.flatten(start_dim=1).sum(dim=1)

    return {
        "total_spikes": int(spike_counts.sum().item()),
        "spikes_per_time": [int(x.item()) for x in spike_counts],
        "max_spikes_in_frame": int(spike_counts.max().item()),
    }


def print_motion_result_summary(result: dict[str, Any]) -> None:
    """Print spike and localization summaries for one motion result."""
    print("=" * 80)
    print("condition:", result["condition"])
    print("axis:", result["axis"])
    print("velocity:", (result["velocity_y"], result["velocity_x"]))
    print()
    print("Spike summary:")
    print(spike_count_summary(result))
    print()
    print("Naive:")
    print(result["naive_summary"])
    print()
    print("Motion-compensated:")
    print(result["motion_comp_summary"])


def inspect_motion_condition(
    shape_templates: dict[str, torch.Tensor],
    motion_conditions: dict[str, dict[str, int]],
    shape_name: str = "seven",
    condition_name: str = "right_fast",
    t_plot: int = 4,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Run, summarize, and plot one shape/motion condition."""
    result = get_motion_result(
        shape_templates=shape_templates,
        motion_conditions=motion_conditions,
        shape_name=shape_name,
        condition_name=condition_name,
        t_plot=t_plot,
        **run_kwargs,
    )

    print_motion_result_summary(result)
    plot_motion_condition_panel(
        result,
        t_plot=t_plot,
        title=f"{shape_name}: {condition_name}, t={t_plot}",
    )
    plot_localization_curve(
        result,
        title=f"{shape_name}: {condition_name} localization",
    )

    return result


def plot_input_current_over_time(
    result: dict[str, Any],
    title: str | None = None,
    vmin: float = 0,
    vmax: float | None = None,
) -> None:
    """Plot blurred input current frames for a motion result."""
    input_current = result["input_current"]
    condition = result["condition"]

    if title is None:
        title = f"Blurred input current over time: {condition}"

    if vmax is None:
        vmax = input_current.max().item()

    plot_frame_sequence(
        input_current,
        title=title,
        vmin=vmin,
        vmax=vmax,
    )


def spike_summary_across_shapes(
    shape_templates: dict[str, torch.Tensor],
    motion_conditions: dict[str, dict[str, int]],
    **run_kwargs: Any,
) -> pd.DataFrame:
    """Build spike-count rows for every shape and motion condition."""
    rows = []

    for shape_name in shape_templates:
        for condition_name in motion_conditions:
            result = get_motion_result(
                shape_templates=shape_templates,
                motion_conditions=motion_conditions,
                shape_name=shape_name,
                condition_name=condition_name,
                **run_kwargs,
            )
            spike_summary = spike_count_summary(result)

            rows.append({
                "shape": shape_name,
                "condition": condition_name,
                "velocity_y": result["velocity_y"],
                "velocity_x": result["velocity_x"],
                "total_spikes": spike_summary["total_spikes"],
                "max_spikes_in_frame": spike_summary["max_spikes_in_frame"],
                "spikes_per_time": spike_summary["spikes_per_time"],
            })

    return pd.DataFrame(rows)


def combined_motion_summary_across_shapes(
    shape_templates: dict[str, torch.Tensor],
    motion_conditions: dict[str, dict[str, int]],
    **run_kwargs: Any,
) -> pd.DataFrame:
    """Summarize spikes and localization for every shape and motion condition."""
    rows = []

    for shape_name in shape_templates:
        for condition_name in motion_conditions:
            result = get_motion_result(
                shape_templates=shape_templates,
                motion_conditions=motion_conditions,
                shape_name=shape_name,
                condition_name=condition_name,
                **run_kwargs,
            )
            spike_summary = spike_count_summary(result)
            naive = result["naive_summary"]
            motion = result["motion_comp_summary"]

            rows.append({
                "shape": shape_name,
                "condition": condition_name,
                "velocity_y": result["velocity_y"],
                "velocity_x": result["velocity_x"],
                "total_spikes": spike_summary["total_spikes"],
                "naive_best_offset": naive["best_offset"],
                "naive_score_at_zero": naive["score_at_zero"],
                "naive_dominance": naive["dominance"],
                "motion_best_offset": motion["best_offset"],
                "motion_score_at_zero": motion["score_at_zero"],
                "motion_dominance": motion["dominance"],
            })

    return pd.DataFrame(rows)


def apply_motion_compensated_2d_buffer_with_estimated_velocity(
    frames: torch.Tensor,
    buffer_size: int = 4,
    estimated_velocity_y: float = 0,
    estimated_velocity_x: float = 6,
) -> torch.Tensor:
    """Motion-compensate frames using a candidate, possibly wrong, velocity."""
    return apply_motion_compensated_2d_buffer(
        frames,
        buffer_size=buffer_size,
        velocity_y=estimated_velocity_y,
        velocity_x=estimated_velocity_x,
    )


def scene_level_template_score(
    frame: torch.Tensor,
    true_top: int,
    true_left: int,
    template: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Measure how much total frame energy lies on the true template location."""
    patch = crop_patch(
        frame,
        top=true_top,
        left=true_left,
        patch_height=template.shape[0],
        patch_width=template.shape[1],
    )
    template = template.to(device=frame.device, dtype=frame.dtype)
    true_object_energy = patch[template > 0].sum()
    total_frame_energy = torch.clamp(frame, min=0).sum()

    return (true_object_energy / (total_frame_energy + eps)).item()


def run_velocity_sensitivity_experiment(
    shape_name: str,
    condition_name: str,
    shape_templates: dict[str, torch.Tensor],
    motion_conditions: dict[str, dict[str, int]],
    estimated_velocities: list[int] | None = None,
    canvas_height: int = 64,
    canvas_width: int = 64,
    steps: int = 5,
    t_plot: int = 4,
    intensity: float = 1.0,
    blur_kernel_length: int = 9,
    blur_sigma_along: float = 1.3,
    blur_sigma_across: float = 0.1,
    blur_gain: float = 1.2,
    lif_beta: float = 0.9,
    lif_threshold: float = 0.75,
    buffer_size: int = 4,
) -> dict[str, Any]:
    """Test how reconstruction changes as the compensation velocity varies."""
    shape = shape_templates[shape_name]
    params = motion_conditions[condition_name]

    true_velocity_y = params["velocity_y"]
    true_velocity_x = params["velocity_x"]

    if abs(true_velocity_x) >= abs(true_velocity_y):
        axis = "x"
        true_velocity = true_velocity_x
    else:
        axis = "y"
        true_velocity = true_velocity_y

    if estimated_velocities is None:
        estimated_velocities = velocity_candidates_for_condition(params)

    frames, origins = make_moving_2d_shape(
        shape=shape,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        steps=steps,
        start_top=params["start_top"],
        start_left=params["start_left"],
        velocity_y=true_velocity_y,
        velocity_x=true_velocity_x,
        intensity=intensity,
    )

    input_current = apply_directional_blur_to_frames(
        frames,
        velocity_y=true_velocity_y,
        velocity_x=true_velocity_x,
        kernel_length=blur_kernel_length,
        sigma_along=blur_sigma_along,
        sigma_across=blur_sigma_across,
        gain=blur_gain,
    )

    spikes, membrane = run_lif_2d_layer(
        input_current,
        beta=lif_beta,
        threshold=lif_threshold,
    )

    true_top, true_left = origins[t_plot]

    base_result = {
        "shape_name": shape_name,
        "condition": condition_name,
        "frames": frames,
        "input_current": input_current,
        "spikes": spikes,
        "membrane": membrane,
        "origins": origins,
        "axis": axis,
        "true_velocity": true_velocity,
        "t_plot": t_plot,
    }

    rows = []
    compensated_frames_by_velocity = {}

    for estimated_velocity in estimated_velocities:
        if axis == "x":
            estimated_velocity_y = 0
            estimated_velocity_x = estimated_velocity
        else:
            estimated_velocity_y = estimated_velocity
            estimated_velocity_x = 0

        mem_compensated = apply_motion_compensated_2d_buffer_with_estimated_velocity(
            membrane,
            buffer_size=buffer_size,
            estimated_velocity_y=estimated_velocity_y,
            estimated_velocity_x=estimated_velocity_x,
        )
        compensated_frames_by_velocity[estimated_velocity] = mem_compensated

        scene_score = scene_level_template_score(
            mem_compensated[t_plot],
            true_top=true_top,
            true_left=true_left,
            template=shape,
        )

        rows.append({
            "shape": shape_name,
            "condition": condition_name,
            "axis": axis,
            "true_velocity": true_velocity,
            "estimated_velocity": estimated_velocity,
            "velocity_error": estimated_velocity - true_velocity,
            "scene_score": scene_score,
        })

    return {
        "base_result": base_result,
        "sensitivity_df": pd.DataFrame(rows),
        "compensated_frames_by_velocity": compensated_frames_by_velocity,
    }


def velocity_candidates_for_condition(
    condition: dict[str, int],
    positive_candidates: list[int] | None = None,
    negative_candidates: list[int] | None = None,
) -> list[int]:
    """Return candidate scalar compensation speeds for a motion condition."""
    true_vy = condition["velocity_y"]
    true_vx = condition["velocity_x"]
    true_velocity = true_vx if abs(true_vx) >= abs(true_vy) else true_vy

    if positive_candidates is None:
        positive_candidates = [0, 6, 8, 10, 12, 16, 20]
    if negative_candidates is None:
        negative_candidates = [-20, -16, -12, -10, -8, -6, 0]

    return positive_candidates if true_velocity > 0 else negative_candidates


def run_velocity_sensitivity_by_condition(
    shape_name: str,
    shape_templates: dict[str, torch.Tensor],
    motion_conditions: dict[str, dict[str, int]],
    **experiment_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run velocity-sensitivity experiments for one shape across conditions."""
    results = {}
    explicit_estimated_velocities = experiment_kwargs.get("estimated_velocities")
    base_kwargs = {
        key: value
        for key, value in experiment_kwargs.items()
        if key != "estimated_velocities"
    }

    for condition_name, condition in motion_conditions.items():
        if explicit_estimated_velocities is None:
            estimated_velocities = velocity_candidates_for_condition(condition)
        else:
            estimated_velocities = explicit_estimated_velocities

        results[condition_name] = run_velocity_sensitivity_experiment(
            shape_name=shape_name,
            condition_name=condition_name,
            shape_templates=shape_templates,
            motion_conditions=motion_conditions,
            estimated_velocities=estimated_velocities,
            **base_kwargs,
        )

    return results


def run_velocity_sensitivity_across_shapes(
    shape_templates: dict[str, torch.Tensor],
    motion_conditions: dict[str, dict[str, int]],
    **experiment_kwargs: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run velocity-sensitivity experiments for every shape and condition."""
    return {
        shape_name: run_velocity_sensitivity_by_condition(
            shape_name=shape_name,
            shape_templates=shape_templates,
            motion_conditions=motion_conditions,
            **experiment_kwargs,
        )
        for shape_name in shape_templates
    }


def plot_velocity_scene_score_by_condition(
    velocity_sensitivity_by_condition: dict[str, dict[str, Any]],
) -> None:
    """Plot scene-level velocity sensitivity curves for several conditions."""
    plt.figure(figsize=(8, 5))

    for condition_name, sensitivity_result in velocity_sensitivity_by_condition.items():
        df = sensitivity_result["sensitivity_df"]
        plt.plot(
            df["estimated_velocity"],
            df["scene_score"],
            marker="o",
            label=condition_name,
        )

    plt.axvline(0, linestyle=":", label="zero compensation")
    plt.xlabel("Estimated compensation velocity")
    plt.ylabel("Scene score")
    plt.title("Velocity sensitivity across motion directions")
    plt.legend()
    plt.show()


def plot_compensated_velocity_examples(
    sensitivity_result: dict[str, Any],
    velocities_to_show: list[int] | None = None,
    t_plot: int = 4,
) -> None:
    """Plot compensated membrane examples for selected candidate velocities."""
    if velocities_to_show is None:
        velocities_to_show = [0, 6, 10, 12]

    frames_by_velocity = sensitivity_result["compensated_frames_by_velocity"]
    df = sensitivity_result["sensitivity_df"]
    shape = df["shape"].iloc[0]
    condition = df["condition"].iloc[0]

    images = [
        (frames_by_velocity[velocity][t_plot], f"est vel = {velocity}")
        for velocity in velocities_to_show
        if velocity in frames_by_velocity
    ]

    if len(images) == 0:
        raise ValueError("No requested velocities were found.")

    vmax_candidates = [
        image.max().item()
        for image, _ in images
        if image.max().item() > 0
    ]
    vmax = max(vmax_candidates) if vmax_candidates else 1.0

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(4 * len(images), 4),
    )

    if len(images) == 1:
        axes = [axes]

    for ax, (image, label) in zip(axes, images):
        ax.imshow(
            image.detach().cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(f"Velocity compensation examples: {shape}, {condition}, t={t_plot}")
    plt.show()


def make_ideal_compensated_target(
    sensitivity_result: dict[str, Any],
    t_plot: int = 4,
    buffer_size: int = 4,
) -> torch.Tensor:
    """Use true velocity on input current to build the ideal accumulated target."""
    base = sensitivity_result["base_result"]
    df = sensitivity_result["sensitivity_df"]
    axis = base["axis"]
    true_velocity = df["true_velocity"].iloc[0]

    if axis == "x":
        true_velocity_y = 0
        true_velocity_x = true_velocity
    else:
        true_velocity_y = true_velocity
        true_velocity_x = 0

    ideal_compensated_input = apply_motion_compensated_2d_buffer_with_estimated_velocity(
        base["input_current"],
        buffer_size=buffer_size,
        estimated_velocity_y=true_velocity_y,
        estimated_velocity_x=true_velocity_x,
    )

    return ideal_compensated_input[t_plot]


def make_binary_target_mask(
    target: torch.Tensor,
    threshold_fraction: float = 0.25,
) -> torch.Tensor:
    """Convert an ideal accumulated target into a binary target mask."""
    target = torch.clamp(target, min=0)
    threshold = threshold_fraction * target.max().clamp_min(1e-8)
    return (target >= threshold).float()


def make_soft_mask_from_binary_mask(
    binary_mask: torch.Tensor,
    sigma: float = 0.75,
    radius: int = 2,
) -> torch.Tensor:
    """Give full credit on target pixels and partial credit nearby."""
    size = 2 * radius + 1
    coords = torch.arange(
        size,
        dtype=binary_mask.dtype,
        device=binary_mask.device,
    ) - radius
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.max().clamp_min(1e-8)

    soft_mask = F.conv2d(
        binary_mask[None, None, :, :],
        kernel[None, None, :, :],
        padding=radius,
    )[0, 0]

    return soft_mask / soft_mask.max().clamp_min(1e-8)


def target_mask_energy_score(
    frame: torch.Tensor,
    target_mask: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Return the fraction of candidate energy that falls on a target mask."""
    frame = torch.clamp(frame, min=0)
    target_energy = (frame * target_mask).sum()
    total_energy = frame.sum()

    return (target_energy / (total_energy + eps)).item()


def compare_target_mask_scores(
    sensitivity_result: dict[str, Any],
    t_plot: int = 4,
    threshold_fraction: float = 0.25,
    sigma: float = 0.75,
    radius: int = 2,
    buffer_size: int = 4,
) -> pd.DataFrame:
    """Compare hard and soft target-mask scores across candidate velocities."""
    target = make_ideal_compensated_target(
        sensitivity_result,
        t_plot=t_plot,
        buffer_size=buffer_size,
    )
    binary_mask = make_binary_target_mask(
        target,
        threshold_fraction=threshold_fraction,
    )
    soft_mask = make_soft_mask_from_binary_mask(
        binary_mask,
        sigma=sigma,
        radius=radius,
    )

    rows = []

    for estimated_velocity, mem_compensated in sensitivity_result[
        "compensated_frames_by_velocity"
    ].items():
        frame = mem_compensated[t_plot]

        rows.append({
            "estimated_velocity": estimated_velocity,
            "hard_target_mask_score": target_mask_energy_score(frame, binary_mask),
            "soft_target_mask_score": target_mask_energy_score(frame, soft_mask),
        })

    return pd.DataFrame(rows).sort_values("estimated_velocity")


def plot_target_mask_score_by_condition(
    velocity_sensitivity_by_condition: dict[str, dict[str, Any]],
    score_column: str = "soft_target_mask_score",
    t_plot: int = 4,
    threshold_fraction: float = 0.25,
    sigma: float = 0.75,
    radius: int = 2,
    buffer_size: int = 4,
) -> None:
    """Plot target-mask scores for every condition in a sensitivity dict."""
    plt.figure(figsize=(8, 5))

    for condition_name, sensitivity_result in velocity_sensitivity_by_condition.items():
        df = compare_target_mask_scores(
            sensitivity_result,
            t_plot=t_plot,
            threshold_fraction=threshold_fraction,
            sigma=sigma,
            radius=radius,
            buffer_size=buffer_size,
        )
        plt.plot(
            df["estimated_velocity"],
            df[score_column],
            marker="o",
            label=condition_name,
        )

    plt.axvline(0, linestyle=":", label="zero compensation")
    plt.xlabel("Estimated compensation velocity")
    plt.ylabel(score_column)
    plt.title("Velocity sensitivity using ideal accumulated target mask")
    plt.legend()
    plt.show()


def ideal_target_match_and_residual(
    sensitivity_result: dict[str, Any],
    t_plot: int = 4,
    threshold_fraction: float = 0.25,
    buffer_size: int = 4,
) -> pd.DataFrame:
    """Measure target and residual energy against the ideal target mask."""
    target = make_ideal_compensated_target(
        sensitivity_result,
        t_plot=t_plot,
        buffer_size=buffer_size,
    )
    target_mask = make_binary_target_mask(
        target,
        threshold_fraction=threshold_fraction,
    )

    rows = []

    for estimated_velocity, mem_compensated in sensitivity_result[
        "compensated_frames_by_velocity"
    ].items():
        frame = torch.clamp(mem_compensated[t_plot], min=0)
        target_energy = (frame * target_mask).sum().item()
        residual_energy = (frame * (1.0 - target_mask)).sum().item()
        total_energy = frame.sum().item()

        rows.append({
            "estimated_velocity": estimated_velocity,
            "target_energy": target_energy,
            "residual_energy": residual_energy,
            "total_energy": total_energy,
            "target_over_residual": target_energy / (residual_energy + 1e-8),
            "target_minus_residual": target_energy - residual_energy,
        })

    return pd.DataFrame(rows).sort_values("estimated_velocity")


def weighted_target_residual_score(
    sensitivity_result: dict[str, Any],
    t_plot: int = 4,
    threshold_fraction: float = 0.25,
    residual_penalty: float = 0.05,
    buffer_size: int = 4,
) -> pd.DataFrame:
    """Score candidate velocities by target energy minus residual penalty."""
    target = make_ideal_compensated_target(
        sensitivity_result,
        t_plot=t_plot,
        buffer_size=buffer_size,
    )
    target_mask = make_binary_target_mask(
        target,
        threshold_fraction=threshold_fraction,
    )

    rows = []

    for estimated_velocity, mem_compensated in sensitivity_result[
        "compensated_frames_by_velocity"
    ].items():
        frame = torch.clamp(mem_compensated[t_plot], min=0)
        target_energy = (frame * target_mask).sum().item()
        residual_energy = (frame * (1.0 - target_mask)).sum().item()
        score = target_energy - residual_penalty * residual_energy

        rows.append({
            "estimated_velocity": estimated_velocity,
            "target_energy": target_energy,
            "residual_energy": residual_energy,
            "score": score,
        })

    return pd.DataFrame(rows).sort_values("estimated_velocity")


def summarize_weighted_metric_across_shapes(
    velocity_sensitivity_all_shapes: dict[str, dict[str, dict[str, Any]]],
    t_plot: int = 4,
    threshold_fraction: float = 0.25,
    residual_penalty: float = 0.05,
    buffer_size: int = 4,
) -> pd.DataFrame:
    """Summarize the winning weighted target-residual velocity per shape/condition."""
    rows = []

    for shape_name, condition_results in velocity_sensitivity_all_shapes.items():
        for condition_name, sensitivity_result in condition_results.items():
            df = weighted_target_residual_score(
                sensitivity_result,
                t_plot=t_plot,
                threshold_fraction=threshold_fraction,
                residual_penalty=residual_penalty,
                buffer_size=buffer_size,
            )
            true_velocity = sensitivity_result["sensitivity_df"]["true_velocity"].iloc[0]
            best_row = df.loc[df["score"].idxmax()]

            rows.append({
                "shape": shape_name,
                "condition": condition_name,
                "true_velocity": true_velocity,
                "best_estimated_velocity": best_row["estimated_velocity"],
                "velocity_error": best_row["estimated_velocity"] - true_velocity,
                "best_score": best_row["score"],
                "target_energy_at_best": best_row["target_energy"],
                "residual_energy_at_best": best_row["residual_energy"],
                "correct_velocity_wins": best_row["estimated_velocity"] == true_velocity,
            })

    return pd.DataFrame(rows)

# Notebook 09 motion/object-memory helpers

# Notebook 09 motion/object-memory helper defaults.
NOTEBOOK09_CANVAS_HEIGHT = 128
NOTEBOOK09_CANVAS_WIDTH = 128
NOTEBOOK09_STEPS = 7
NOTEBOOK09_INTENSITY = 1.0

CANVAS_HEIGHT = NOTEBOOK09_CANVAS_HEIGHT
CANVAS_WIDTH = NOTEBOOK09_CANVAS_WIDTH
STEPS = NOTEBOOK09_STEPS
INTENSITY = NOTEBOOK09_INTENSITY
SHAPE_TEMPLATES = make_toy_2d_shape_templates()
DIRECTION_CONFIG = {
    "right": {"velocity_y": 0, "velocity_x": 1, "start_top": 60, "start_left": 20},
    "left": {"velocity_y": 0, "velocity_x": -1, "start_top": 60, "start_left": 103},
    "down": {"velocity_y": 1, "velocity_x": 0, "start_top": 20, "start_left": 60},
    "up": {"velocity_y": -1, "velocity_x": 0, "start_top": 103, "start_left": 60},
}
GOOD_VELOCITY_PARAMS = {
    "threshold_velocity": 15.0,
    "input_scale": 1.0,
    "beta_velocity": 0.8,
    "inhibition_strength": 3.0,
    "inhibition_radius": 1,
    "internal_steps": 5,
}
def make_unblurred_spike_case(
    speed,
    direction_name,
    shape_name="seven",
):
    config = DIRECTION_CONFIG[direction_name]

    velocity_y = config["velocity_y"] * speed
    velocity_x = config["velocity_x"] * speed

    frames, origins = make_moving_2d_shape(
        shape=SHAPE_TEMPLATES[shape_name],
        canvas_height=CANVAS_HEIGHT,
        canvas_width=CANVAS_WIDTH,
        steps=STEPS,
        start_top=config["start_top"],
        start_left=config["start_left"],
        velocity_y=velocity_y,
        velocity_x=velocity_x,
        intensity=INTENSITY,
    )

    # Treat binary object pixels as spikes.
    spikes = (frames > 0).float()

    return {
        "shape_name": shape_name,
        "direction_name": direction_name,
        "speed": speed,
        "velocity_y": velocity_y,
        "velocity_x": velocity_x,
        "frames": frames,
        "spikes": spikes,
        "origins": origins,
    }

def make_displacement_candidates(
    speeds=range(1, 7),
    directions=("right", "left", "down", "up"),
):
    candidates = []

    for speed in speeds:
        if "right" in directions:
            candidates.append({"direction": "right", "speed": speed, "dy": 0, "dx": speed})
        if "left" in directions:
            candidates.append({"direction": "left", "speed": speed, "dy": 0, "dx": -speed})
        if "down" in directions:
            candidates.append({"direction": "down", "speed": speed, "dy": speed, "dx": 0})
        if "up" in directions:
            candidates.append({"direction": "up", "speed": speed, "dy": -speed, "dx": 0})

    return candidates

def score_candidate_displacement(
    previous_spikes,
    current_spikes,
    dy,
    dx,
):
    shifted_previous = shift_2d_frame(
        previous_spikes,
        shift_y=dy,
        shift_x=dx,
    )

    overlap = (shifted_previous * current_spikes).sum().item()

    previous_count = previous_spikes.sum().item()
    current_count = current_spikes.sum().item()

    union = ((shifted_previous + current_spikes) > 0).float().sum().item()

    if union > 0:
        iou = overlap / union
    else:
        iou = 0.0

    return {
        "overlap": overlap,
        "iou": iou,
        "previous_count": previous_count,
        "current_count": current_count,
    }

def predict_displacement_between_frames(
    previous_spikes,
    current_spikes,
    candidate_speeds=range(1, 7),
    candidate_directions=("right", "left", "down", "up"),
):
    candidates = make_displacement_candidates(
        speeds=candidate_speeds,
        directions=candidate_directions,
    )

    rows = []

    for candidate in candidates:
        score = score_candidate_displacement(
            previous_spikes=previous_spikes,
            current_spikes=current_spikes,
            dy=candidate["dy"],
            dx=candidate["dx"],
        )

        rows.append({
            "direction": candidate["direction"],
            "speed": candidate["speed"],
            "dy": candidate["dy"],
            "dx": candidate["dx"],
            **score,
        })

    scores = pd.DataFrame(rows)

    # Primary readout: overlap.
    # Tie-breaker: IoU.
    scores = scores.sort_values(
        ["overlap", "iou"],
        ascending=False,
    ).reset_index(drop=True)

    return scores

def test_spike_overlap_motion_estimator(
    shape_name="seven",
    speeds=range(1, 7),
    directions=("right", "left", "down", "up"),
    previous_t=5,
    current_t=6,
):
    rows = []

    for true_speed in speeds:
        for true_direction in directions:
            case = make_unblurred_spike_case(
                speed=true_speed,
                direction_name=true_direction,
                shape_name=shape_name,
            )

            scores = predict_displacement_between_frames(
                previous_spikes=case["spikes"][previous_t],
                current_spikes=case["spikes"][current_t],
                candidate_speeds=range(1, 7),
                candidate_directions=directions,
            )

            best = scores.iloc[0]

            rows.append({
                "shape": shape_name,
                "true_direction": true_direction,
                "true_speed": true_speed,
                "estimated_direction": best["direction"],
                "estimated_speed": best["speed"],
                "overlap": best["overlap"],
                "iou": best["iou"],
                "correct_direction": best["direction"] == true_direction,
                "correct_speed": best["speed"] == true_speed,
                "correct_exact": (
                    best["direction"] == true_direction
                    and best["speed"] == true_speed
                ),
            })

    return pd.DataFrame(rows)

def plot_input_next_to_spikes(case):
    frames = case["frames"]
    spikes = case["spikes"]

    num_steps = frames.shape[0]

    fig, axes = plt.subplots(2, num_steps, figsize=(2.5 * num_steps, 5))

    for t in range(num_steps):
        axes[0, t].imshow(frames[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"Input t={t}")
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])

        axes[1, t].imshow(spikes[t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(f"Spikes t={t}\ncount={int(spikes[t].sum().item())}")
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])

    plt.suptitle(
        f"Input vs spikes: {case['shape_name']}, {case['direction_name']}, speed {case['speed']}"
    )
    plt.tight_layout()
    plt.show()

def compute_on_off_difference_layer(previous_spikes, current_spikes):
    """
    Spike-only difference layer.

    previous_spikes: [H, W], binary
    current_spikes:  [H, W], binary

    Returns two separate spiking populations:
        ON[y, x]  = 1 if activity appeared at (y, x)
        OFF[y, x] = 1 if activity disappeared at (y, x)
    """

    previous_spikes = (previous_spikes > 0).float()
    current_spikes = (current_spikes > 0).float()

    on_spikes = ((previous_spikes == 0) & (current_spikes == 1)).float()
    off_spikes = ((previous_spikes == 1) & (current_spikes == 0)).float()
    same_spikes = ((previous_spikes == 1) & (current_spikes == 1)).float()

    return {
        "on_spikes": on_spikes,
        "off_spikes": off_spikes,
        "same_spikes": same_spikes,
    }

def plot_on_off_difference_layer(case, previous_t=5, current_t=6):
    previous_spikes = case["spikes"][previous_t]
    current_spikes = case["spikes"][current_t]

    diff = compute_on_off_difference_layer(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
    )

    signed_difference = diff["on_spikes"] - diff["off_spikes"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    axes[0].imshow(previous_spikes, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Previous spikes\nt={previous_t}")

    axes[1].imshow(current_spikes, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Current spikes\nt={current_t}")

    axes[2].imshow(diff["off_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"OFF population\ncount={int(diff['off_spikes'].sum().item())}")

    axes[3].imshow(diff["on_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"ON population\ncount={int(diff['on_spikes'].sum().item())}")

    axes[4].imshow(signed_difference, cmap="bwr", vmin=-1, vmax=1)
    axes[4].set_title("Signed view\nOFF=-1, ON=+1")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"ON/OFF difference layer: {case['shape_name']}, "
        f"{case['direction_name']}, speed {case['speed']}"
    )
    plt.tight_layout()
    plt.show()

def compute_motion_pair_layer(
    off_spikes,
    on_spikes,
    max_offset=6,
):
    """
    Motion-pair layer.

    For each spatial offset (dy, dx), this layer counts local OFF/ON pairs:

        OFF[y, x] and ON[y + dy, x + dx]

    This is a neural-style interpretation:
        each offset corresponds to a population of neurons
        with fixed spatial-offset wiring.

    Returns a DataFrame with one row per offset population.
    """

    height, width = off_spikes.shape

    rows = []

    off_positions = torch.nonzero(off_spikes > 0, as_tuple=False)

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            if dy == 0 and dx == 0:
                continue

            pair_count = 0

            for pos in off_positions:
                y = int(pos[0].item())
                x = int(pos[1].item())

                target_y = y + dy
                target_x = x + dx

                if 0 <= target_y < height and 0 <= target_x < width:
                    if on_spikes[target_y, target_x] > 0:
                        pair_count += 1

            rows.append({
                "dy": dy,
                "dx": dx,
                "pair_spike_count": pair_count,
                "offset_magnitude": (dy ** 2 + dx ** 2) ** 0.5,
            })

    motion_pair_scores = pd.DataFrame(rows)

    motion_pair_scores = motion_pair_scores.sort_values(
        ["pair_spike_count", "offset_magnitude"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return motion_pair_scores

def estimate_velocity_from_motion_pairs(
    previous_spikes,
    current_spikes,
    max_offset=6,
):
    """
    Estimate velocity from ON/OFF difference neurons
    and motion-pair populations.

    Output is a 2D velocity vector:
        estimated_dy, estimated_dx
    """

    diff = compute_on_off_difference_layer(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
    )

    motion_scores = compute_motion_pair_layer(
        off_spikes=diff["off_spikes"],
        on_spikes=diff["on_spikes"],
        max_offset=max_offset,
    )

    best = motion_scores.iloc[0]

    estimated_dy = int(best["dy"])
    estimated_dx = int(best["dx"])

    estimated_speed = (estimated_dy ** 2 + estimated_dx ** 2) ** 0.5

    if estimated_dx == 0 and estimated_dy == 0:
        estimated_direction = "none"
    elif abs(estimated_dx) >= abs(estimated_dy):
        if estimated_dx > 0:
            estimated_direction = "right" if estimated_dy == 0 else ("down-right" if estimated_dy > 0 else "up-right")
        else:
            estimated_direction = "left" if estimated_dy == 0 else ("down-left" if estimated_dy > 0 else "up-left")
    else:
        if estimated_dy > 0:
            estimated_direction = "down" if estimated_dx == 0 else ("down-right" if estimated_dx > 0 else "down-left")
        else:
            estimated_direction = "up" if estimated_dx == 0 else ("up-right" if estimated_dx > 0 else "up-left")

    return {
        "estimated_dy": estimated_dy,
        "estimated_dx": estimated_dx,
        "estimated_speed": estimated_speed,
        "estimated_direction": estimated_direction,
        "best_pair_spike_count": int(best["pair_spike_count"]),
        "difference_layer": diff,
        "motion_pair_scores": motion_scores,
    }

def plot_top_motion_pair_scores(result, top_n=15):
    scores = result["motion_pair_scores"].head(top_n).copy()

    labels = [
        f"dy={int(row.dy)}, dx={int(row.dx)}"
        for _, row in scores.iterrows()
    ]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, scores["pair_spike_count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Motion-pair spike count")
    plt.title("Top motion-pair offset populations")
    plt.tight_layout()
    plt.show()

def test_motion_pair_estimator(
    shape_name="seven",
    speeds=range(1, 7),
    directions=("right", "left", "down", "up"),
    previous_t=5,
    current_t=6,
    max_offset=6,
):
    rows = []

    for true_speed in speeds:
        for true_direction in directions:
            case = make_unblurred_spike_case(
                speed=true_speed,
                direction_name=true_direction,
                shape_name=shape_name,
            )

            result = estimate_velocity_from_motion_pairs(
                previous_spikes=case["spikes"][previous_t],
                current_spikes=case["spikes"][current_t],
                max_offset=max_offset,
            )

            rows.append({
                "shape": shape_name,
                "true_direction": true_direction,
                "true_speed": true_speed,
                "true_dy": case["velocity_y"],
                "true_dx": case["velocity_x"],
                "estimated_direction": result["estimated_direction"],
                "estimated_speed": result["estimated_speed"],
                "estimated_dy": result["estimated_dy"],
                "estimated_dx": result["estimated_dx"],
                "best_pair_spike_count": result["best_pair_spike_count"],
                "correct_dy": result["estimated_dy"] == case["velocity_y"],
                "correct_dx": result["estimated_dx"] == case["velocity_x"],
                "correct_vector": (
                    result["estimated_dy"] == case["velocity_y"]
                    and result["estimated_dx"] == case["velocity_x"]
                ),
            })

    return pd.DataFrame(rows)

def compare_box_speeds_difference_and_motion_pairs(
    speeds=(3, 4, 5),
    direction_name="right",
    previous_t=5,
    current_t=6,
    max_offset=6,
):
    for speed in speeds:
        case = make_unblurred_spike_case(
            speed=speed,
            direction_name=direction_name,
            shape_name="box",
        )

        result = estimate_velocity_from_motion_pairs(
            previous_spikes=case["spikes"][previous_t],
            current_spikes=case["spikes"][current_t],
            max_offset=max_offset,
        )

        diff = result["difference_layer"]
        signed_difference = diff["on_spikes"] - diff["off_spikes"]

        print("=" * 80)
        print(f"BOX | true direction={direction_name} | true speed={speed}")
        print(f"Estimated dy={result['estimated_dy']}, dx={result['estimated_dx']}")
        print(f"Estimated speed={result['estimated_speed']}")
        print(f"Estimated direction={result['estimated_direction']}")
        print(f"Best pair spike count={result['best_pair_spike_count']}")
        print()
        print("Top motion-pair offsets:")
        display(result["motion_pair_scores"].head(10))

        fig, axes = plt.subplots(1, 5, figsize=(16, 3))

        axes[0].imshow(case["spikes"][previous_t], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Previous spikes\nt={previous_t}")

        axes[1].imshow(case["spikes"][current_t], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Current spikes\nt={current_t}")

        axes[2].imshow(diff["off_spikes"], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"OFF\ncount={int(diff['off_spikes'].sum().item())}")

        axes[3].imshow(diff["on_spikes"], cmap="gray", vmin=0, vmax=1)
        axes[3].set_title(f"ON\ncount={int(diff['on_spikes'].sum().item())}")

        axes[4].imshow(signed_difference, cmap="bwr", vmin=-1, vmax=1)
        axes[4].set_title("Signed diff\nOFF=-1, ON=+1")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(
            f"Box speed {speed}, direction {direction_name}: difference layer"
        )
        plt.tight_layout()
        plt.show()

def compare_box_speeds_zoomed(
    speeds=(3, 4, 5),
    direction_name="right",
    previous_t=5,
    current_t=6,
    max_offset=6,
    padding=8,
):
    for speed in speeds:
        case = make_unblurred_spike_case(
            speed=speed,
            direction_name=direction_name,
            shape_name="box",
        )

        result = estimate_velocity_from_motion_pairs(
            previous_spikes=case["spikes"][previous_t],
            current_spikes=case["spikes"][current_t],
            max_offset=max_offset,
        )

        diff = result["difference_layer"]
        signed_difference = diff["on_spikes"] - diff["off_spikes"]

        previous_spikes = case["spikes"][previous_t]
        current_spikes = case["spikes"][current_t]

        combined = (
            previous_spikes
            + current_spikes
            + diff["off_spikes"]
            + diff["on_spikes"]
        )

        active_positions = torch.nonzero(combined > 0, as_tuple=False)

        y_min = max(0, int(active_positions[:, 0].min().item()) - padding)
        y_max = min(CANVAS_HEIGHT, int(active_positions[:, 0].max().item()) + padding + 1)
        x_min = max(0, int(active_positions[:, 1].min().item()) - padding)
        x_max = min(CANVAS_WIDTH, int(active_positions[:, 1].max().item()) + padding + 1)

        print("=" * 80)
        print(f"BOX | true direction={direction_name} | true speed={speed}")
        print(f"Estimated dy={result['estimated_dy']}, dx={result['estimated_dx']}")
        print(f"Estimated speed={result['estimated_speed']}")
        print(f"Estimated direction={result['estimated_direction']}")
        print(f"Best pair spike count={result['best_pair_spike_count']}")
        print()
        display(result["motion_pair_scores"].head(10))

        fig, axes = plt.subplots(1, 5, figsize=(16, 3))

        images = [
            previous_spikes,
            current_spikes,
            diff["off_spikes"],
            diff["on_spikes"],
            signed_difference,
        ]

        titles = [
            f"Previous\nt={previous_t}",
            f"Current\nt={current_t}",
            f"OFF\ncount={int(diff['off_spikes'].sum().item())}",
            f"ON\ncount={int(diff['on_spikes'].sum().item())}",
            "Signed diff\nOFF=-1, ON=+1",
        ]

        cmaps = ["gray", "gray", "gray", "gray", "bwr"]
        vmins = [0, 0, 0, 0, -1]
        vmaxs = [1, 1, 1, 1, 1]

        for i, ax in enumerate(axes):
            ax.imshow(
                images[i][y_min:y_max, x_min:x_max],
                cmap=cmaps[i],
                vmin=vmins[i],
                vmax=vmaxs[i],
            )
            ax.set_title(titles[i])
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(
            f"Zoomed box speed {speed}, direction {direction_name}"
        )
        plt.tight_layout()
        plt.show()

def compute_motion_pair_layer_with_same_as_activity(
    off_spikes,
    on_spikes,
    same_spikes,
    max_offset=6,
):
    """
    Motion-pair layer using ON, OFF, and SAME populations.

    Previous-side evidence:
        OFF + SAME

    Current-side evidence:
        ON + SAME

    For each offset (dy, dx), count:
        previous_evidence[y, x] and current_evidence[y + dy, x + dx]
    """

    previous_evidence = ((off_spikes > 0) | (same_spikes > 0)).float()
    current_evidence = ((on_spikes > 0) | (same_spikes > 0)).float()

    height, width = previous_evidence.shape
    rows = []

    previous_positions = torch.nonzero(previous_evidence > 0, as_tuple=False)

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            if dy == 0 and dx == 0:
                continue

            pair_count = 0

            for pos in previous_positions:
                y = int(pos[0].item())
                x = int(pos[1].item())

                target_y = y + dy
                target_x = x + dx

                if 0 <= target_y < height and 0 <= target_x < width:
                    if current_evidence[target_y, target_x] > 0:
                        pair_count += 1

            rows.append({
                "dy": dy,
                "dx": dx,
                "pair_spike_count": pair_count,
                "offset_magnitude": (dy ** 2 + dx ** 2) ** 0.5,
            })

    motion_pair_scores = pd.DataFrame(rows)

    motion_pair_scores = motion_pair_scores.sort_values(
        ["pair_spike_count", "offset_magnitude"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return motion_pair_scores

def estimate_velocity_from_motion_pairs_with_same_as_activity(
    previous_spikes,
    current_spikes,
    max_offset=6,
):
    diff = compute_on_off_difference_layer(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
    )

    motion_scores = compute_motion_pair_layer_with_same_as_activity(
        off_spikes=diff["off_spikes"],
        on_spikes=diff["on_spikes"],
        same_spikes=diff["same_spikes"],
        max_offset=max_offset,
    )

    best = motion_scores.iloc[0]

    estimated_dy = int(best["dy"])
    estimated_dx = int(best["dx"])
    estimated_speed = (estimated_dy ** 2 + estimated_dx ** 2) ** 0.5

    if estimated_dx == 0 and estimated_dy == 0:
        estimated_direction = "none"
    elif abs(estimated_dx) >= abs(estimated_dy):
        if estimated_dx > 0:
            estimated_direction = "right" if estimated_dy == 0 else ("down-right" if estimated_dy > 0 else "up-right")
        else:
            estimated_direction = "left" if estimated_dy == 0 else ("down-left" if estimated_dy > 0 else "up-left")
    else:
        if estimated_dy > 0:
            estimated_direction = "down" if estimated_dx == 0 else ("down-right" if estimated_dx > 0 else "down-left")
        else:
            estimated_direction = "up" if estimated_dx == 0 else ("up-right" if estimated_dx > 0 else "up-left")

    return {
        "estimated_dy": estimated_dy,
        "estimated_dx": estimated_dx,
        "estimated_speed": estimated_speed,
        "estimated_direction": estimated_direction,
        "best_pair_spike_count": int(best["pair_spike_count"]),
        "difference_layer": diff,
        "motion_pair_scores": motion_scores,
    }

def test_motion_pair_estimator_with_same_as_activity(
    shape_name="seven",
    speeds=range(1, 7),
    directions=("right", "left", "down", "up"),
    previous_t=5,
    current_t=6,
    max_offset=6,
):
    rows = []

    for true_speed in speeds:
        for true_direction in directions:
            case = make_unblurred_spike_case(
                speed=true_speed,
                direction_name=true_direction,
                shape_name=shape_name,
            )

            result = estimate_velocity_from_motion_pairs_with_same_as_activity(
                previous_spikes=case["spikes"][previous_t],
                current_spikes=case["spikes"][current_t],
                max_offset=max_offset,
            )

            rows.append({
                "shape": shape_name,
                "true_direction": true_direction,
                "true_speed": true_speed,
                "true_dy": case["velocity_y"],
                "true_dx": case["velocity_x"],
                "estimated_direction": result["estimated_direction"],
                "estimated_speed": result["estimated_speed"],
                "estimated_dy": result["estimated_dy"],
                "estimated_dx": result["estimated_dx"],
                "best_pair_spike_count": result["best_pair_spike_count"],
                "correct_dy": result["estimated_dy"] == case["velocity_y"],
                "correct_dx": result["estimated_dx"] == case["velocity_x"],
                "correct_vector": (
                    result["estimated_dy"] == case["velocity_y"]
                    and result["estimated_dx"] == case["velocity_x"]
                ),
            })

    return pd.DataFrame(rows)

def compare_original_vs_same_activity_for_case(
    shape_name="box",
    speed=4,
    direction_name="right",
    previous_t=5,
    current_t=6,
    max_offset=6,
):
    case = make_unblurred_spike_case(
        speed=speed,
        direction_name=direction_name,
        shape_name=shape_name,
    )

    original_result = estimate_velocity_from_motion_pairs(
        previous_spikes=case["spikes"][previous_t],
        current_spikes=case["spikes"][current_t],
        max_offset=max_offset,
    )

    same_activity_result = estimate_velocity_from_motion_pairs_with_same_as_activity(
        previous_spikes=case["spikes"][previous_t],
        current_spikes=case["spikes"][current_t],
        max_offset=max_offset,
    )

    diff = original_result["difference_layer"]

    print("=" * 80)
    print(f"Shape={shape_name}, direction={direction_name}, true speed={speed}")
    print()
    print("Original OFF-to-ON model:")
    print(
        "estimated dy:",
        original_result["estimated_dy"],
        "estimated dx:",
        original_result["estimated_dx"],
        "estimated speed:",
        original_result["estimated_speed"],
    )
    display(original_result["motion_pair_scores"].head(10))

    print()
    print("Improved OFF/SAME-to-ON/SAME model:")
    print(
        "estimated dy:",
        same_activity_result["estimated_dy"],
        "estimated dx:",
        same_activity_result["estimated_dx"],
        "estimated speed:",
        same_activity_result["estimated_speed"],
    )
    display(same_activity_result["motion_pair_scores"].head(10))

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    images = [
        case["spikes"][previous_t],
        case["spikes"][current_t],
        diff["off_spikes"],
        diff["on_spikes"],
        diff["same_spikes"],
    ]

    titles = [
        f"Previous spikes\nt={previous_t}",
        f"Current spikes\nt={current_t}",
        "OFF",
        "ON",
        "SAME",
    ]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"Why SAME fixes {shape_name}, speed {speed}, direction {direction_name}"
    )
    plt.tight_layout()
    plt.show()

def make_unblurred_spike_case_from_velocity(
    velocity_y,
    velocity_x,
    shape_name="seven",
    start_top=50,
    start_left=50,
):
    frames, origins = make_moving_2d_shape(
        shape=SHAPE_TEMPLATES[shape_name],
        canvas_height=CANVAS_HEIGHT,
        canvas_width=CANVAS_WIDTH,
        steps=STEPS,
        start_top=start_top,
        start_left=start_left,
        velocity_y=velocity_y,
        velocity_x=velocity_x,
        intensity=INTENSITY,
    )

    spikes = (frames > 0).float()

    return {
        "shape_name": shape_name,
        "velocity_y": velocity_y,
        "velocity_x": velocity_x,
        "frames": frames,
        "spikes": spikes,
        "origins": origins,
    }

def velocity_vector_to_direction_name(dy, dx):
    if dy == 0 and dx == 0:
        return "none"

    vertical = ""
    horizontal = ""

    if dy > 0:
        vertical = "down"
    elif dy < 0:
        vertical = "up"

    if dx > 0:
        horizontal = "right"
    elif dx < 0:
        horizontal = "left"

    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    elif vertical:
        return vertical
    else:
        return horizontal

def test_same_activity_estimator_on_velocity_vectors(
    shape_name="seven",
    velocity_values=(-3, -2, -1, 0, 1, 2, 3),
    previous_t=5,
    current_t=6,
    max_offset=6,
    start_top=50,
    start_left=50,
):
    rows = []

    for velocity_y in velocity_values:
        for velocity_x in velocity_values:
            if velocity_y == 0 and velocity_x == 0:
                continue

            case = make_unblurred_spike_case_from_velocity(
                velocity_y=velocity_y,
                velocity_x=velocity_x,
                shape_name=shape_name,
                start_top=start_top,
                start_left=start_left,
            )

            result = estimate_velocity_from_motion_pairs_with_same_as_activity(
                previous_spikes=case["spikes"][previous_t],
                current_spikes=case["spikes"][current_t],
                max_offset=max_offset,
            )

            true_direction = velocity_vector_to_direction_name(
                velocity_y,
                velocity_x,
            )

            rows.append({
                "shape": shape_name,
                "true_dy": velocity_y,
                "true_dx": velocity_x,
                "true_direction": true_direction,
                "true_speed": (velocity_y ** 2 + velocity_x ** 2) ** 0.5,
                "estimated_dy": result["estimated_dy"],
                "estimated_dx": result["estimated_dx"],
                "estimated_direction": result["estimated_direction"],
                "estimated_speed": result["estimated_speed"],
                "best_pair_spike_count": result["best_pair_spike_count"],
                "correct_dy": result["estimated_dy"] == velocity_y,
                "correct_dx": result["estimated_dx"] == velocity_x,
                "correct_vector": (
                    result["estimated_dy"] == velocity_y
                    and result["estimated_dx"] == velocity_x
                ),
                "correct_direction": result["estimated_direction"] == true_direction,
            })

    return pd.DataFrame(rows)

def test_one_case_across_all_frame_pairs(
    shape_name="seven",
    velocity_y=1,
    velocity_x=2,
    max_offset=6,
    start_top=50,
    start_left=50,
):
    case = make_unblurred_spike_case_from_velocity(
        velocity_y=velocity_y,
        velocity_x=velocity_x,
        shape_name=shape_name,
        start_top=start_top,
        start_left=start_left,
    )

    rows = []

    for previous_t in range(STEPS - 1):
        current_t = previous_t + 1

        result = estimate_velocity_from_motion_pairs_with_same_as_activity(
            previous_spikes=case["spikes"][previous_t],
            current_spikes=case["spikes"][current_t],
            max_offset=max_offset,
        )

        rows.append({
            "shape": shape_name,
            "previous_t": previous_t,
            "current_t": current_t,
            "true_dy": velocity_y,
            "true_dx": velocity_x,
            "true_direction": velocity_vector_to_direction_name(velocity_y, velocity_x),
            "true_speed": (velocity_y ** 2 + velocity_x ** 2) ** 0.5,
            "estimated_dy": result["estimated_dy"],
            "estimated_dx": result["estimated_dx"],
            "estimated_direction": result["estimated_direction"],
            "estimated_speed": result["estimated_speed"],
            "best_pair_spike_count": result["best_pair_spike_count"],
            "correct_vector": (
                result["estimated_dy"] == velocity_y
                and result["estimated_dx"] == velocity_x
            ),
            "correct_direction": (
                result["estimated_direction"]
                == velocity_vector_to_direction_name(velocity_y, velocity_x)
            ),
        })

    return pd.DataFrame(rows)

def test_same_activity_estimator_all_frame_pairs(
    shape_name="seven",
    velocity_values=(-3, -2, -1, 0, 1, 2, 3),
    max_offset=6,
    start_top=50,
    start_left=50,
):
    rows = []

    for velocity_y in velocity_values:
        for velocity_x in velocity_values:
            if velocity_y == 0 and velocity_x == 0:
                continue

            case = make_unblurred_spike_case_from_velocity(
                velocity_y=velocity_y,
                velocity_x=velocity_x,
                shape_name=shape_name,
                start_top=start_top,
                start_left=start_left,
            )

            true_direction = velocity_vector_to_direction_name(
                velocity_y,
                velocity_x,
            )

            for previous_t in range(STEPS - 1):
                current_t = previous_t + 1

                result = estimate_velocity_from_motion_pairs_with_same_as_activity(
                    previous_spikes=case["spikes"][previous_t],
                    current_spikes=case["spikes"][current_t],
                    max_offset=max_offset,
                )

                rows.append({
                    "shape": shape_name,
                    "previous_t": previous_t,
                    "current_t": current_t,
                    "true_dy": velocity_y,
                    "true_dx": velocity_x,
                    "true_direction": true_direction,
                    "true_speed": (velocity_y ** 2 + velocity_x ** 2) ** 0.5,
                    "estimated_dy": result["estimated_dy"],
                    "estimated_dx": result["estimated_dx"],
                    "estimated_direction": result["estimated_direction"],
                    "estimated_speed": result["estimated_speed"],
                    "best_pair_spike_count": result["best_pair_spike_count"],
                    "correct_dy": result["estimated_dy"] == velocity_y,
                    "correct_dx": result["estimated_dx"] == velocity_x,
                    "correct_vector": (
                        result["estimated_dy"] == velocity_y
                        and result["estimated_dx"] == velocity_x
                    ),
                    "correct_direction": result["estimated_direction"] == true_direction,
                })

    return pd.DataFrame(rows)

def plot_input_next_to_spikes_general(case, title=None):
    frames = case["frames"]
    spikes = case["spikes"]

    num_steps = frames.shape[0]

    fig, axes = plt.subplots(2, num_steps, figsize=(2.5 * num_steps, 5))

    for t in range(num_steps):
        axes[0, t].imshow(frames[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"Input t={t}")
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])

        axes[1, t].imshow(spikes[t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(
            f"Spikes t={t}\ncount={int(spikes[t].sum().item())}"
        )
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])

    if title is None:
        if "velocity_sequence" in case:
            title = (
                f"Variable-velocity case: {case['shape_name']}\n"
                f"velocity sequence = {case['velocity_sequence']}"
            )
        elif "direction_name" in case and "speed" in case:
            title = (
                f"Input vs spikes: {case['shape_name']}, "
                f"{case['direction_name']}, speed {case['speed']}"
            )
        elif "velocity_y" in case and "velocity_x" in case:
            title = (
                f"Input vs spikes: {case['shape_name']}, "
                f"velocity dy={case['velocity_y']}, dx={case['velocity_x']}"
            )
        else:
            title = f"Input vs spikes: {case.get('shape_name', 'unknown shape')}"

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_input_next_to_spikes_zoomed(case, padding=8, title=None):
    frames = case["frames"]
    spikes = case["spikes"]

    num_steps = frames.shape[0]

    active_positions = torch.nonzero(spikes.sum(dim=0) > 0, as_tuple=False)

    if active_positions.numel() == 0:
        print("No active pixels found.")
        return

    y_min = max(0, int(active_positions[:, 0].min().item()) - padding)
    y_max = min(CANVAS_HEIGHT, int(active_positions[:, 0].max().item()) + padding + 1)
    x_min = max(0, int(active_positions[:, 1].min().item()) - padding)
    x_max = min(CANVAS_WIDTH, int(active_positions[:, 1].max().item()) + padding + 1)

    fig, axes = plt.subplots(2, num_steps, figsize=(2.5 * num_steps, 5))

    for t in range(num_steps):
        axes[0, t].imshow(
            frames[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, t].set_title(f"Input t={t}")
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])

        axes[1, t].imshow(
            spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, t].set_title(
            f"Spikes t={t}\ncount={int(spikes[t].sum().item())}"
        )
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])

    if title is None:
        if "velocity_sequence" in case:
            title = (
                f"Zoomed variable-velocity case: {case['shape_name']}\n"
                f"velocity sequence = {case['velocity_sequence']}"
            )
        else:
            title = f"Zoomed input vs spikes: {case.get('shape_name', 'unknown shape')}"

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def estimate_variable_velocity_sequence(
    case,
    max_offset=6,
):
    velocity_sequence = case["velocity_sequence"]
    spikes = case["spikes"]

    rows = []

    for previous_t, true_velocity in enumerate(velocity_sequence):
        current_t = previous_t + 1

        true_dy, true_dx = true_velocity

        result = estimate_velocity_from_motion_pairs_with_same_as_activity(
            previous_spikes=spikes[previous_t],
            current_spikes=spikes[current_t],
            max_offset=max_offset,
        )

        true_direction = velocity_vector_to_direction_name(true_dy, true_dx)

        rows.append({
            "previous_t": previous_t,
            "current_t": current_t,
            "true_dy": true_dy,
            "true_dx": true_dx,
            "true_direction": true_direction,
            "true_speed": (true_dy ** 2 + true_dx ** 2) ** 0.5,
            "estimated_dy": result["estimated_dy"],
            "estimated_dx": result["estimated_dx"],
            "estimated_direction": result["estimated_direction"],
            "estimated_speed": result["estimated_speed"],
            "best_pair_spike_count": result["best_pair_spike_count"],
            "correct_vector": (
                result["estimated_dy"] == true_dy
                and result["estimated_dx"] == true_dx
            ),
            "correct_direction": result["estimated_direction"] == true_direction,
        })

    return pd.DataFrame(rows)

def make_random_velocity_sequence(
    num_transitions=6,
    velocity_values=(-3, -2, -1, 0, 1, 2, 3),
    allow_zero_velocity=False,
):
    velocity_sequence = []

    for _ in range(num_transitions):
        while True:
            velocity_y = int(torch.randint(
                low=0,
                high=len(velocity_values),
                size=(1,),
            ).item())

            velocity_x = int(torch.randint(
                low=0,
                high=len(velocity_values),
                size=(1,),
            ).item())

            dy = velocity_values[velocity_y]
            dx = velocity_values[velocity_x]

            if allow_zero_velocity or not (dy == 0 and dx == 0):
                break

        velocity_sequence.append((dy, dx))

    return velocity_sequence

def test_random_variable_velocity_sequences(
    num_sequences=100,
    num_transitions=6,
    shapes=("seven", "ell", "tee", "box"),
    velocity_values=(-3, -2, -1, 0, 1, 2, 3),
    max_offset=6,
    start_top=50,
    start_left=50,
    random_seed=0,
):
    torch.manual_seed(random_seed)

    rows = []

    for sequence_index in range(num_sequences):
        for shape_name in shapes:
            velocity_sequence = make_random_velocity_sequence(
                num_transitions=num_transitions,
                velocity_values=velocity_values,
                allow_zero_velocity=False,
            )

            case = make_variable_velocity_spike_case(
                velocity_sequence=velocity_sequence,
                shape_name=shape_name,
                start_top=start_top,
                start_left=start_left,
            )

            result_table = estimate_variable_velocity_sequence(
                case,
                max_offset=max_offset,
            )

            result_table = result_table.copy()
            result_table["sequence_index"] = sequence_index
            result_table["shape"] = shape_name

            rows.append(result_table)

    return pd.concat(rows, ignore_index=True)

def place_shape_clipped_on_canvas(
    shape,
    canvas_height,
    canvas_width,
    top,
    left,
    intensity=1.0,
):
    """
    Place a shape on the canvas, allowing it to be partially outside the frame.
    Only the visible part is drawn.
    """

    frame = torch.zeros(
        (canvas_height, canvas_width),
        dtype=torch.float32,
    )

    shape_height, shape_width = shape.shape

    top = int(top)
    left = int(left)

    canvas_y_start = max(0, top)
    canvas_y_end = min(canvas_height, top + shape_height)

    canvas_x_start = max(0, left)
    canvas_x_end = min(canvas_width, left + shape_width)

    shape_y_start = canvas_y_start - top
    shape_y_end = shape_y_start + (canvas_y_end - canvas_y_start)

    shape_x_start = canvas_x_start - left
    shape_x_end = shape_x_start + (canvas_x_end - canvas_x_start)

    if canvas_y_start < canvas_y_end and canvas_x_start < canvas_x_end:
        frame[
            canvas_y_start:canvas_y_end,
            canvas_x_start:canvas_x_end,
        ] = shape[
            shape_y_start:shape_y_end,
            shape_x_start:shape_x_end,
        ].float() * intensity

    return frame

def make_boundary_entry_spike_case(
    velocity_y,
    velocity_x,
    shape_name="seven",
    start_top=60,
    start_left=-4,
    steps=STEPS,
):
    """
    Create a sequence where the object may start partially outside the canvas
    and enter the visible frame.
    """

    shape = SHAPE_TEMPLATES[shape_name]

    frames = []
    positions = []

    for t in range(steps):
        top = start_top + t * velocity_y
        left = start_left + t * velocity_x

        frame = place_shape_clipped_on_canvas(
            shape=shape,
            canvas_height=CANVAS_HEIGHT,
            canvas_width=CANVAS_WIDTH,
            top=top,
            left=left,
            intensity=INTENSITY,
        )

        frames.append(frame)
        positions.append((top, left))

    frames = torch.stack(frames, dim=0)
    spikes = (frames > 0).float()

    return {
        "shape_name": shape_name,
        "velocity_y": velocity_y,
        "velocity_x": velocity_x,
        "positions": positions,
        "frames": frames,
        "spikes": spikes,
    }

def estimate_constant_velocity_case_all_frame_pairs(case, max_offset=6):
    rows = []

    velocity_y = case["velocity_y"]
    velocity_x = case["velocity_x"]

    true_direction = velocity_vector_to_direction_name(
        velocity_y,
        velocity_x,
    )

    for previous_t in range(case["spikes"].shape[0] - 1):
        current_t = previous_t + 1

        result = estimate_velocity_from_motion_pairs_with_same_as_activity(
            previous_spikes=case["spikes"][previous_t],
            current_spikes=case["spikes"][current_t],
            max_offset=max_offset,
        )

        rows.append({
            "shape": case["shape_name"],
            "previous_t": previous_t,
            "current_t": current_t,
            "true_dy": velocity_y,
            "true_dx": velocity_x,
            "true_direction": true_direction,
            "estimated_dy": result["estimated_dy"],
            "estimated_dx": result["estimated_dx"],
            "estimated_direction": result["estimated_direction"],
            "best_pair_spike_count": result["best_pair_spike_count"],
            "correct_vector": (
                result["estimated_dy"] == velocity_y
                and result["estimated_dx"] == velocity_x
            ),
            "correct_direction": result["estimated_direction"] == true_direction,
        })

    return pd.DataFrame(rows)

def count_visible_spikes_per_frame(case):
    rows = []

    for t in range(case["spikes"].shape[0]):
        rows.append({
            "t": t,
            "visible_spike_count": int(case["spikes"][t].sum().item()),
        })

    return pd.DataFrame(rows)

def estimate_constant_velocity_case_all_frame_pairs_with_visibility(case, max_offset=6):
    rows = []

    velocity_y = case["velocity_y"]
    velocity_x = case["velocity_x"]

    true_direction = velocity_vector_to_direction_name(
        velocity_y,
        velocity_x,
    )

    for previous_t in range(case["spikes"].shape[0] - 1):
        current_t = previous_t + 1

        result = estimate_velocity_from_motion_pairs_with_same_as_activity(
            previous_spikes=case["spikes"][previous_t],
            current_spikes=case["spikes"][current_t],
            max_offset=max_offset,
        )

        previous_visible_spikes = int(case["spikes"][previous_t].sum().item())
        current_visible_spikes = int(case["spikes"][current_t].sum().item())

        rows.append({
            "shape": case["shape_name"],
            "previous_t": previous_t,
            "current_t": current_t,
            "previous_visible_spikes": previous_visible_spikes,
            "current_visible_spikes": current_visible_spikes,
            "min_visible_spikes": min(previous_visible_spikes, current_visible_spikes),
            "true_dy": velocity_y,
            "true_dx": velocity_x,
            "true_direction": true_direction,
            "estimated_dy": result["estimated_dy"],
            "estimated_dx": result["estimated_dx"],
            "estimated_direction": result["estimated_direction"],
            "best_pair_spike_count": result["best_pair_spike_count"],
            "correct_vector": (
                result["estimated_dy"] == velocity_y
                and result["estimated_dx"] == velocity_x
            ),
            "correct_direction": result["estimated_direction"] == true_direction,
        })

    return pd.DataFrame(rows)

def visible_spike_count(frame):
    return int((frame > 0).sum().item())

def test_boundary_entry_visibility_sweep(
    shape_name="seven",
    velocity_y=0,
    velocity_x=2,
    start_top=60,
    start_left_values=range(-8, 3),
    steps=STEPS,
    max_offset=6,
):
    rows = []

    for start_left in start_left_values:
        case = make_boundary_entry_spike_case(
            velocity_y=velocity_y,
            velocity_x=velocity_x,
            shape_name=shape_name,
            start_top=start_top,
            start_left=start_left,
            steps=steps,
        )

        result_table = estimate_constant_velocity_case_all_frame_pairs_with_visibility(
            case,
            max_offset=max_offset,
        )

        result_table = result_table.copy()
        result_table["start_left"] = start_left
        result_table["initial_visible_spikes"] = visible_spike_count(case["spikes"][0])
        result_table["shape"] = shape_name

        rows.append(result_table)

    return pd.concat(rows, ignore_index=True)

def estimate_velocity_with_confidence(
    previous_spikes,
    current_spikes,
    max_offset=10,
    min_visible_spikes=3,
    min_winner_margin=1,
):
    result = estimate_velocity_from_motion_pairs_with_same_as_activity(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
        max_offset=max_offset,
    )

    scores = result["motion_pair_scores"].copy()

    best_score = int(scores.iloc[0]["pair_spike_count"])
    second_score = int(scores.iloc[1]["pair_spike_count"])
    winner_margin = best_score - second_score

    previous_visible_spikes = int((previous_spikes > 0).sum().item())
    current_visible_spikes = int((current_spikes > 0).sum().item())
    min_visible = min(previous_visible_spikes, current_visible_spikes)

    enough_visible_evidence = min_visible >= min_visible_spikes
    strong_winner = winner_margin >= min_winner_margin

    high_confidence = enough_visible_evidence and strong_winner

    result["previous_visible_spikes"] = previous_visible_spikes
    result["current_visible_spikes"] = current_visible_spikes
    result["min_visible_spikes"] = min_visible
    result["best_score"] = best_score
    result["second_best_score"] = second_score
    result["winner_margin"] = winner_margin
    result["high_confidence"] = high_confidence

    return result

def visualize_boundary_entry_case(
    shape_name="ell",
    velocity_y=0,
    velocity_x=2,
    start_top=60,
    start_left=-4,
    max_offset=6,
    steps=STEPS,
):
    case = make_boundary_entry_spike_case(
        velocity_y=velocity_y,
        velocity_x=velocity_x,
        shape_name=shape_name,
        start_top=start_top,
        start_left=start_left,
        steps=steps,
    )

    plot_input_next_to_spikes_zoomed(
        case,
        padding=6,
    )

    results = estimate_constant_velocity_case_all_frame_pairs_with_visibility(
        case,
        max_offset=max_offset,
    )

    display(results)

    return case, results

def lif_step(input_current, membrane, beta=0.0, threshold=1.0, reset_mode="zero"):
    """
    One simple LIF step.

    input_current: synaptic input to neuron population
    membrane: previous membrane potential
    beta: leak factor
    threshold: spike threshold
    reset_mode: "zero" or "subtract"
    """

    membrane = beta * membrane + input_current
    spikes = (membrane >= threshold).float()

    if reset_mode == "zero":
        membrane = membrane * (1.0 - spikes)
    elif reset_mode == "subtract":
        membrane = membrane - spikes * threshold
    else:
        raise ValueError("reset_mode must be 'zero' or 'subtract'.")

    return spikes, membrane

def compute_lif_event_populations(
    previous_spikes,
    current_spikes,
    excitatory_weight=1.0,
    inhibitory_weight=-1.0,
    threshold_on=0.5,
    threshold_off=0.5,
    threshold_same=1.5,
    beta=0.0,
    reset_mode="zero",
):
    """
    LIF implementation of ON, OFF, and SAME event populations.

    ON[y,x]:
        current excites
        previous inhibits

    OFF[y,x]:
        previous excites
        current inhibits

    SAME[y,x]:
        previous excites
        current excites
        threshold requires both
    """

    previous_spikes = (previous_spikes > 0).float()
    current_spikes = (current_spikes > 0).float()

    # ON: current present, previous absent
    on_input = (
        excitatory_weight * current_spikes
        + inhibitory_weight * previous_spikes
    )

    # OFF: previous present, current absent
    off_input = (
        excitatory_weight * previous_spikes
        + inhibitory_weight * current_spikes
    )

    # SAME: previous and current both present
    same_input = (
        excitatory_weight * previous_spikes
        + excitatory_weight * current_spikes
    )

    # Fresh membranes for this comparison step.
    on_membrane = torch.zeros_like(previous_spikes)
    off_membrane = torch.zeros_like(previous_spikes)
    same_membrane = torch.zeros_like(previous_spikes)

    on_spikes, on_membrane = lif_step(
        input_current=on_input,
        membrane=on_membrane,
        beta=beta,
        threshold=threshold_on,
        reset_mode=reset_mode,
    )

    off_spikes, off_membrane = lif_step(
        input_current=off_input,
        membrane=off_membrane,
        beta=beta,
        threshold=threshold_off,
        reset_mode=reset_mode,
    )

    same_spikes, same_membrane = lif_step(
        input_current=same_input,
        membrane=same_membrane,
        beta=beta,
        threshold=threshold_same,
        reset_mode=reset_mode,
    )

    return {
        "on_spikes": on_spikes,
        "off_spikes": off_spikes,
        "same_spikes": same_spikes,
        "on_input": on_input,
        "off_input": off_input,
        "same_input": same_input,
        "on_membrane": on_membrane,
        "off_membrane": off_membrane,
        "same_membrane": same_membrane,
    }

def plot_lif_event_populations(case, previous_t=5, current_t=6):
    previous_spikes = case["spikes"][previous_t]
    current_spikes = case["spikes"][current_t]

    lif_events = compute_lif_event_populations(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
    )

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    axes[0].imshow(previous_spikes, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Previous\n t={previous_t}")

    axes[1].imshow(current_spikes, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Current\n t={current_t}")

    axes[2].imshow(lif_events["off_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"OFF LIF\ncount={int(lif_events['off_spikes'].sum().item())}"
    )

    axes[3].imshow(lif_events["on_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(
        f"ON LIF\ncount={int(lif_events['on_spikes'].sum().item())}"
    )

    axes[4].imshow(lif_events["same_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[4].set_title(
        f"SAME LIF\ncount={int(lif_events['same_spikes'].sum().item())}"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("LIF ON/OFF/SAME event populations")
    plt.tight_layout()
    plt.show()

def compute_lif_motion_pair_layer_fast(
    off_spikes,
    on_spikes,
    same_spikes,
    max_offset=6,
    old_weight=1.0,
    new_weight=1.0,
    threshold_pair=1.5,
    beta=0.0,
    reset_mode="zero",
):
    """
    Faster LIF motion-pair layer.

    Only evaluates source positions where old-side evidence exists:
        OFF[y,x] or SAME[y,x]

    This is equivalent for clean one-step coincidence detection, because
    a motion-pair neuron with no old-side evidence cannot cross threshold.
    """

    off_spikes = (off_spikes > 0).float()
    on_spikes = (on_spikes > 0).float()
    same_spikes = (same_spikes > 0).float()

    height, width = off_spikes.shape

    old_side_evidence = ((off_spikes > 0) | (same_spikes > 0)).float()
    old_positions = torch.nonzero(old_side_evidence > 0, as_tuple=False)

    rows = []

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            if dy == 0 and dx == 0:
                continue

            pair_spike_count = 0
            pair_membrane_sum = 0.0

            for pos in old_positions:
                y = int(pos[0].item())
                x = int(pos[1].item())

                target_y = y + dy
                target_x = x + dx

                if not (0 <= target_y < height and 0 <= target_x < width):
                    continue

                old_side_input = (
                    old_weight * off_spikes[y, x]
                    + old_weight * same_spikes[y, x]
                )

                new_side_input = (
                    new_weight * on_spikes[target_y, target_x]
                    + new_weight * same_spikes[target_y, target_x]
                )

                pair_input = old_side_input + new_side_input

                pair_membrane = torch.zeros((), dtype=torch.float32)

                pair_spike, pair_membrane = lif_step(
                    input_current=pair_input,
                    membrane=pair_membrane,
                    beta=beta,
                    threshold=threshold_pair,
                    reset_mode=reset_mode,
                )

                pair_spike_count += int(pair_spike.item())
                pair_membrane_sum += float(pair_input.item())

            rows.append({
                "dy": dy,
                "dx": dx,
                "pair_spike_count": pair_spike_count,
                "pair_membrane_sum": pair_membrane_sum,
                "offset_magnitude": (dy ** 2 + dx ** 2) ** 0.5,
            })

    motion_pair_scores = pd.DataFrame(rows)

    motion_pair_scores = motion_pair_scores.sort_values(
        ["pair_spike_count", "offset_magnitude"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return motion_pair_scores

def estimate_velocity_from_lif_event_and_lif_motion_pairs(
    previous_spikes,
    current_spikes,
    max_offset=6,
):
    lif_events = compute_lif_event_populations(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
    )

    motion_scores = compute_lif_motion_pair_layer_fast(
        off_spikes=lif_events["off_spikes"],
        on_spikes=lif_events["on_spikes"],
        same_spikes=lif_events["same_spikes"],
        max_offset=max_offset,
    )

    best = motion_scores.iloc[0]

    estimated_dy = int(best["dy"])
    estimated_dx = int(best["dx"])
    estimated_speed = (estimated_dy ** 2 + estimated_dx ** 2) ** 0.5

    if estimated_dx == 0 and estimated_dy == 0:
        estimated_direction = "none"
    elif abs(estimated_dx) >= abs(estimated_dy):
        if estimated_dx > 0:
            estimated_direction = "right" if estimated_dy == 0 else ("down-right" if estimated_dy > 0 else "up-right")
        else:
            estimated_direction = "left" if estimated_dy == 0 else ("down-left" if estimated_dy > 0 else "up-left")
    else:
        if estimated_dy > 0:
            estimated_direction = "down" if estimated_dx == 0 else ("down-right" if estimated_dx > 0 else "down-left")
        else:
            estimated_direction = "up" if estimated_dx == 0 else ("up-right" if estimated_dx > 0 else "up-left")

    return {
        "estimated_dy": estimated_dy,
        "estimated_dx": estimated_dx,
        "estimated_speed": estimated_speed,
        "estimated_direction": estimated_direction,
        "best_pair_spike_count": int(best["pair_spike_count"]),
        "lif_events": lif_events,
        "motion_pair_scores": motion_scores,
    }

def compute_lif_motion_pair_layer_fast_with_zero_velocity(
    off_spikes,
    on_spikes,
    same_spikes,
    max_offset=6,
    old_weight=1.0,
    new_weight=1.0,
    threshold_pair=1.5,
    beta=0.0,
    reset_mode="zero",
):
    """
    Fast LIF motion-pair layer including zero velocity (dy=0, dx=0).

    MotionPair[dy, dx, y, x] receives:
        old-side evidence at (y, x):
            OFF[y, x] or SAME[y, x]

        new-side evidence at (y + dy, x + dx):
            ON[y + dy, x + dx] or SAME[y + dy, x + dx]

    The neuron spikes if old-side and new-side evidence coincide.
    """

    off_spikes = (off_spikes > 0).float()
    on_spikes = (on_spikes > 0).float()
    same_spikes = (same_spikes > 0).float()

    height, width = off_spikes.shape

    old_side_evidence = ((off_spikes > 0) | (same_spikes > 0)).float()
    old_positions = torch.nonzero(old_side_evidence > 0, as_tuple=False)

    rows = []

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):

            pair_spike_count = 0

            for pos in old_positions:
                y = int(pos[0].item())
                x = int(pos[1].item())

                target_y = y + dy
                target_x = x + dx

                if not (0 <= target_y < height and 0 <= target_x < width):
                    continue

                old_side_input = (
                    old_weight * off_spikes[y, x]
                    + old_weight * same_spikes[y, x]
                )

                new_side_input = (
                    new_weight * on_spikes[target_y, target_x]
                    + new_weight * same_spikes[target_y, target_x]
                )

                pair_input = old_side_input + new_side_input

                pair_membrane = torch.zeros((), dtype=torch.float32)

                pair_spike, pair_membrane = lif_step(
                    input_current=pair_input,
                    membrane=pair_membrane,
                    beta=beta,
                    threshold=threshold_pair,
                    reset_mode=reset_mode,
                )

                pair_spike_count += int(pair_spike.item())

            rows.append({
                "dy": dy,
                "dx": dx,
                "pair_spike_count": pair_spike_count,
                "offset_magnitude": (dy ** 2 + dx ** 2) ** 0.5,
            })

    motion_pair_scores = pd.DataFrame(rows)

    motion_pair_scores = motion_pair_scores.sort_values(
        ["pair_spike_count", "offset_magnitude"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return motion_pair_scores

def motion_pair_scores_to_velocity_input(
    motion_pair_scores,
    max_offset=6,
):
    """
    Convert motion-pair support into a velocity-layer input grid.

    velocity_input[i, j] corresponds to:
        dy = i - max_offset
        dx = j - max_offset

    Example:
        velocity_input[max_offset + 0, max_offset + 4]
        is input to Velocity[0, 4].
    """

    size = 2 * max_offset + 1
    velocity_input = torch.zeros((size, size), dtype=torch.float32)

    for _, row in motion_pair_scores.iterrows():
        dy = int(row["dy"])
        dx = int(row["dx"])

        i = dy + max_offset
        j = dx + max_offset

        velocity_input[i, j] = float(row["pair_spike_count"])

    return velocity_input

def compute_velocity_lateral_inhibition(
    previous_velocity_spikes,
    inhibition_strength=2.0,
    inhibition_radius=1,
):
    """
    Compute soft lateral inhibition between nearby velocity neurons.

    If Velocity[dy,dx] spiked on the previous internal step,
    it inhibits nearby Velocity[dy',dx'] neurons.

    inhibition_radius=1 means inhibit immediate neighbors in velocity space.
    """

    previous_velocity_spikes = (previous_velocity_spikes > 0).float()

    size_y, size_x = previous_velocity_spikes.shape
    inhibition = torch.zeros_like(previous_velocity_spikes)

    spiking_positions = torch.nonzero(previous_velocity_spikes > 0, as_tuple=False)

    for pos in spiking_positions:
        i = int(pos[0].item())
        j = int(pos[1].item())

        for di in range(-inhibition_radius, inhibition_radius + 1):
            for dj in range(-inhibition_radius, inhibition_radius + 1):
                target_i = i + di
                target_j = j + dj

                if not (0 <= target_i < size_y and 0 <= target_j < size_x):
                    continue

                if target_i == i and target_j == j:
                    continue

                inhibition[target_i, target_j] += inhibition_strength

    return inhibition

def run_lif_velocity_layer_with_lateral_inhibition(
    velocity_input,
    beta=0.8,
    threshold_velocity=5.0,
    input_scale=1.0,
    inhibition_strength=3.0,
    inhibition_radius=1,
    internal_steps=5,
    reset_mode="subtract",
):
    """
    LIF velocity accumulator layer with soft lateral inhibition.

    velocity_input[i,j] is the excitatory drive to Velocity[dy,dx].

    The layer runs for several internal steps:
        membrane accumulates velocity evidence
        velocity spikes inhibit nearby velocity neurons
        stronger velocity hypotheses should remain active more reliably
    """

    velocity_input = velocity_input.float()

    velocity_membrane = torch.zeros_like(velocity_input)
    previous_velocity_spikes = torch.zeros_like(velocity_input)

    spike_history = []
    membrane_history = []
    inhibition_history = []

    for internal_t in range(internal_steps):
        inhibition = compute_velocity_lateral_inhibition(
            previous_velocity_spikes=previous_velocity_spikes,
            inhibition_strength=inhibition_strength,
            inhibition_radius=inhibition_radius,
        )

        current_input = input_scale * velocity_input - inhibition

        velocity_spikes, velocity_membrane = lif_step(
            input_current=current_input,
            membrane=velocity_membrane,
            beta=beta,
            threshold=threshold_velocity,
            reset_mode=reset_mode,
        )

        spike_history.append(velocity_spikes.clone())
        membrane_history.append(velocity_membrane.clone())
        inhibition_history.append(inhibition.clone())

        previous_velocity_spikes = velocity_spikes.clone()

    spike_history = torch.stack(spike_history, dim=0)
    membrane_history = torch.stack(membrane_history, dim=0)
    inhibition_history = torch.stack(inhibition_history, dim=0)

    return {
        "velocity_spike_history": spike_history,
        "velocity_membrane_history": membrane_history,
        "velocity_inhibition_history": inhibition_history,
        "final_velocity_spikes": spike_history[-1],
        "final_velocity_membrane": membrane_history[-1],
    }

def decode_velocity_spikes(
    velocity_spikes,
    max_offset=6,
):
    """
    Decode active Velocity[dy,dx] neurons from a velocity spike grid.
    """

    active_positions = torch.nonzero(velocity_spikes > 0, as_tuple=False)

    rows = []

    for pos in active_positions:
        i = int(pos[0].item())
        j = int(pos[1].item())

        dy = i - max_offset
        dx = j - max_offset

        rows.append({
            "dy": dy,
            "dx": dx,
            "speed": (dy ** 2 + dx ** 2) ** 0.5,
        })

    return pd.DataFrame(rows)

def summarize_velocity_spike_history(
    velocity_spike_history,
    max_offset=6,
):
    """
    Summarize how often each Velocity[dy,dx] neuron spiked
    over the internal decision window.

    No artificial preference for lower speed.
    """

    total_spikes = velocity_spike_history.sum(dim=0)

    rows = []

    size = total_spikes.shape[0]

    for i in range(size):
        for j in range(size):
            spike_count = int(total_spikes[i, j].item())

            if spike_count == 0:
                continue

            dy = i - max_offset
            dx = j - max_offset

            rows.append({
                "dy": dy,
                "dx": dx,
                "velocity_spike_count": spike_count,
                "speed": (dy ** 2 + dx ** 2) ** 0.5,
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "dy",
            "dx",
            "velocity_spike_count",
            "speed",
        ])

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["velocity_spike_count"],
        ascending=[False],
    ).reset_index(drop=True)

    return summary

def estimate_velocity_with_lif_motion_pairs_and_inhibition(
    previous_spikes,
    current_spikes,
    max_offset=6,
    threshold_velocity=5.0,
    input_scale=1.0,
    beta_velocity=0.8,
    inhibition_strength=3.0,
    inhibition_radius=1,
    internal_steps=5,
):
    """
    Full velocity estimate using:

        LIF ON/OFF/SAME event populations
        LIF motion-pair coincidence neurons
        LIF velocity accumulator neurons
        soft lateral inhibition in velocity space
    """

    lif_events = compute_lif_event_populations(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
    )

    motion_pair_scores = compute_lif_motion_pair_layer_fast_with_zero_velocity(
        off_spikes=lif_events["off_spikes"],
        on_spikes=lif_events["on_spikes"],
        same_spikes=lif_events["same_spikes"],
        max_offset=max_offset,
    )

    velocity_input = motion_pair_scores_to_velocity_input(
        motion_pair_scores=motion_pair_scores,
        max_offset=max_offset,
    )

    velocity_layer = run_lif_velocity_layer_with_lateral_inhibition(
        velocity_input=velocity_input,
        beta=beta_velocity,
        threshold_velocity=threshold_velocity,
        input_scale=input_scale,
        inhibition_strength=inhibition_strength,
        inhibition_radius=inhibition_radius,
        internal_steps=internal_steps,
        reset_mode="subtract",
    )

    velocity_summary = summarize_velocity_spike_history(
        velocity_spike_history=velocity_layer["velocity_spike_history"],
        max_offset=max_offset,
    )

    final_active_velocities = decode_velocity_spikes(
        velocity_spikes=velocity_layer["final_velocity_spikes"],
        max_offset=max_offset,
    )

    if len(velocity_summary) > 0:
        best = velocity_summary.iloc[0]

        estimated_dy = int(best["dy"])
        estimated_dx = int(best["dx"])
        estimated_speed = float(best["speed"])
    else:
        estimated_dy = None
        estimated_dx = None
        estimated_speed = None

    return {
        "estimated_dy": estimated_dy,
        "estimated_dx": estimated_dx,
        "estimated_speed": estimated_speed,
        "lif_events": lif_events,
        "motion_pair_scores": motion_pair_scores,
        "velocity_input": velocity_input,
        "velocity_layer": velocity_layer,
        "velocity_summary": velocity_summary,
        "final_active_velocities": final_active_velocities,
    }

def test_lif_velocity_inhibition_clean_directions(
    shapes=("seven", "ell", "tee", "box"),
    speeds=(1, 2, 3, 4, 5, 6),
    directions=("right", "left", "down", "up"),
    max_offset=6,
    velocity_params=None,
    previous_t=5,
    current_t=6,
):
    if velocity_params is None:
        velocity_params = {}

    rows = []

    for shape_name in shapes:
        for speed in speeds:
            for direction_name in directions:
                case = make_unblurred_spike_case(
                    speed=speed,
                    direction_name=direction_name,
                    shape_name=shape_name,
                )

                result = estimate_velocity_with_lif_motion_pairs_and_inhibition(
                    previous_spikes=case["spikes"][previous_t],
                    current_spikes=case["spikes"][current_t],
                    max_offset=max_offset,
                    **velocity_params,
                )

                true_dy = case["velocity_y"]
                true_dx = case["velocity_x"]

                rows.append({
                    "shape": shape_name,
                    "speed": speed,
                    "direction": direction_name,
                    "true_dy": true_dy,
                    "true_dx": true_dx,
                    "estimated_dy": result["estimated_dy"],
                    "estimated_dx": result["estimated_dx"],
                    "correct": (
                        result["estimated_dy"] == true_dy
                        and result["estimated_dx"] == true_dx
                    ),
                    "num_velocity_outputs": len(result["velocity_summary"]),
                })

    return pd.DataFrame(rows)

def test_lif_velocity_inhibition_arbitrary_vectors(
    shapes=("seven", "ell", "tee", "box"),
    velocity_values=(-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
    max_offset=6,
    velocity_params=None,
    previous_t=5,
    current_t=6,
    start_top=50,
    start_left=50,
):
    if velocity_params is None:
        velocity_params = {}

    rows = []

    for shape_name in shapes:
        for true_dy in velocity_values:
            for true_dx in velocity_values:
                case = make_unblurred_spike_case_from_velocity(
                    velocity_y=true_dy,
                    velocity_x=true_dx,
                    shape_name=shape_name,
                    start_top=start_top,
                    start_left=start_left,
                )

                result = estimate_velocity_with_lif_motion_pairs_and_inhibition(
                    previous_spikes=case["spikes"][previous_t],
                    current_spikes=case["spikes"][current_t],
                    max_offset=max_offset,
                    **velocity_params,
                )

                rows.append({
                    "shape": shape_name,
                    "true_dy": true_dy,
                    "true_dx": true_dx,
                    "estimated_dy": result["estimated_dy"],
                    "estimated_dx": result["estimated_dx"],
                    "correct": (
                        result["estimated_dy"] == true_dy
                        and result["estimated_dx"] == true_dx
                    ),
                    "num_velocity_outputs": len(result["velocity_summary"]),
                })

    return pd.DataFrame(rows)

def test_lif_velocity_inhibition_all_frame_pairs(
    shapes=("seven", "ell", "tee", "box"),
    velocity_values=(-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
    max_offset=6,
    velocity_params=None,
    start_top=50,
    start_left=50,
):
    if velocity_params is None:
        velocity_params = {}

    rows = []

    for shape_name in shapes:
        for true_dy in velocity_values:
            for true_dx in velocity_values:
                case = make_unblurred_spike_case_from_velocity(
                    velocity_y=true_dy,
                    velocity_x=true_dx,
                    shape_name=shape_name,
                    start_top=start_top,
                    start_left=start_left,
                )

                num_steps = case["spikes"].shape[0]

                for previous_t in range(num_steps - 1):
                    current_t = previous_t + 1

                    result = estimate_velocity_with_lif_motion_pairs_and_inhibition(
                        previous_spikes=case["spikes"][previous_t],
                        current_spikes=case["spikes"][current_t],
                        max_offset=max_offset,
                        **velocity_params,
                    )

                    rows.append({
                        "shape": shape_name,
                        "previous_t": previous_t,
                        "current_t": current_t,
                        "true_dy": true_dy,
                        "true_dx": true_dx,
                        "estimated_dy": result["estimated_dy"],
                        "estimated_dx": result["estimated_dx"],
                        "correct": (
                            result["estimated_dy"] == true_dy
                            and result["estimated_dx"] == true_dx
                        ),
                        "num_velocity_outputs": len(result["velocity_summary"]),
                    })

    return pd.DataFrame(rows)

def compute_lif_motion_pair_spike_tensor(
    off_spikes,
    on_spikes,
    same_spikes,
    max_offset=6,
    old_weight=1.0,
    new_weight=1.0,
    threshold_pair=1.5,
    beta=0.0,
    reset_mode="zero",
):
    """
    Explicit LIF motion-pair spike tensor.

    For each candidate velocity (dy, dx), and source pixel (y, x),
    simulate a MotionPair[dy, dx, y, x] coincidence neuron.

    Returns:
        motion_pair_spikes:
            shape [num_offsets, height, width]

        offset_table:
            DataFrame mapping offset_index -> dy, dx

    motion_pair_spikes[k, y, x] = 1 means:
        MotionPair[dy_k, dx_k, y, x] spiked.
    """

    off_spikes = (off_spikes > 0).float()
    on_spikes = (on_spikes > 0).float()
    same_spikes = (same_spikes > 0).float()

    height, width = off_spikes.shape

    offsets = []

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            offsets.append((dy, dx))

    num_offsets = len(offsets)

    motion_pair_spikes = torch.zeros(
        (num_offsets, height, width),
        dtype=torch.float32,
    )

    old_side_evidence = ((off_spikes > 0) | (same_spikes > 0)).float()
    old_positions = torch.nonzero(old_side_evidence > 0, as_tuple=False)

    for offset_index, (dy, dx) in enumerate(offsets):
        for pos in old_positions:
            y = int(pos[0].item())
            x = int(pos[1].item())

            target_y = y + dy
            target_x = x + dx

            if not (0 <= target_y < height and 0 <= target_x < width):
                continue

            old_side_input = (
                old_weight * off_spikes[y, x]
                + old_weight * same_spikes[y, x]
            )

            new_side_input = (
                new_weight * on_spikes[target_y, target_x]
                + new_weight * same_spikes[target_y, target_x]
            )

            pair_input = old_side_input + new_side_input

            pair_membrane = torch.zeros((), dtype=torch.float32)

            pair_spike, pair_membrane = lif_step(
                input_current=pair_input,
                membrane=pair_membrane,
                beta=beta,
                threshold=threshold_pair,
                reset_mode=reset_mode,
            )

            motion_pair_spikes[offset_index, y, x] = pair_spike

    offset_table = pd.DataFrame([
        {
            "offset_index": offset_index,
            "dy": dy,
            "dx": dx,
            "speed": (dy ** 2 + dx ** 2) ** 0.5,
        }
        for offset_index, (dy, dx) in enumerate(offsets)
    ])

    return {
        "motion_pair_spikes": motion_pair_spikes,
        "offset_table": offset_table,
    }

def motion_pair_spike_tensor_to_velocity_input(
    motion_pair_spikes,
    offset_table,
    max_offset=6,
):
    """
    Convert explicit MotionPair[dy,dx,y,x] spikes into Velocity[dy,dx] input.

    This represents synaptic fan-in:
        all MotionPair[dy,dx,:,:] neurons excite Velocity[dy,dx].
    """

    size = 2 * max_offset + 1
    velocity_input = torch.zeros((size, size), dtype=torch.float32)

    for _, row in offset_table.iterrows():
        offset_index = int(row["offset_index"])
        dy = int(row["dy"])
        dx = int(row["dx"])

        i = dy + max_offset
        j = dx + max_offset

        velocity_input[i, j] = motion_pair_spikes[offset_index].sum()

    return velocity_input

def summarize_motion_pair_spike_tensor(
    motion_pair_spikes,
    offset_table,
):
    """
    Debug summary of explicit motion-pair spike tensor.
    """

    rows = []

    for _, row in offset_table.iterrows():
        offset_index = int(row["offset_index"])
        dy = int(row["dy"])
        dx = int(row["dx"])

        pair_spike_count = int(motion_pair_spikes[offset_index].sum().item())

        rows.append({
            "offset_index": offset_index,
            "dy": dy,
            "dx": dx,
            "pair_spike_count": pair_spike_count,
            "speed": (dy ** 2 + dx ** 2) ** 0.5,
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["pair_spike_count"],
        ascending=[False],
    ).reset_index(drop=True)

    return summary

def estimate_velocity_with_explicit_motion_pairs_and_inhibition(
    previous_spikes,
    current_spikes,
    max_offset=6,
    threshold_velocity=15.0,
    input_scale=1.0,
    beta_velocity=0.8,
    inhibition_strength=3.0,
    inhibition_radius=1,
    internal_steps=5,
):
    """
    Full velocity estimator using:

        LIF ON/OFF/SAME event neurons
        explicit LIF MotionPair[dy,dx,y,x] spike tensor
        LIF Velocity[dy,dx] accumulator neurons
        lateral inhibition in velocity space
    """

    lif_events = compute_lif_event_populations(
        previous_spikes=previous_spikes,
        current_spikes=current_spikes,
    )

    motion_pair_layer = compute_lif_motion_pair_spike_tensor(
        off_spikes=lif_events["off_spikes"],
        on_spikes=lif_events["on_spikes"],
        same_spikes=lif_events["same_spikes"],
        max_offset=max_offset,
    )

    motion_pair_spikes = motion_pair_layer["motion_pair_spikes"]
    offset_table = motion_pair_layer["offset_table"]

    motion_pair_summary = summarize_motion_pair_spike_tensor(
        motion_pair_spikes=motion_pair_spikes,
        offset_table=offset_table,
    )

    velocity_input = motion_pair_spike_tensor_to_velocity_input(
        motion_pair_spikes=motion_pair_spikes,
        offset_table=offset_table,
        max_offset=max_offset,
    )

    velocity_layer = run_lif_velocity_layer_with_lateral_inhibition(
        velocity_input=velocity_input,
        beta=beta_velocity,
        threshold_velocity=threshold_velocity,
        input_scale=input_scale,
        inhibition_strength=inhibition_strength,
        inhibition_radius=inhibition_radius,
        internal_steps=internal_steps,
        reset_mode="subtract",
    )

    velocity_summary = summarize_velocity_spike_history(
        velocity_spike_history=velocity_layer["velocity_spike_history"],
        max_offset=max_offset,
    )

    final_active_velocities = decode_velocity_spikes(
        velocity_spikes=velocity_layer["final_velocity_spikes"],
        max_offset=max_offset,
    )

    if len(velocity_summary) > 0:
        best = velocity_summary.iloc[0]

        estimated_dy = int(best["dy"])
        estimated_dx = int(best["dx"])
        estimated_speed = float(best["speed"])
    else:
        estimated_dy = None
        estimated_dx = None
        estimated_speed = None

    return {
        "estimated_dy": estimated_dy,
        "estimated_dx": estimated_dx,
        "estimated_speed": estimated_speed,
        "lif_events": lif_events,
        "motion_pair_spikes": motion_pair_spikes,
        "offset_table": offset_table,
        "motion_pair_summary": motion_pair_summary,
        "velocity_input": velocity_input,
        "velocity_layer": velocity_layer,
        "velocity_summary": velocity_summary,
        "final_active_velocities": final_active_velocities,
    }

def test_explicit_motion_pair_velocity_inhibition_arbitrary_vectors(
    shapes=("seven", "ell", "tee", "box"),
    velocity_values=(-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
    max_offset=6,
    velocity_params=None,
    previous_t=5,
    current_t=6,
    start_top=50,
    start_left=50,
):
    if velocity_params is None:
        velocity_params = {}

    rows = []

    for shape_name in shapes:
        for true_dy in velocity_values:
            for true_dx in velocity_values:
                case = make_unblurred_spike_case_from_velocity(
                    velocity_y=true_dy,
                    velocity_x=true_dx,
                    shape_name=shape_name,
                    start_top=start_top,
                    start_left=start_left,
                )

                result = estimate_velocity_with_explicit_motion_pairs_and_inhibition(
                    previous_spikes=case["spikes"][previous_t],
                    current_spikes=case["spikes"][current_t],
                    max_offset=max_offset,
                    **velocity_params,
                )

                rows.append({
                    "shape": shape_name,
                    "true_dy": true_dy,
                    "true_dx": true_dx,
                    "estimated_dy": result["estimated_dy"],
                    "estimated_dx": result["estimated_dx"],
                    "correct": (
                        result["estimated_dy"] == true_dy
                        and result["estimated_dx"] == true_dx
                    ),
                    "num_velocity_outputs": len(result["velocity_summary"]),
                })

    return pd.DataFrame(rows)

def plot_velocity_module_summary_v2(case, result, previous_t=5, current_t=6, max_offset=6):
    previous_spikes = case["spikes"][previous_t]
    current_spikes = case["spikes"][current_t]

    lif_events = result["lif_events"]
    velocity_input = result["velocity_input"]
    total_velocity_spikes = result["velocity_layer"]["velocity_spike_history"].sum(dim=0)

    true_dy = case["velocity_y"]
    true_dx = case["velocity_x"]

    estimated_dy = result["estimated_dy"]
    estimated_dx = result["estimated_dx"]

    extent = [
        -max_offset - 0.5,
        max_offset + 0.5,
        max_offset + 0.5,
        -max_offset - 0.5,
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(previous_spikes, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Previous spikes\n t={previous_t}")

    axes[0, 1].imshow(current_spikes, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Current spikes\n t={current_t}")

    axes[0, 2].imshow(lif_events["off_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(
        f"OFF LIF\ncount={int(lif_events['off_spikes'].sum().item())}"
    )

    axes[0, 3].imshow(lif_events["on_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title(
        f"ON LIF\ncount={int(lif_events['on_spikes'].sum().item())}"
    )

    axes[1, 0].imshow(lif_events["same_spikes"], cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"SAME LIF\ncount={int(lif_events['same_spikes'].sum().item())}"
    )

    velocity_input_image = axes[1, 1].imshow(
        velocity_input,
        extent=extent,
    )
    axes[1, 1].set_title("Velocity input\nfrom MotionPair spikes")
    axes[1, 1].set_xlabel("dx")
    axes[1, 1].set_ylabel("dy")
    axes[1, 1].set_xticks(range(-max_offset, max_offset + 1, 2))
    axes[1, 1].set_yticks(range(-max_offset, max_offset + 1, 2))
    plt.colorbar(velocity_input_image, ax=axes[1, 1], fraction=0.046, pad=0.04)

    velocity_spike_image = axes[1, 2].imshow(
        total_velocity_spikes,
        extent=extent,
    )
    axes[1, 2].set_title("Total velocity spikes\nafter inhibition")
    axes[1, 2].set_xlabel("dx")
    axes[1, 2].set_ylabel("dy")
    axes[1, 2].set_xticks(range(-max_offset, max_offset + 1, 2))
    axes[1, 2].set_yticks(range(-max_offset, max_offset + 1, 2))
    plt.colorbar(velocity_spike_image, ax=axes[1, 2], fraction=0.046, pad=0.04)

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.05,
        0.75,
        "Velocity result",
        fontsize=14,
        fontweight="bold",
        transform=axes[1, 3].transAxes,
    )
    axes[1, 3].text(
        0.05,
        0.55,
        f"True velocity:\n dy={true_dy}, dx={true_dx}",
        fontsize=12,
        transform=axes[1, 3].transAxes,
    )
    axes[1, 3].text(
        0.05,
        0.35,
        f"Estimated velocity:\n dy={estimated_dy}, dx={estimated_dx}",
        fontsize=12,
        transform=axes[1, 3].transAxes,
    )
    axes[1, 3].text(
        0.05,
        0.15,
        f"Correct: {estimated_dy == true_dy and estimated_dx == true_dx}",
        fontsize=12,
        transform=axes[1, 3].transAxes,
    )

    for ax in axes.flatten():
        if ax is not axes[1, 3]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[1, 1].set_xticks(range(-max_offset, max_offset + 1, 2))
    axes[1, 1].set_yticks(range(-max_offset, max_offset + 1, 2))
    axes[1, 2].set_xticks(range(-max_offset, max_offset + 1, 2))
    axes[1, 2].set_yticks(range(-max_offset, max_offset + 1, 2))

    plt.suptitle(
        "LIF velocity module: events → motion pairs → velocity inhibition",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

def compute_motion_bound_object(
    previous_spikes,
    current_spikes,
    estimated_dy,
    estimated_dx,
):
    """
    Motion-based object binding.

    Given an estimated velocity (dy, dx), bind old-frame pixels to
    current-frame pixels that are consistent with that velocity.

    Returns:
        bound_previous:
            old-frame pixels that have a matching current-frame pixel

        bound_current:
            current-frame pixels that match an old-frame pixel

        stabilized_current_to_previous:
            current-frame object shifted back into previous-frame coordinates

    This is not yet a LIF layer. It is the binding/readout step that uses
    the winning velocity population.
    """

    previous_spikes = (previous_spikes > 0).float()
    current_spikes = (current_spikes > 0).float()

    height, width = previous_spikes.shape

    bound_previous = torch.zeros_like(previous_spikes)
    bound_current = torch.zeros_like(current_spikes)
    stabilized_current_to_previous = torch.zeros_like(previous_spikes)

    previous_positions = torch.nonzero(previous_spikes > 0, as_tuple=False)

    for pos in previous_positions:
        y = int(pos[0].item())
        x = int(pos[1].item())

        target_y = y + estimated_dy
        target_x = x + estimated_dx

        if not (0 <= target_y < height and 0 <= target_x < width):
            continue

        if current_spikes[target_y, target_x] > 0:
            bound_previous[y, x] = 1.0
            bound_current[target_y, target_x] = 1.0
            stabilized_current_to_previous[y, x] = 1.0

    return {
        "bound_previous": bound_previous,
        "bound_current": bound_current,
        "stabilized_current_to_previous": stabilized_current_to_previous,
    }

def plot_motion_binding_result(case, velocity_result, binding_result, previous_t=5, current_t=6):
    previous_spikes = case["spikes"][previous_t]
    current_spikes = case["spikes"][current_t]

    estimated_dy = velocity_result["estimated_dy"]
    estimated_dx = velocity_result["estimated_dx"]

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))

    axes[0].imshow(previous_spikes, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Previous spikes\n t={previous_t}")

    axes[1].imshow(current_spikes, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Current spikes\n t={current_t}")

    axes[2].imshow(binding_result["bound_previous"], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Bound previous\nold object pixels")

    axes[3].imshow(binding_result["bound_current"], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Bound current\nnew object pixels")

    axes[4].imshow(binding_result["stabilized_current_to_previous"], cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Stabilized object\ncurrent shifted back")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"Motion binding using estimated velocity dy={estimated_dy}, dx={estimated_dx}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

def shift_frame(frame, shift_y, shift_x):
    """
    Shift a 2D frame by integer pixels.

    Positive shift_y moves pixels down.
    Positive shift_x moves pixels right.
    """

    height, width = frame.shape
    shifted = torch.zeros_like(frame)

    source_y_start = max(0, -shift_y)
    source_y_end = min(height, height - shift_y)

    source_x_start = max(0, -shift_x)
    source_x_end = min(width, width - shift_x)

    target_y_start = max(0, shift_y)
    target_y_end = min(height, height + shift_y)

    target_x_start = max(0, shift_x)
    target_x_end = min(width, width + shift_x)

    if source_y_start < source_y_end and source_x_start < source_x_end:
        shifted[
            target_y_start:target_y_end,
            target_x_start:target_x_end,
        ] = frame[
            source_y_start:source_y_end,
            source_x_start:source_x_end,
        ]

    return shifted

def align_frames_using_velocity(
    spikes,
    estimated_dy,
    estimated_dx,
    reference_t=0,
):
    """
    Align all frames into the coordinate system of reference_t.

    If the object moves by (dy, dx) each frame, then frame t is shifted back by:
        -(t - reference_t) * dy
        -(t - reference_t) * dx
    """

    aligned_frames = []

    num_steps = spikes.shape[0]

    for t in range(num_steps):
        delta_t = t - reference_t

        shift_y = -delta_t * estimated_dy
        shift_x = -delta_t * estimated_dx

        aligned = shift_frame(
            frame=spikes[t],
            shift_y=shift_y,
            shift_x=shift_x,
        )

        aligned_frames.append(aligned)

    return torch.stack(aligned_frames, dim=0)

def accumulate_aligned_frames_lif(
    aligned_frames,
    beta=0.8,
    threshold=2.5,
    reset_mode="subtract",
):
    """
    LIF accumulation over aligned object frames.

    Each pixel has a LIF accumulator.
    Repeated aligned spikes at the same object-centered location
    push the membrane over threshold.
    """

    membrane = torch.zeros_like(aligned_frames[0])
    spike_history = []
    membrane_history = []

    for t in range(aligned_frames.shape[0]):
        spikes, membrane = lif_step(
            input_current=aligned_frames[t],
            membrane=membrane,
            beta=beta,
            threshold=threshold,
            reset_mode=reset_mode,
        )

        spike_history.append(spikes.clone())
        membrane_history.append(membrane.clone())

    return {
        "accumulated_spike_history": torch.stack(spike_history, dim=0),
        "accumulated_membrane_history": torch.stack(membrane_history, dim=0),
        "final_accumulated_spikes": spike_history[-1],
        "final_membrane": membrane_history[-1],
    }

def plot_temporal_alignment_and_accumulation(
    case,
    aligned_frames,
    accumulation_result,
):
    original_spikes = case["spikes"]
    accumulated_spike_history = accumulation_result["accumulated_spike_history"]
    final_membrane = accumulation_result["final_membrane"]

    num_steps = original_spikes.shape[0]

    fig, axes = plt.subplots(3, num_steps, figsize=(2.5 * num_steps, 7.5))

    for t in range(num_steps):
        axes[0, t].imshow(original_spikes[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"Original\nt={t}")

        axes[1, t].imshow(aligned_frames[t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(f"Aligned\nt={t}")

        axes[2, t].imshow(accumulated_spike_history[t], cmap="gray", vmin=0, vmax=1)
        axes[2, t].set_title(
            f"Accumulator spikes\nt={t}\ncount={int(accumulated_spike_history[t].sum().item())}"
        )

        for row in range(3):
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])

    plt.suptitle(
        "Motion-aligned temporal accumulation for shape recognition",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.imshow(final_membrane)
    plt.title("Final LIF membrane after repeated aligned frames")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()

def get_winning_motion_pair_sheet(result):
    """
    Extract the MotionPair[estimated_dy, estimated_dx, :, :] sheet.

    This sheet is already in object-centered coordinates:
        MotionPair[dy, dx, y, x] spikes when old position (y, x)
        matches current position (y + dy, x + dx).

    So the winning sheet can be interpreted as a stabilized object spike map.
    """

    estimated_dy = result["estimated_dy"]
    estimated_dx = result["estimated_dx"]

    offset_table = result["offset_table"]
    motion_pair_spikes = result["motion_pair_spikes"]

    matching_rows = offset_table[
        (offset_table["dy"] == estimated_dy)
        & (offset_table["dx"] == estimated_dx)
    ]

    if len(matching_rows) != 1:
        raise ValueError("Could not uniquely identify winning velocity in offset_table.")

    offset_index = int(matching_rows.iloc[0]["offset_index"])

    winning_sheet = motion_pair_spikes[offset_index]

    return {
        "stable_object_spikes": winning_sheet,
        "offset_index": offset_index,
        "estimated_dy": estimated_dy,
        "estimated_dx": estimated_dx,
    }

def plot_stable_object_from_motion_pairs(case, velocity_result, stable_object_result, previous_t=5, current_t=6):
    previous_spikes = case["spikes"][previous_t]
    current_spikes = case["spikes"][current_t]

    stable_object_spikes = stable_object_result["stable_object_spikes"]

    estimated_dy = stable_object_result["estimated_dy"]
    estimated_dx = stable_object_result["estimated_dx"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    axes[0].imshow(previous_spikes, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Previous input\n t={previous_t}")

    axes[1].imshow(current_spikes, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Current input\n t={current_t}")

    axes[2].imshow(stable_object_spikes, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"StableObject from\nMotionPair[{estimated_dy},{estimated_dx},:,:]"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Stable object layer from winning motion-pair sheet")
    plt.tight_layout()
    plt.show()

def update_object_memory_with_velocity_gate(
    previous_memory_trace,
    current_input_spikes,
    memory_to_input_dy,
    memory_to_input_dx,
    memory_membrane,
    trace_weight=1.0,
    input_weight=1.0,
    velocity_gate_weight=1.0,
    threshold_memory=2.5,
    beta_memory=0.8,
    reset_mode="subtract",
):
    """
    Velocity-gated object-memory update.

    ObjectMemory[y,x] represents a stable object-centered location.

    memory_to_input_dy, memory_to_input_dx tell us where that memory location
    should appear in the current input frame.

    For each ObjectMemory[y,x]:

        previous memory trace at (y,x)
        + current input at (y + memory_to_input_dy, x + memory_to_input_dx)
        + velocity gate
        -> ObjectMemory[y,x] spike

    The important correction:
        For frame t, we do NOT always look one step ahead.
        We look at the current predicted input position.
    """

    previous_memory_trace = (previous_memory_trace > 0).float()
    current_input_spikes = (current_input_spikes > 0).float()

    height, width = current_input_spikes.shape

    routed_current_input = torch.zeros_like(current_input_spikes)

    active_memory_positions = torch.nonzero(
        previous_memory_trace > 0,
        as_tuple=False,
    )

    for pos in active_memory_positions:
        y = int(pos[0].item())
        x = int(pos[1].item())

        source_y = y + memory_to_input_dy
        source_x = x + memory_to_input_dx

        if not (0 <= source_y < height and 0 <= source_x < width):
            continue

        if current_input_spikes[source_y, source_x] > 0:
            routed_current_input[y, x] = 1.0

    # Only locations with predicted current input should receive update current.
    memory_input_current = (
        trace_weight * previous_memory_trace
        + input_weight * routed_current_input
        + velocity_gate_weight * routed_current_input
    )

    # Require both old memory and a matched routed current input.
    memory_input_current = memory_input_current * previous_memory_trace * routed_current_input

    memory_spikes, memory_membrane = lif_step(
        input_current=memory_input_current,
        membrane=memory_membrane,
        beta=beta_memory,
        threshold=threshold_memory,
        reset_mode=reset_mode,
    )

    # Keep tracking the same object locations.
    new_memory_trace = previous_memory_trace.clone()

    return {
        "memory_spikes": memory_spikes,
        "memory_membrane": memory_membrane,
        "memory_trace": new_memory_trace,
        "routed_current_input": routed_current_input,
        "memory_input_current": memory_input_current,
    }

def run_velocity_gated_object_memory(
    case,
    max_offset=6,
    velocity_params=None,
    trace_weight=1.0,
    input_weight=1.0,
    velocity_gate_weight=1.0,
    threshold_memory=2.5,
    beta_memory=0.8,
    reset_mode="subtract",
):
    """
    Run velocity-gated ObjectMemory over all frames.

    Frame 0 initializes ObjectMemory.

    For each transition t-1 -> t:
        1. Estimate velocity from input frames.
        2. Add that velocity to cumulative displacement.
        3. Route current input back into the original ObjectMemory coordinates.
        4. Run LIF update.
    """

    if velocity_params is None:
        velocity_params = {}

    spikes = case["spikes"]
    num_steps = spikes.shape[0]

    memory_trace = (spikes[0] > 0).float()
    memory_membrane = torch.zeros_like(memory_trace)

    cumulative_dy = 0
    cumulative_dx = 0

    memory_spike_history = []
    memory_trace_history = []
    memory_membrane_history = []
    routed_input_history = []
    velocity_rows = []

    # Frame 0 initializes memory.
    memory_spike_history.append(memory_trace.clone())
    memory_trace_history.append(memory_trace.clone())
    memory_membrane_history.append(memory_membrane.clone())
    routed_input_history.append(memory_trace.clone())

    velocity_rows.append({
        "t": 0,
        "previous_t": None,
        "current_t": 0,
        "estimated_dy": None,
        "estimated_dx": None,
        "cumulative_dy": cumulative_dy,
        "cumulative_dx": cumulative_dx,
        "note": "initialize memory from first frame",
    })

    for t in range(1, num_steps):
        velocity_result = estimate_velocity_with_explicit_motion_pairs_and_inhibition(
            previous_spikes=spikes[t - 1],
            current_spikes=spikes[t],
            max_offset=max_offset,
            **velocity_params,
        )

        estimated_dy = velocity_result["estimated_dy"]
        estimated_dx = velocity_result["estimated_dx"]

        if estimated_dy is None or estimated_dx is None:
            memory_spikes = torch.zeros_like(memory_trace)
            routed_current_input = torch.zeros_like(memory_trace)
            memory_input_current = torch.zeros_like(memory_trace)
        else:
            cumulative_dy += estimated_dy
            cumulative_dx += estimated_dx

            update_result = update_object_memory_with_velocity_gate(
                previous_memory_trace=memory_trace,
                current_input_spikes=spikes[t],
                memory_to_input_dy=cumulative_dy,
                memory_to_input_dx=cumulative_dx,
                memory_membrane=memory_membrane,
                trace_weight=trace_weight,
                input_weight=input_weight,
                velocity_gate_weight=velocity_gate_weight,
                threshold_memory=threshold_memory,
                beta_memory=beta_memory,
                reset_mode=reset_mode,
            )

            memory_spikes = update_result["memory_spikes"]
            memory_membrane = update_result["memory_membrane"]
            memory_trace = update_result["memory_trace"]
            routed_current_input = update_result["routed_current_input"]
            memory_input_current = update_result["memory_input_current"]

        memory_spike_history.append(memory_spikes.clone())
        memory_trace_history.append(memory_trace.clone())
        memory_membrane_history.append(memory_membrane.clone())
        routed_input_history.append(routed_current_input.clone())

        velocity_rows.append({
            "t": t,
            "previous_t": t - 1,
            "current_t": t,
            "estimated_dy": estimated_dy,
            "estimated_dx": estimated_dx,
            "cumulative_dy": cumulative_dy,
            "cumulative_dx": cumulative_dx,
            "note": "velocity-gated memory update",
        })

    return {
        "memory_spike_history": torch.stack(memory_spike_history, dim=0),
        "memory_trace_history": torch.stack(memory_trace_history, dim=0),
        "memory_membrane_history": torch.stack(memory_membrane_history, dim=0),
        "routed_input_history": torch.stack(routed_input_history, dim=0),
        "velocity_table": pd.DataFrame(velocity_rows),
    }

def plot_object_memory_result(case, object_memory_result):
    input_spikes = case["spikes"]
    memory_spike_history = object_memory_result["memory_spike_history"]
    memory_trace_history = object_memory_result["memory_trace_history"]
    routed_input_history = object_memory_result["routed_input_history"]

    num_steps = input_spikes.shape[0]

    fig, axes = plt.subplots(4, num_steps, figsize=(2.5 * num_steps, 10))

    for t in range(num_steps):
        axes[0, t].imshow(input_spikes[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"Input spikes\n t={t}")

        axes[1, t].imshow(routed_input_history[t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(
            f"Velocity-routed\ncurrent input t={t}"
        )

        axes[2, t].imshow(memory_spike_history[t], cmap="gray", vmin=0, vmax=1)
        axes[2, t].set_title(
            f"ObjectMemory spikes\n t={t}\ncount={int(memory_spike_history[t].sum().item())}"
        )

        axes[3, t].imshow(memory_trace_history[t], cmap="gray", vmin=0, vmax=1)
        axes[3, t].set_title(
            f"ObjectMemory trace\n t={t}"
        )

        for row in range(4):
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])

    plt.suptitle(
        "Velocity-gated object memory: moving input, stable memory",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

def get_velocity_gate_from_result(
    velocity_result,
    use_total_spikes=True,
):
    """
    Return a velocity gate grid from the velocity layer.

    Shape:
        [2*max_offset+1, 2*max_offset+1]

    Entry [i,j] corresponds to Velocity[dy,dx].

    This avoids decoding one Python velocity winner.
    The object-memory routing will use all active velocity populations.
    """

    velocity_spike_history = velocity_result["velocity_layer"]["velocity_spike_history"]

    if use_total_spikes:
        velocity_gate = velocity_spike_history.sum(dim=0)
        velocity_gate = (velocity_gate > 0).float()
    else:
        velocity_gate = velocity_spike_history[-1]
        velocity_gate = (velocity_gate > 0).float()

    return velocity_gate

def route_current_input_with_velocity_population(
    previous_memory_trace,
    current_input_spikes,
    velocity_gate,
    max_offset=6,
):
    """
    Route current input into ObjectMemory coordinates using all active
    velocity populations in parallel.

    For each ObjectMemory[y,x] and each possible velocity (dy,dx):

        Velocity[dy,dx]
        +
        CurrentInput[y+dy, x+dx]
        ->
        RoutedInput[y,x]

    No decoded estimated_dy/estimated_dx is used.
    """

    previous_memory_trace = (previous_memory_trace > 0).float()
    current_input_spikes = (current_input_spikes > 0).float()
    velocity_gate = (velocity_gate > 0).float()

    height, width = current_input_spikes.shape

    routed_current_input = torch.zeros_like(current_input_spikes)

    active_memory_positions = torch.nonzero(
        previous_memory_trace > 0,
        as_tuple=False,
    )

    active_velocity_positions = torch.nonzero(
        velocity_gate > 0,
        as_tuple=False,
    )

    for memory_pos in active_memory_positions:
        y = int(memory_pos[0].item())
        x = int(memory_pos[1].item())

        for velocity_pos in active_velocity_positions:
            velocity_i = int(velocity_pos[0].item())
            velocity_j = int(velocity_pos[1].item())

            dy = velocity_i - max_offset
            dx = velocity_j - max_offset

            source_y = y + dy
            source_x = x + dx

            if not (0 <= source_y < height and 0 <= source_x < width):
                continue

            if current_input_spikes[source_y, source_x] > 0:
                routed_current_input[y, x] = 1.0

    return routed_current_input

def run_parallel_velocity_gated_object_memory(
    case,
    max_offset=6,
    velocity_params=None,
    trace_weight=1.0,
    input_weight=1.0,
    threshold_memory=1.5,
    beta_memory=0.8,
    reset_mode="subtract",
):
    """
    Run ObjectMemory using parallel velocity-gated routes.

    Important:
        This version does NOT decode estimated_dy, estimated_dx
        to choose a single route.

    Instead:
        velocity layer spikes create a velocity_gate grid
        all active velocity routes are applied in parallel
    """

    if velocity_params is None:
        velocity_params = {}

    spikes = case["spikes"]
    num_steps = spikes.shape[0]

    memory_trace = (spikes[0] > 0).float()
    memory_membrane = torch.zeros_like(memory_trace)

    memory_spike_history = []
    memory_trace_history = []
    memory_membrane_history = []
    routed_input_history = []
    velocity_gate_history = []
    velocity_rows = []

    memory_spike_history.append(memory_trace.clone())
    memory_trace_history.append(memory_trace.clone())
    memory_membrane_history.append(memory_membrane.clone())
    routed_input_history.append(memory_trace.clone())

    velocity_gate_history.append(
        torch.zeros((2 * max_offset + 1, 2 * max_offset + 1), dtype=torch.float32)
    )

    velocity_rows.append({
        "t": 0,
        "previous_t": None,
        "current_t": 0,
        "num_active_velocity_gates": 0,
        "note": "initialize memory from first frame",
    })

    for t in range(1, num_steps):
        velocity_result = estimate_velocity_with_explicit_motion_pairs_and_inhibition(
            previous_spikes=spikes[t - 1],
            current_spikes=spikes[t],
            max_offset=max_offset,
            **velocity_params,
        )

        velocity_gate = get_velocity_gate_from_result(
            velocity_result=velocity_result,
            use_total_spikes=True,
        )

        routed_current_input = route_current_input_with_velocity_population(
            previous_memory_trace=memory_trace,
            current_input_spikes=spikes[t],
            velocity_gate=velocity_gate,
            max_offset=max_offset,
        )

        update_result = update_object_memory_from_routed_input(
            previous_memory_trace=memory_trace,
            routed_current_input=routed_current_input,
            memory_membrane=memory_membrane,
            trace_weight=trace_weight,
            input_weight=input_weight,
            threshold_memory=threshold_memory,
            beta_memory=beta_memory,
            reset_mode=reset_mode,
        )

        memory_spikes = update_result["memory_spikes"]
        memory_membrane = update_result["memory_membrane"]
        memory_trace = update_result["memory_trace"]

        memory_spike_history.append(memory_spikes.clone())
        memory_trace_history.append(memory_trace.clone())
        memory_membrane_history.append(memory_membrane.clone())
        routed_input_history.append(routed_current_input.clone())
        velocity_gate_history.append(velocity_gate.clone())

        velocity_rows.append({
            "t": t,
            "previous_t": t - 1,
            "current_t": t,
            "num_active_velocity_gates": int(velocity_gate.sum().item()),
            "note": "parallel velocity-gated memory update",
        })

    return {
        "memory_spike_history": torch.stack(memory_spike_history, dim=0),
        "memory_trace_history": torch.stack(memory_trace_history, dim=0),
        "memory_membrane_history": torch.stack(memory_membrane_history, dim=0),
        "routed_input_history": torch.stack(routed_input_history, dim=0),
        "velocity_gate_history": torch.stack(velocity_gate_history, dim=0),
        "velocity_table": pd.DataFrame(velocity_rows),
    }

def initialize_displacement_gate(max_displacement):
    """
    Displacement[D_y, D_x] population.

    Initially the object memory is anchored to frame 0, so cumulative
    displacement is zero.
    """

    size = 2 * max_displacement + 1
    displacement_gate = torch.zeros((size, size), dtype=torch.float32)

    center = max_displacement
    displacement_gate[center, center] = 1.0

    return displacement_gate

def update_displacement_gate_from_velocity_gate(
    previous_displacement_gate,
    velocity_gate,
    max_velocity=6,
    max_displacement=36,
):
    """
    Update cumulative displacement using velocity populations.

    Previous Displacement[D_y, D_x]
    +
    Velocity[dy, dx]
    ->
    New Displacement[D_y + dy, D_x + dx]

    This is a population update, not Python decoding one velocity.
    """

    previous_displacement_gate = (previous_displacement_gate > 0).float()
    velocity_gate = (velocity_gate > 0).float()

    new_displacement_gate = torch.zeros_like(previous_displacement_gate)

    active_displacements = torch.nonzero(
        previous_displacement_gate > 0,
        as_tuple=False,
    )

    active_velocities = torch.nonzero(
        velocity_gate > 0,
        as_tuple=False,
    )

    for displacement_pos in active_displacements:
        disp_i = int(displacement_pos[0].item())
        disp_j = int(displacement_pos[1].item())

        current_D_y = disp_i - max_displacement
        current_D_x = disp_j - max_displacement

        for velocity_pos in active_velocities:
            vel_i = int(velocity_pos[0].item())
            vel_j = int(velocity_pos[1].item())

            dy = vel_i - max_velocity
            dx = vel_j - max_velocity

            new_D_y = current_D_y + dy
            new_D_x = current_D_x + dx

            new_i = new_D_y + max_displacement
            new_j = new_D_x + max_displacement

            if not (0 <= new_i < new_displacement_gate.shape[0]):
                continue

            if not (0 <= new_j < new_displacement_gate.shape[1]):
                continue

            new_displacement_gate[new_i, new_j] = 1.0

    return new_displacement_gate

def route_current_input_with_displacement_population(
    previous_memory_trace,
    current_input_spikes,
    displacement_gate,
    max_displacement=36,
):
    """
    Route current input into ObjectMemory coordinates using cumulative
    displacement populations.

    For each ObjectMemory[y,x] and active displacement (D_y,D_x):

        Displacement[D_y,D_x]
        +
        CurrentInput[y + D_y, x + D_x]
        ->
        RoutedInput[y,x]

    This is what keeps ObjectMemory anchored to frame 0.
    """

    previous_memory_trace = (previous_memory_trace > 0).float()
    current_input_spikes = (current_input_spikes > 0).float()
    displacement_gate = (displacement_gate > 0).float()

    height, width = current_input_spikes.shape

    routed_current_input = torch.zeros_like(current_input_spikes)

    active_memory_positions = torch.nonzero(
        previous_memory_trace > 0,
        as_tuple=False,
    )

    active_displacements = torch.nonzero(
        displacement_gate > 0,
        as_tuple=False,
    )

    for memory_pos in active_memory_positions:
        y = int(memory_pos[0].item())
        x = int(memory_pos[1].item())

        for displacement_pos in active_displacements:
            disp_i = int(displacement_pos[0].item())
            disp_j = int(displacement_pos[1].item())

            D_y = disp_i - max_displacement
            D_x = disp_j - max_displacement

            source_y = y + D_y
            source_x = x + D_x

            if not (0 <= source_y < height and 0 <= source_x < width):
                continue

            if current_input_spikes[source_y, source_x] > 0:
                routed_current_input[y, x] = 1.0

    return routed_current_input

def update_object_memory_from_routed_input(
    previous_memory_trace,
    routed_current_input,
    memory_membrane,
    trace_weight=1.0,
    input_weight=1.0,
    threshold_memory=1.5,
    beta_memory=0.8,
    reset_mode="subtract",
):
    """
    LIF object-memory update after displacement-gated routing.

    ObjectMemory[y,x] receives:
        previous memory trace at (y,x)
        routed current input at (y,x)

    If both are present, the neuron gets enough current to spike.
    """

    previous_memory_trace = (previous_memory_trace > 0).float()
    routed_current_input = (routed_current_input > 0).float()

    memory_input_current = (
        trace_weight * previous_memory_trace
        + input_weight * routed_current_input
    )

    memory_input_current = (
        memory_input_current
        * previous_memory_trace
        * routed_current_input
    )

    memory_spikes, memory_membrane = lif_step(
        input_current=memory_input_current,
        membrane=memory_membrane,
        beta=beta_memory,
        threshold=threshold_memory,
        reset_mode=reset_mode,
    )

    new_memory_trace = previous_memory_trace.clone()

    return {
        "memory_spikes": memory_spikes,
        "memory_membrane": memory_membrane,
        "memory_trace": new_memory_trace,
        "memory_input_current": memory_input_current,
    }

def run_displacement_gated_object_memory(
    case,
    max_velocity=6,
    max_displacement=36,
    velocity_params=None,
    trace_weight=1.0,
    input_weight=1.0,
    threshold_memory=1.5,
    beta_memory=0.8,
    reset_mode="subtract",
):
    """
    Run ObjectMemory using:

        Velocity populations
        ->
        Displacement populations
        ->
        displacement-gated routing into ObjectMemory

    This avoids Python decoding one velocity for routing.
    """

    if velocity_params is None:
        velocity_params = {}

    spikes = case["spikes"]
    num_steps = spikes.shape[0]

    memory_trace = (spikes[0] > 0).float()
    memory_membrane = torch.zeros_like(memory_trace)

    displacement_gate = initialize_displacement_gate(
        max_displacement=max_displacement,
    )

    memory_spike_history = []
    memory_trace_history = []
    memory_membrane_history = []
    routed_input_history = []
    velocity_gate_history = []
    displacement_gate_history = []
    table_rows = []

    memory_spike_history.append(memory_trace.clone())
    memory_trace_history.append(memory_trace.clone())
    memory_membrane_history.append(memory_membrane.clone())
    routed_input_history.append(memory_trace.clone())

    velocity_gate_history.append(
        torch.zeros((2 * max_velocity + 1, 2 * max_velocity + 1), dtype=torch.float32)
    )

    displacement_gate_history.append(displacement_gate.clone())

    table_rows.append({
        "t": 0,
        "previous_t": None,
        "current_t": 0,
        "num_active_velocity_gates": 0,
        "num_active_displacement_gates": int(displacement_gate.sum().item()),
        "note": "initialize memory and zero displacement",
    })

    for t in range(1, num_steps):
        velocity_result = estimate_velocity_with_explicit_motion_pairs_and_inhibition(
            previous_spikes=spikes[t - 1],
            current_spikes=spikes[t],
            max_offset=max_velocity,
            **velocity_params,
        )

        velocity_gate = get_velocity_gate_from_result(
            velocity_result=velocity_result,
            use_total_spikes=True,
        )

        displacement_gate = update_displacement_gate_from_velocity_gate(
            previous_displacement_gate=displacement_gate,
            velocity_gate=velocity_gate,
            max_velocity=max_velocity,
            max_displacement=max_displacement,
        )

        routed_current_input = route_current_input_with_displacement_population(
            previous_memory_trace=memory_trace,
            current_input_spikes=spikes[t],
            displacement_gate=displacement_gate,
            max_displacement=max_displacement,
        )

        update_result = update_object_memory_from_routed_input(
            previous_memory_trace=memory_trace,
            routed_current_input=routed_current_input,
            memory_membrane=memory_membrane,
            trace_weight=trace_weight,
            input_weight=input_weight,
            threshold_memory=threshold_memory,
            beta_memory=beta_memory,
            reset_mode=reset_mode,
        )

        memory_spikes = update_result["memory_spikes"]
        memory_membrane = update_result["memory_membrane"]
        memory_trace = update_result["memory_trace"]

        memory_spike_history.append(memory_spikes.clone())
        memory_trace_history.append(memory_trace.clone())
        memory_membrane_history.append(memory_membrane.clone())
        routed_input_history.append(routed_current_input.clone())
        velocity_gate_history.append(velocity_gate.clone())
        displacement_gate_history.append(displacement_gate.clone())

        table_rows.append({
            "t": t,
            "previous_t": t - 1,
            "current_t": t,
            "num_active_velocity_gates": int(velocity_gate.sum().item()),
            "num_active_displacement_gates": int(displacement_gate.sum().item()),
            "note": "displacement-gated memory update",
        })

    return {
        "memory_spike_history": torch.stack(memory_spike_history, dim=0),
        "memory_trace_history": torch.stack(memory_trace_history, dim=0),
        "memory_membrane_history": torch.stack(memory_membrane_history, dim=0),
        "routed_input_history": torch.stack(routed_input_history, dim=0),
        "velocity_gate_history": torch.stack(velocity_gate_history, dim=0),
        "displacement_gate_history": torch.stack(displacement_gate_history, dim=0),
        "table": pd.DataFrame(table_rows),
    }

def test_displacement_object_memory_constant_velocities(
    velocities,
    shape_name="seven",
    start_top=50,
    start_left=30,
    max_velocity=6,
    max_displacement=36,
    velocity_params=None,
):
    if velocity_params is None:
        velocity_params = {}

    rows = []

    for true_dy, true_dx in velocities:
        case = make_unblurred_spike_case_from_velocity(
            velocity_y=true_dy,
            velocity_x=true_dx,
            shape_name=shape_name,
            start_top=start_top,
            start_left=start_left,
        )

        result = run_displacement_gated_object_memory(
            case=case,
            max_velocity=max_velocity,
            max_displacement=max_displacement,
            velocity_params=velocity_params,
            trace_weight=1.0,
            input_weight=1.0,
            threshold_memory=1.5,
            beta_memory=0.8,
            reset_mode="subtract",
        )

        memory_spike_counts = result["memory_spike_history"].sum(dim=(1, 2))
        routed_input_counts = result["routed_input_history"].sum(dim=(1, 2))

        final_memory_trace_count = int(result["memory_trace_history"][-1].sum().item())

        rows.append({
            "true_dy": true_dy,
            "true_dx": true_dx,
            "final_memory_trace_count": final_memory_trace_count,
            "min_routed_input_count_after_t0": int(routed_input_counts[1:].min().item()),
            "max_routed_input_count_after_t0": int(routed_input_counts[1:].max().item()),
            "min_memory_spike_count_after_t0": int(memory_spike_counts[1:].min().item()),
            "max_memory_spike_count_after_t0": int(memory_spike_counts[1:].max().item()),
            "num_steps": case["spikes"].shape[0],
        })

    return pd.DataFrame(rows)

def make_variable_velocity_spike_case(
    velocity_sequence,
    shape_name="seven",
    start_top=50,
    start_left=30,
):
    """
    Make a moving shape where the velocity can change at each transition.

    velocity_sequence has length STEPS - 1.

    Example for 7 frames:
        velocity_sequence = [
            (0, 2),
            (0, 3),
            (0, 1),
            (0, 4),
            (0, 2),
            (0, 5),
        ]

    This means:
        frame 0 starts at start_top, start_left
        frame 1 moves by velocity_sequence[0]
        frame 2 moves by velocity_sequence[1]
        ...
    """

    shape = SHAPE_TEMPLATES[shape_name]

    frames = []
    origins = []

    current_top = start_top
    current_left = start_left

    for t in range(STEPS):
        frame = torch.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=torch.float32)

        shape_height, shape_width = shape.shape

        top = int(current_top)
        left = int(current_left)

        y0 = max(0, top)
        y1 = min(CANVAS_HEIGHT, top + shape_height)

        x0 = max(0, left)
        x1 = min(CANVAS_WIDTH, left + shape_width)

        shape_y0 = y0 - top
        shape_y1 = shape_y0 + (y1 - y0)

        shape_x0 = x0 - left
        shape_x1 = shape_x0 + (x1 - x0)

        if y0 < y1 and x0 < x1:
            frame[y0:y1, x0:x1] = shape[shape_y0:shape_y1, shape_x0:shape_x1] * INTENSITY

        frames.append(frame)
        origins.append((top, left))

        if t < len(velocity_sequence):
            dy, dx = velocity_sequence[t]
            current_top += dy
            current_left += dx

    frames = torch.stack(frames, dim=0)
    spikes = (frames > 0).float()

    return {
        "shape_name": shape_name,
        "velocity_sequence": velocity_sequence,
        "frames": frames,
        "spikes": spikes,
        "origins": origins,
    }

def summarize_displacement_gate_history(
    displacement_gate_history,
    max_displacement=36,
):
    rows = []

    for t in range(displacement_gate_history.shape[0]):
        active_positions = torch.nonzero(
            displacement_gate_history[t] > 0,
            as_tuple=False,
        )

        for pos in active_positions:
            i = int(pos[0].item())
            j = int(pos[1].item())

            D_y = i - max_displacement
            D_x = j - max_displacement

            rows.append({
                "t": t,
                "D_y": D_y,
                "D_x": D_x,
            })

    return pd.DataFrame(rows)

def expected_displacements_from_velocity_sequence(velocity_sequence):
    rows = []

    D_y = 0
    D_x = 0

    rows.append({
        "t": 0,
        "expected_D_y": D_y,
        "expected_D_x": D_x,
    })

    for t, (dy, dx) in enumerate(velocity_sequence, start=1):
        D_y += dy
        D_x += dx

        rows.append({
            "t": t,
            "expected_D_y": D_y,
            "expected_D_x": D_x,
        })

    return pd.DataFrame(rows)

def plot_object_memory_result_zoomed(
    case,
    object_memory_result,
    padding=8,
):
    input_spikes = case["spikes"]
    memory_spike_history = object_memory_result["memory_spike_history"]
    memory_trace_history = object_memory_result["memory_trace_history"]
    routed_input_history = object_memory_result["routed_input_history"]

    crop = get_activity_crop_from_tensors(
        [
            input_spikes,
            memory_spike_history,
            memory_trace_history,
            routed_input_history,
        ],
        padding=padding,
    )

    if crop is None:
        print("No activity found to plot.")
        return

    y_min, y_max, x_min, x_max = crop

    num_steps = input_spikes.shape[0]

    fig, axes = plt.subplots(4, num_steps, figsize=(2.5 * num_steps, 10))

    for t in range(num_steps):
        axes[0, t].imshow(
            input_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, t].set_title(f"Input\n t={t}")

        axes[1, t].imshow(
            routed_input_history[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, t].set_title(f"Routed input\n t={t}")

        axes[2, t].imshow(
            memory_spike_history[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[2, t].set_title(
            f"Memory spikes\n t={t}\ncount={int(memory_spike_history[t].sum().item())}"
        )

        axes[3, t].imshow(
            memory_trace_history[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[3, t].set_title(f"Memory trace\n t={t}")

        for row in range(4):
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])

    plt.suptitle(
        f"Zoomed velocity/displacement-gated object memory\ncrop y={y_min}:{y_max}, x={x_min}:{x_max}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

def get_activity_crop_from_tensors(
    tensor_list,
    padding=12,
    min_height=55,
    min_width=55,
):
    """
    Compute one crop box that contains activity from a list of tensors,
    but force the crop to have a minimum height/width so motion is visible.
    """

    activity = None

    for tensor in tensor_list:
        if tensor.dim() == 3:
            current_activity = tensor.sum(dim=0) > 0
        elif tensor.dim() == 2:
            current_activity = tensor > 0
        else:
            raise ValueError("Expected tensor with shape [T,H,W] or [H,W].")

        if activity is None:
            activity = current_activity.clone()
        else:
            activity = activity | current_activity

    active_positions = torch.nonzero(activity > 0, as_tuple=False)

    if active_positions.numel() == 0:
        return None

    height, width = activity.shape

    y_min = max(0, int(active_positions[:, 0].min().item()) - padding)
    y_max = min(height, int(active_positions[:, 0].max().item()) + padding + 1)

    x_min = max(0, int(active_positions[:, 1].min().item()) - padding)
    x_max = min(width, int(active_positions[:, 1].max().item()) + padding + 1)

    crop_height = y_max - y_min
    crop_width = x_max - x_min

    if crop_height < min_height:
        extra = min_height - crop_height
        y_min = max(0, y_min - extra // 2)
        y_max = min(height, y_max + extra - extra // 2)

    if crop_width < min_width:
        extra = min_width - crop_width
        x_min = max(0, x_min - extra // 2)
        x_max = min(width, x_max + extra - extra // 2)

    return y_min, y_max, x_min, x_max

def compute_lif_local_feature_map_over_time(
    input_spike_history,
    kernel,
    threshold=None,
    beta=0.8,
    reset_mode="subtract",
):
    """
    Compute a spiking local feature map over time.

    input_spike_history:
        [T, H, W]

    kernel:
        small binary receptive field, e.g.
            horizontal: [[1, 1, 1]]
            vertical:   [[1], [1], [1]]

    Output:
        feature_spike_history[t, y, x]
            feature neuron at (y,x) spikes at time t
    """

    input_spike_history = (input_spike_history > 0).float()

    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel_height, kernel_width = kernel.shape

    num_steps, height, width = input_spike_history.shape

    if threshold is None:
        threshold = float(kernel.sum().item()) - 0.5

    feature_membrane = torch.zeros((height, width), dtype=torch.float32)

    feature_spike_history = []
    feature_membrane_history = []

    for t in range(num_steps):
        current_feature_input = torch.zeros((height, width), dtype=torch.float32)

        for y in range(height - kernel_height + 1):
            for x in range(width - kernel_width + 1):
                patch = input_spike_history[
                    t,
                    y:y + kernel_height,
                    x:x + kernel_width,
                ]

                feature_current = (patch * kernel).sum()

                center_y = y + kernel_height // 2
                center_x = x + kernel_width // 2

                current_feature_input[center_y, center_x] = feature_current

        feature_spikes, feature_membrane = lif_step(
            input_current=current_feature_input,
            membrane=feature_membrane,
            beta=beta,
            threshold=threshold,
            reset_mode=reset_mode,
        )

        feature_spike_history.append(feature_spikes.clone())
        feature_membrane_history.append(feature_membrane.clone())

    return {
        "feature_spike_history": torch.stack(feature_spike_history, dim=0),
        "feature_membrane_history": torch.stack(feature_membrane_history, dim=0),
    }

def compute_basic_spiking_shape_features(
    object_memory_spike_history,
    beta=0.8,
):
    """
    Compute basic reusable spiking feature maps from ObjectMemory spikes.
    """

    horizontal_result = compute_lif_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[[1, 1, 1]],
        threshold=2.5,
        beta=beta,
        reset_mode="subtract",
    )

    vertical_result = compute_lif_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1],
            [1],
            [1],
        ],
        threshold=2.5,
        beta=beta,
        reset_mode="subtract",
    )

    corner_1_result = compute_lif_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [0, 1],
        ],
        threshold=2.5,
        beta=beta,
        reset_mode="subtract",
    )

    corner_2_result = compute_lif_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [1, 0],
        ],
        threshold=2.5,
        beta=beta,
        reset_mode="subtract",
    )

    corner_3_result = compute_lif_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 0],
            [1, 1],
        ],
        threshold=2.5,
        beta=beta,
        reset_mode="subtract",
    )

    corner_4_result = compute_lif_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [0, 1],
            [1, 1],
        ],
        threshold=2.5,
        beta=beta,
        reset_mode="subtract",
    )

    corner_spike_history = torch.clamp(
        corner_1_result["feature_spike_history"]
        + corner_2_result["feature_spike_history"]
        + corner_3_result["feature_spike_history"]
        + corner_4_result["feature_spike_history"],
        min=0,
        max=1,
    )

    return {
        "horizontal": horizontal_result,
        "vertical": vertical_result,
        "corner_1": corner_1_result,
        "corner_2": corner_2_result,
        "corner_3": corner_3_result,
        "corner_4": corner_4_result,
        "corner_spike_history": corner_spike_history,
    }

def plot_spiking_feature_maps(
    object_memory_spike_history,
    features_result,
    padding=8,
):
    """
    Plot ObjectMemory spikes next to basic feature maps over time.

    Rows:
        1. ObjectMemory spikes
        2. Horizontal feature spikes
        3. Vertical feature spikes
        4. Corner feature spikes
    """

    horizontal_spikes = features_result["horizontal"]["feature_spike_history"]
    vertical_spikes = features_result["vertical"]["feature_spike_history"]
    corner_spikes = features_result["corner_spike_history"]

    crop = get_activity_crop_from_tensors(
        [
            object_memory_spike_history,
            horizontal_spikes,
            vertical_spikes,
            corner_spikes,
        ],
        padding=padding,
        min_height=55,
        min_width=55,
    )

    if crop is None:
        print("No activity found to plot.")
        return

    y_min, y_max, x_min, x_max = crop

    num_steps = object_memory_spike_history.shape[0]

    fig, axes = plt.subplots(4, num_steps, figsize=(2.5 * num_steps, 10))

    for t in range(num_steps):
        axes[0, t].imshow(
            object_memory_spike_history[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, t].set_title(
            f"ObjectMemory\n t={t}\ncount={int(object_memory_spike_history[t].sum().item())}"
        )

        axes[1, t].imshow(
            horizontal_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, t].set_title(
            f"Horizontal\n t={t}\ncount={int(horizontal_spikes[t].sum().item())}"
        )

        axes[2, t].imshow(
            vertical_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[2, t].set_title(
            f"Vertical\n t={t}\ncount={int(vertical_spikes[t].sum().item())}"
        )

        axes[3, t].imshow(
            corner_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[3, t].set_title(
            f"Corner\n t={t}\ncount={int(corner_spikes[t].sum().item())}"
        )

        for row in range(4):
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])

    plt.suptitle(
        "Spiking feature maps from stabilized ObjectMemory",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

def compute_strict_local_feature_map_over_time(
    input_spike_history,
    kernel,
    threshold=None,
):
    """
    Strict instantaneous local feature detector.

    This detects spatial patterns at each time step.
    It does NOT accumulate partial feature evidence over time.

    input_spike_history:
        [T, H, W]

    kernel:
        small binary receptive field

    Output:
        feature_spike_history[t, y, x]
    """

    input_spike_history = (input_spike_history > 0).float()

    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel_height, kernel_width = kernel.shape

    num_steps, height, width = input_spike_history.shape

    if threshold is None:
        threshold = float(kernel.sum().item()) - 0.5

    feature_spike_history = []

    for t in range(num_steps):
        feature_spikes = torch.zeros((height, width), dtype=torch.float32)

        for y in range(height - kernel_height + 1):
            for x in range(width - kernel_width + 1):
                patch = input_spike_history[
                    t,
                    y:y + kernel_height,
                    x:x + kernel_width,
                ]

                feature_current = (patch * kernel).sum()

                center_y = y + kernel_height // 2
                center_x = x + kernel_width // 2

                if feature_current >= threshold:
                    feature_spikes[center_y, center_x] = 1.0

        feature_spike_history.append(feature_spikes.clone())

    return {
        "feature_spike_history": torch.stack(feature_spike_history, dim=0),
    }

def compute_strict_basic_shape_features(
    object_memory_spike_history,
):
    """
    Strict reusable spatial feature maps from ObjectMemory spikes.
    No temporal accumulation yet.
    """

    horizontal_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[[1, 1, 1]],
        threshold=2.5,
    )

    vertical_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1],
            [1],
            [1],
        ],
        threshold=2.5,
    )

    corner_1_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [0, 1],
        ],
        threshold=2.5,
    )

    corner_2_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [1, 0],
        ],
        threshold=2.5,
    )

    corner_3_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 0],
            [1, 1],
        ],
        threshold=2.5,
    )

    corner_4_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [0, 1],
            [1, 1],
        ],
        threshold=2.5,
    )

    corner_spike_history = torch.clamp(
        corner_1_result["feature_spike_history"]
        + corner_2_result["feature_spike_history"]
        + corner_3_result["feature_spike_history"]
        + corner_4_result["feature_spike_history"],
        min=0,
        max=1,
    )

    return {
        "horizontal": horizontal_result,
        "vertical": vertical_result,
        "corner_1": corner_1_result,
        "corner_2": corner_2_result,
        "corner_3": corner_3_result,
        "corner_4": corner_4_result,
        "corner_spike_history": corner_spike_history,
    }

def compute_strict_basic_shape_features_with_diagonals(
    object_memory_spike_history,
):
    """
    Strict reusable spatial feature maps from ObjectMemory spikes.

    Features:
        horizontal
        vertical
        corner
        diagonal_down_right
        diagonal_down_left

    No temporal accumulation here.
    """

    horizontal_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[[1, 1, 1]],
        threshold=2.5,
    )

    vertical_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1],
            [1],
            [1],
        ],
        threshold=2.5,
    )

    diagonal_down_right_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        threshold=2.5,
    )

    diagonal_down_left_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        threshold=2.5,
    )

    corner_1_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [0, 1],
        ],
        threshold=2.5,
    )

    corner_2_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [1, 0],
        ],
        threshold=2.5,
    )

    corner_3_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 0],
            [1, 1],
        ],
        threshold=2.5,
    )

    corner_4_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [0, 1],
            [1, 1],
        ],
        threshold=2.5,
    )

    corner_spike_history = torch.clamp(
        corner_1_result["feature_spike_history"]
        + corner_2_result["feature_spike_history"]
        + corner_3_result["feature_spike_history"]
        + corner_4_result["feature_spike_history"],
        min=0,
        max=1,
    )

    return {
        "horizontal": horizontal_result,
        "vertical": vertical_result,
        "diagonal_down_right": diagonal_down_right_result,
        "diagonal_down_left": diagonal_down_left_result,
        "corner_1": corner_1_result,
        "corner_2": corner_2_result,
        "corner_3": corner_3_result,
        "corner_4": corner_4_result,
        "corner_spike_history": corner_spike_history,
    }

def plot_spiking_feature_maps_with_diagonals(
    object_memory_spike_history,
    features_result,
    padding=8,
):
    """
    Plot ObjectMemory spikes next to basic feature maps over time.

    Rows:
        1. ObjectMemory spikes
        2. Horizontal feature spikes
        3. Vertical feature spikes
        4. Diagonal down-right feature spikes
        5. Diagonal down-left feature spikes
        6. Corner feature spikes
    """

    horizontal_spikes = features_result["horizontal"]["feature_spike_history"]
    vertical_spikes = features_result["vertical"]["feature_spike_history"]
    diagonal_down_right_spikes = features_result["diagonal_down_right"]["feature_spike_history"]
    diagonal_down_left_spikes = features_result["diagonal_down_left"]["feature_spike_history"]
    corner_spikes = features_result["corner_spike_history"]

    crop = get_activity_crop_from_tensors(
        [
            object_memory_spike_history,
            horizontal_spikes,
            vertical_spikes,
            diagonal_down_right_spikes,
            diagonal_down_left_spikes,
            corner_spikes,
        ],
        padding=padding,
        min_height=55,
        min_width=55,
    )

    if crop is None:
        print("No activity found to plot.")
        return

    y_min, y_max, x_min, x_max = crop

    num_steps = object_memory_spike_history.shape[0]

    fig, axes = plt.subplots(6, num_steps, figsize=(2.5 * num_steps, 15))

    for t in range(num_steps):
        axes[0, t].imshow(
            object_memory_spike_history[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, t].set_title(
            f"ObjectMemory\n t={t}\ncount={int(object_memory_spike_history[t].sum().item())}"
        )

        axes[1, t].imshow(
            horizontal_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, t].set_title(
            f"Horizontal\n t={t}\ncount={int(horizontal_spikes[t].sum().item())}"
        )

        axes[2, t].imshow(
            vertical_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[2, t].set_title(
            f"Vertical\n t={t}\ncount={int(vertical_spikes[t].sum().item())}"
        )

        axes[3, t].imshow(
            diagonal_down_right_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[3, t].set_title(
            f"Diag down-right\n t={t}\ncount={int(diagonal_down_right_spikes[t].sum().item())}"
        )

        axes[4, t].imshow(
            diagonal_down_left_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[4, t].set_title(
            f"Diag down-left\n t={t}\ncount={int(diagonal_down_left_spikes[t].sum().item())}"
        )

        axes[5, t].imshow(
            corner_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[5, t].set_title(
            f"Corner\n t={t}\ncount={int(corner_spikes[t].sum().item())}"
        )

        for row in range(6):
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])

    plt.suptitle(
        "Strict spiking feature maps from stabilized ObjectMemory",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

def compute_strict_basic_shape_features_with_junctions(
    object_memory_spike_history,
):
    """
    Strict reusable spatial feature maps from ObjectMemory spikes.

    Features:
        horizontal
        vertical
        diagonal_down_right
        diagonal_down_left
        small_corner
        horizontal_to_diagonal_junction

    No temporal accumulation here.
    """

    horizontal_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[[1, 1, 1]],
        threshold=2.5,
    )

    vertical_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1],
            [1],
            [1],
        ],
        threshold=2.5,
    )

    diagonal_down_right_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        threshold=2.5,
    )

    diagonal_down_left_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        threshold=2.5,
    )

    # Small generic 2x2 corners.
    small_corner_1_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [0, 1],
        ],
        threshold=2.5,
    )

    small_corner_2_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1],
            [1, 0],
        ],
        threshold=2.5,
    )

    small_corner_3_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 0],
            [1, 1],
        ],
        threshold=2.5,
    )

    small_corner_4_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [0, 1],
            [1, 1],
        ],
        threshold=2.5,
    )

    small_corner_spike_history = torch.clamp(
        small_corner_1_result["feature_spike_history"]
        + small_corner_2_result["feature_spike_history"]
        + small_corner_3_result["feature_spike_history"]
        + small_corner_4_result["feature_spike_history"],
        min=0,
        max=1,
    )

    # Horizontal-to-diagonal junctions.
    # These are more relevant for a 7 than pure L-corners.
    junction_down_left_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        threshold=4.5,
    )

    junction_down_right_result = compute_strict_local_feature_map_over_time(
        input_spike_history=object_memory_spike_history,
        kernel=[
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
        ],
        threshold=4.5,
    )

    horizontal_to_diagonal_junction_spike_history = torch.clamp(
        junction_down_left_result["feature_spike_history"]
        + junction_down_right_result["feature_spike_history"],
        min=0,
        max=1,
    )

    return {
        "horizontal": horizontal_result,
        "vertical": vertical_result,
        "diagonal_down_right": diagonal_down_right_result,
        "diagonal_down_left": diagonal_down_left_result,

        "small_corner_1": small_corner_1_result,
        "small_corner_2": small_corner_2_result,
        "small_corner_3": small_corner_3_result,
        "small_corner_4": small_corner_4_result,
        "small_corner_spike_history": small_corner_spike_history,

        "junction_down_left": junction_down_left_result,
        "junction_down_right": junction_down_right_result,
        "horizontal_to_diagonal_junction_spike_history": horizontal_to_diagonal_junction_spike_history,
    }

def plot_spiking_feature_maps_with_junctions(
    object_memory_spike_history,
    features_result,
    padding=8,
):
    """
    Plot ObjectMemory spikes next to strict feature maps over time.

    Rows:
        1. ObjectMemory spikes
        2. Horizontal
        3. Vertical
        4. Diagonal down-right
        5. Diagonal down-left
        6. Small corner
        7. Horizontal-to-diagonal junction
    """

    horizontal_spikes = features_result["horizontal"]["feature_spike_history"]
    vertical_spikes = features_result["vertical"]["feature_spike_history"]
    diagonal_down_right_spikes = features_result["diagonal_down_right"]["feature_spike_history"]
    diagonal_down_left_spikes = features_result["diagonal_down_left"]["feature_spike_history"]
    small_corner_spikes = features_result["small_corner_spike_history"]
    junction_spikes = features_result["horizontal_to_diagonal_junction_spike_history"]

    crop = get_activity_crop_from_tensors(
        [
            object_memory_spike_history,
            horizontal_spikes,
            vertical_spikes,
            diagonal_down_right_spikes,
            diagonal_down_left_spikes,
            small_corner_spikes,
            junction_spikes,
        ],
        padding=padding,
        min_height=55,
        min_width=55,
    )

    if crop is None:
        print("No activity found to plot.")
        return

    y_min, y_max, x_min, x_max = crop
    num_steps = object_memory_spike_history.shape[0]

    fig, axes = plt.subplots(7, num_steps, figsize=(2.5 * num_steps, 17))

    for t in range(num_steps):
        axes[0, t].imshow(
            object_memory_spike_history[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, t].set_title(
            f"ObjectMemory\n t={t}\ncount={int(object_memory_spike_history[t].sum().item())}"
        )

        axes[1, t].imshow(
            horizontal_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, t].set_title(
            f"Horizontal\n t={t}\ncount={int(horizontal_spikes[t].sum().item())}"
        )

        axes[2, t].imshow(
            vertical_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[2, t].set_title(
            f"Vertical\n t={t}\ncount={int(vertical_spikes[t].sum().item())}"
        )

        axes[3, t].imshow(
            diagonal_down_right_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[3, t].set_title(
            f"Diag down-right\n t={t}\ncount={int(diagonal_down_right_spikes[t].sum().item())}"
        )

        axes[4, t].imshow(
            diagonal_down_left_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[4, t].set_title(
            f"Diag down-left\n t={t}\ncount={int(diagonal_down_left_spikes[t].sum().item())}"
        )

        axes[5, t].imshow(
            small_corner_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[5, t].set_title(
            f"Small corner\n t={t}\ncount={int(small_corner_spikes[t].sum().item())}"
        )

        axes[6, t].imshow(
            junction_spikes[t, y_min:y_max, x_min:x_max],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[6, t].set_title(
            f"Junction\n t={t}\ncount={int(junction_spikes[t].sum().item())}"
        )

        for row in range(7):
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])

    plt.suptitle(
        "Strict feature maps from stabilized ObjectMemory",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

def run_seven_evidence_neuron(
    features_result,
    horizontal_weight=1.0,
    diagonal_down_left_weight=1.5,
    junction_weight=2.0,
    small_corner_weight=0.5,
    vertical_inhibition_weight=1.0,
    diagonal_down_right_inhibition_weight=1.0,
    beta=0.8,
    threshold=6.0,
    reset_mode="subtract",
):
    """
    Single LIF SevenEvidence neuron.

    It receives excitatory evidence from features expected in our toy seven:
        horizontal
        diagonal_down_left
        horizontal_to_diagonal_junction
        small_corner

    It receives inhibitory evidence from features not expected in this seven:
        vertical
        diagonal_down_right

    This is a hand-built spiking readout, not a trained classifier yet.
    """

    horizontal_spikes = features_result["horizontal"]["feature_spike_history"]
    vertical_spikes = features_result["vertical"]["feature_spike_history"]
    diagonal_down_left_spikes = features_result["diagonal_down_left"]["feature_spike_history"]
    diagonal_down_right_spikes = features_result["diagonal_down_right"]["feature_spike_history"]
    small_corner_spikes = features_result["small_corner_spike_history"]
    junction_spikes = features_result["horizontal_to_diagonal_junction_spike_history"]

    num_steps = horizontal_spikes.shape[0]

    membrane = torch.zeros((), dtype=torch.float32)

    seven_input_history = []
    seven_spike_history = []
    seven_membrane_history = []

    table_rows = []

    for t in range(num_steps):
        horizontal_count = horizontal_spikes[t].sum()
        diagonal_down_left_count = diagonal_down_left_spikes[t].sum()
        junction_count = junction_spikes[t].sum()
        small_corner_count = small_corner_spikes[t].sum()

        vertical_count = vertical_spikes[t].sum()
        diagonal_down_right_count = diagonal_down_right_spikes[t].sum()

        excitatory_input = (
            horizontal_weight * horizontal_count
            + diagonal_down_left_weight * diagonal_down_left_count
            + junction_weight * junction_count
            + small_corner_weight * small_corner_count
        )

        inhibitory_input = (
            vertical_inhibition_weight * vertical_count
            + diagonal_down_right_inhibition_weight * diagonal_down_right_count
        )

        current_input = excitatory_input - inhibitory_input

        seven_spike, membrane = lif_step(
            input_current=current_input,
            membrane=membrane,
            beta=beta,
            threshold=threshold,
            reset_mode=reset_mode,
        )

        seven_input_history.append(current_input.clone())
        seven_spike_history.append(seven_spike.clone())
        seven_membrane_history.append(membrane.clone())

        table_rows.append({
            "t": t,
            "horizontal_count": int(horizontal_count.item()),
            "diagonal_down_left_count": int(diagonal_down_left_count.item()),
            "junction_count": int(junction_count.item()),
            "small_corner_count": int(small_corner_count.item()),
            "vertical_count": int(vertical_count.item()),
            "diagonal_down_right_count": int(diagonal_down_right_count.item()),
            "seven_input": float(current_input.item()),
            "seven_spike": int(seven_spike.item()),
            "seven_membrane": float(membrane.item()),
        })

    return {
        "seven_input_history": torch.stack(seven_input_history),
        "seven_spike_history": torch.stack(seven_spike_history),
        "seven_membrane_history": torch.stack(seven_membrane_history),
        "table": pd.DataFrame(table_rows),
    }

def plot_seven_evidence_result(seven_evidence_result):
    """
    Plot SevenEvidence input, membrane, and spikes over time.
    """

    table = seven_evidence_result["table"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(table["t"], table["seven_input"], marker="o")
    axes[0].set_ylabel("Input current")
    axes[0].set_title("SevenEvidence input over time")

    axes[1].plot(table["t"], table["seven_membrane"], marker="o")
    axes[1].set_ylabel("Membrane")
    axes[1].set_title("SevenEvidence membrane over time")

    axes[2].stem(table["t"], table["seven_spike"])
    axes[2].set_ylabel("Spike")
    axes[2].set_xlabel("Time")
    axes[2].set_title("SevenEvidence spikes")

    plt.tight_layout()
    plt.show()

def run_full_seven_evidence_pipeline_for_shape(
    shape_name,
    velocity_sequence,
    start_top=50,
    start_left=30,
    max_velocity=6,
    max_displacement=36,
    velocity_params=None,
    seven_threshold=12.0,
):
    """
    Run the full pipeline on one shape:

        moving input
        -> displacement-gated object memory
        -> strict feature maps with junctions
        -> SevenEvidence neuron

    This lets us compare seven vs non-seven shapes.
    """

    if velocity_params is None:
        velocity_params = GOOD_VELOCITY_PARAMS

    case = make_variable_velocity_spike_case(
        velocity_sequence=velocity_sequence,
        shape_name=shape_name,
        start_top=start_top,
        start_left=start_left,
    )

    object_memory_result = run_displacement_gated_object_memory(
        case=case,
        max_velocity=max_velocity,
        max_displacement=max_displacement,
        velocity_params=velocity_params,
        threshold_memory=1.5,
        beta_memory=0.8,
        reset_mode="subtract",
    )

    features_result = compute_strict_basic_shape_features_with_junctions(
        object_memory_spike_history=object_memory_result["memory_spike_history"],
    )

    seven_evidence_result = run_seven_evidence_neuron(
        features_result=features_result,
        beta=0.8,
        threshold=seven_threshold,
        reset_mode="subtract",
    )

    feature_summary = pd.DataFrame({
        "feature": [
            "horizontal",
            "vertical",
            "diagonal_down_right",
            "diagonal_down_left",
            "small_corner",
            "junction",
        ],
        "total_spikes_over_time": [
            int(features_result["horizontal"]["feature_spike_history"].sum().item()),
            int(features_result["vertical"]["feature_spike_history"].sum().item()),
            int(features_result["diagonal_down_right"]["feature_spike_history"].sum().item()),
            int(features_result["diagonal_down_left"]["feature_spike_history"].sum().item()),
            int(features_result["small_corner_spike_history"].sum().item()),
            int(features_result["horizontal_to_diagonal_junction_spike_history"].sum().item()),
        ],
        "active_frames": [
            int((features_result["horizontal"]["feature_spike_history"].sum(dim=(1, 2)) > 0).sum().item()),
            int((features_result["vertical"]["feature_spike_history"].sum(dim=(1, 2)) > 0).sum().item()),
            int((features_result["diagonal_down_right"]["feature_spike_history"].sum(dim=(1, 2)) > 0).sum().item()),
            int((features_result["diagonal_down_left"]["feature_spike_history"].sum(dim=(1, 2)) > 0).sum().item()),
            int((features_result["small_corner_spike_history"].sum(dim=(1, 2)) > 0).sum().item()),
            int((features_result["horizontal_to_diagonal_junction_spike_history"].sum(dim=(1, 2)) > 0).sum().item()),
        ],
    })

    return {
        "shape_name": shape_name,
        "case": case,
        "object_memory_result": object_memory_result,
        "features_result": features_result,
        "seven_evidence_result": seven_evidence_result,
        "feature_summary": feature_summary,
    }

def run_robustness_test_for_seven_evidence(
    shape_names,
    velocity_sequences,
    seven_threshold=12.0,
):
    """
    Test SevenEvidence across multiple shapes and motion sequences.
    """

    rows = []
    results = {}

    for sequence_name, velocity_sequence in velocity_sequences.items():
        results[sequence_name] = {}

        for shape_name in shape_names:
            print(f"Running shape={shape_name}, sequence={sequence_name}")

            result = run_full_seven_evidence_pipeline_for_shape(
                shape_name=shape_name,
                velocity_sequence=velocity_sequence,
                seven_threshold=seven_threshold,
            )

            results[sequence_name][shape_name] = result

            total_seven_spikes = int(
                result["seven_evidence_result"]["seven_spike_history"].sum().item()
            )

            total_seven_input = float(
                result["seven_evidence_result"]["seven_input_history"].sum().item()
            )

            max_seven_membrane = float(
                result["seven_evidence_result"]["seven_membrane_history"].max().item()
            )

            rows.append({
                "sequence": sequence_name,
                "shape": shape_name,
                "total_seven_spikes": total_seven_spikes,
                "total_seven_input": total_seven_input,
                "max_seven_membrane": max_seven_membrane,
                "classified_as_seven": total_seven_spikes > 0,
            })

    return {
        "table": pd.DataFrame(rows),
        "results": results,
    }

def run_full_seven_evidence_pipeline_for_shape_with_motion_limits(
    shape_name,
    velocity_sequence,
    start_top=50,
    start_left=30,
    max_velocity=12,
    max_displacement=72,
    seven_threshold=12.0,
):
    """
    Same full pipeline, but exposes max_velocity and max_displacement
    so we can test faster motion.
    """

    velocity_params = {
        "threshold_velocity": 15.0,
        "input_scale": 1.0,
        "beta_velocity": 0.8,
        "inhibition_strength": 3.0,
        "inhibition_radius": 1,
        "internal_steps": 5,
    }

    case = make_variable_velocity_spike_case(
        velocity_sequence=velocity_sequence,
        shape_name=shape_name,
        start_top=start_top,
        start_left=start_left,
    )

    object_memory_result = run_displacement_gated_object_memory(
        case=case,
        max_velocity=max_velocity,
        max_displacement=max_displacement,
        velocity_params=velocity_params,
        threshold_memory=1.5,
        beta_memory=0.8,
        reset_mode="subtract",
    )

    features_result = compute_strict_basic_shape_features_with_junctions(
        object_memory_spike_history=object_memory_result["memory_spike_history"],
    )

    seven_evidence_result = run_seven_evidence_neuron(
        features_result=features_result,
        beta=0.8,
        threshold=seven_threshold,
        reset_mode="subtract",
    )

    return {
        "shape_name": shape_name,
        "case": case,
        "object_memory_result": object_memory_result,
        "features_result": features_result,
        "seven_evidence_result": seven_evidence_result,
    }

def run_high_speed_test_with_motion_limits(
    shape_names,
    velocity_sequences,
    max_velocity=12,
    max_displacement=72,
    seven_threshold=12.0,
):
    rows = []
    results = {}

    for sequence_name, velocity_sequence in velocity_sequences.items():
        results[sequence_name] = {}

        for shape_name in shape_names:
            print(f"Running shape={shape_name}, sequence={sequence_name}")

            result = run_full_seven_evidence_pipeline_for_shape_with_motion_limits(
                shape_name=shape_name,
                velocity_sequence=velocity_sequence,
                max_velocity=max_velocity,
                max_displacement=max_displacement,
                seven_threshold=seven_threshold,
            )

            results[sequence_name][shape_name] = result

            total_seven_spikes = int(
                result["seven_evidence_result"]["seven_spike_history"].sum().item()
            )

            total_seven_input = float(
                result["seven_evidence_result"]["seven_input_history"].sum().item()
            )

            max_seven_membrane = float(
                result["seven_evidence_result"]["seven_membrane_history"].max().item()
            )

            rows.append({
                "sequence": sequence_name,
                "shape": shape_name,
                "max_velocity": max_velocity,
                "max_displacement": max_displacement,
                "total_seven_spikes": total_seven_spikes,
                "total_seven_input": total_seven_input,
                "max_seven_membrane": max_seven_membrane,
                "classified_as_seven": total_seven_spikes > 0,
            })

    return {
        "table": pd.DataFrame(rows),
        "results": results,
    }

def randomly_drop_spikes(spike_history, drop_probability, seed=0):
    """
    Randomly remove some spikes from a spike history.

    spike_history:
        [T, H, W]

    drop_probability:
        probability that an existing spike is removed
    """

    torch.manual_seed(seed)

    spike_history = (spike_history > 0).float()

    keep_mask = torch.rand_like(spike_history) > drop_probability

    dropped_spike_history = spike_history * keep_mask.float()

    return dropped_spike_history

def add_missing_spikes_to_case(case, drop_probability, seed=0):
    """
    Return a copy of a case where some spikes are randomly removed.
    """

    noisy_case = dict(case)
    noisy_case["spikes"] = randomly_drop_spikes(
        spike_history=case["spikes"],
        drop_probability=drop_probability,
        seed=seed,
    )

    noisy_case["frames"] = noisy_case["spikes"].clone()

    return noisy_case

def run_full_seven_evidence_pipeline_for_case(
    case,
    max_velocity=12,
    max_displacement=72,
    seven_threshold=12.0,
):
    """
    Run object memory, features, and SevenEvidence from an existing case.
    Useful for noisy/occluded cases.
    """

    velocity_params = {
        "threshold_velocity": 15.0,
        "input_scale": 1.0,
        "beta_velocity": 0.8,
        "inhibition_strength": 3.0,
        "inhibition_radius": 1,
        "internal_steps": 5,
    }

    object_memory_result = run_displacement_gated_object_memory(
        case=case,
        max_velocity=max_velocity,
        max_displacement=max_displacement,
        velocity_params=velocity_params,
        threshold_memory=1.5,
        beta_memory=0.8,
        reset_mode="subtract",
    )

    features_result = compute_strict_basic_shape_features_with_junctions(
        object_memory_spike_history=object_memory_result["memory_spike_history"],
    )

    seven_evidence_result = run_seven_evidence_neuron(
        features_result=features_result,
        beta=0.8,
        threshold=seven_threshold,
        reset_mode="subtract",
    )

    return {
        "case": case,
        "object_memory_result": object_memory_result,
        "features_result": features_result,
        "seven_evidence_result": seven_evidence_result,
    }

