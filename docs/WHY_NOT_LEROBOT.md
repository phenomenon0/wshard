# Why Not LeRobot?

LeRobot is Hugging Face's robotics dataset library. It is the best current tool for distributing robotics datasets on the Hugging Face Hub. WShard and LeRobot solve different problems at different layers of the stack, and in most pipelines they complement each other rather than compete.

---

## LeRobot strengths

- **Hugging Face Hub integration.** `push_to_hub()` / `from_pretrained()` workflows, dataset versioning, metadata cards — the full HF ecosystem applies directly.
- **MP4 video natively.** Camera streams are stored as H.264 video, which dramatically reduces storage for large-scale camera datasets. WShard stores camera data as raw tensors and has no video codec today.
- **Large community of pre-uploaded datasets.** Dozens of real-robot and simulated datasets are already on the Hub in LeRobot format. You can start training on real data without collecting your own.
- **Integration with HF Datasets and Transformers.** Standard `datasets.load_dataset()` interface; integrates with the broader HF training ecosystem.
- **Dataset versioning and reproducibility.** The HF Hub tracks dataset versions. Experiments can be pinned to a specific dataset commit.

---

## Different focus, different unit

**LeRobot's unit is a dataset.** One repository on the Hub contains many episodes (potentially thousands), organized as Parquet row tables plus MP4 video files. The dataset is the thing you version, distribute, and load.

**WShard's unit is an episode.** One `.wshard` file contains one episode. Move it, archive it, mmap it, ship it across a network — the file is self-contained. WShard says nothing about hub hosting or multi-episode datasets.

These are different abstractions. LeRobot answers "how do I share 5,000 episodes with the community?" WShard answers "how do I store one episode as a self-contained unit I can read from Go, Python, or TypeScript without a hub?"

---

## Where Parquet fits (and where it doesn't)

LeRobot v3 stores data as Parquet rows (one row per timestep, channels as columns). Parquet is excellent for columnar analytics, but it has no native multi-dimensional array type. In LeRobot:

- Scalar channels (`reward`, joint positions expanded as `joint_pos_0` … `joint_pos_6`) map naturally to Parquet columns.
- RGB frames are serialized as `large_binary` blobs, one blob per row — the tensor shape is lost in the column encoding.

WShard stores tensors verbatim. A `signal/rgb` block with shape `[T, 84, 84, 3]` is exactly that array on disk, with shape metadata in the index entry. `np.frombuffer` on the mmap'd block gives you the array with zero copies.

**Measured read performance** (`bench/bench_parquet.py`, same 20 MB workload, pyarrow 22.0.0):

| Config | Write median | Read median | File size |
|--------|-------------|-------------|-----------|
| wshard-none | 32.3 ms | **5.6 ms** | 20.25 MB |
| wshard-zstd | 64.0 ms | **4.9 ms** | 20.24 MB |
| parquet-zstd | 26.0 ms | 13.1 ms | 20.25 MB |
| parquet-none | 10.7 ms | 10.2 ms | 20.25 MB |

WShard reads **2.4× faster** than Parquet on the same payload (5.6 ms vs 13.1 ms, wshard-none vs parquet-zstd). The difference is Parquet's column materialization: pyarrow constructs Python lists per column before you get an array. WShard uses `np.frombuffer` directly into the mmap'd block.

Parquet write is faster (10.7 ms vs 32 ms for wshard-none) because pyarrow's column-write path is highly optimized C++, while WShard's Python writer is pure struct-packing today.

---

## Bridging the two

WShard and LeRobot can coexist in the same pipeline:

- **Data collection:** write `.wshard` files per episode (streaming, crash-safe, from Go or Python)
- **Training-time access:** read `.wshard` files directly from the DataLoader
- **Publication:** convert to LeRobot format for hub upload

An experimental converter lives at `examples/wshard_to_lerobot.py`. It is not yet covered by integration tests — treat it as a starting point and report breakage.

The inverse direction (LeRobot → WShard) is on the roadmap, primarily for teams who want to take Hub-distributed datasets and reformat them into per-episode files for local training pipelines with partial-block access.

---

## When to pick LeRobot instead

1. **You want to publish a dataset on the Hugging Face Hub.** LeRobot's tooling (`push_to_hub`, dataset cards, version tracking) is built exactly for this. WShard has no hub integration.
2. **Your dataset is camera-heavy and MP4 compression matters.** LeRobot stores video as H.264. WShard stores raw tensors. For 1080p 30Hz data, this is a 50–200× storage difference.
3. **You want to use pre-existing community datasets.** The Hub already has dozens of LeRobot datasets. Consuming them as LeRobot is zero-friction; converting them to WShard first adds work.
4. **Your team is already in the HF Datasets ecosystem.** `load_dataset()`, transformers-compatible batch iteration, and HF evaluation tooling all work out of the box with LeRobot.
