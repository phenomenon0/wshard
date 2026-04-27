# wshard (Python)

Python reader/writer for the WShard binary episode format.

One `.wshard` file is one episode: observations, actions, rewards, done flags,
metadata, and optional model prediction / uncertainty / residual lanes. Each
block is independently addressable and independently compressed.

```python
from wshard import save_wshard, load_wshard
from wshard.types import Episode, Channel, DType
import numpy as np

ep = Episode(id="ep_001", length=100)
ep.env_id = "CartPole-v1"
ep.observations["state"] = Channel(
    name="state", dtype=DType.FLOAT32, shape=[4],
    data=np.random.randn(100, 4).astype(np.float32),
)
ep.actions["ctrl"] = Channel(
    name="ctrl", dtype=DType.INT32, shape=[1],
    data=np.zeros((100, 1), dtype=np.int32),
)
ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, data=np.ones(100, dtype=np.float32))

save_wshard(ep, "episode.wshard")
loaded = load_wshard("episode.wshard")
```

See the [project README](https://github.com/phenomenon0/wshard) for the
format specification, the cross-language story, and benchmarks.

## Install

```bash
pip install "git+https://github.com/phenomenon0/wshard.git#subdirectory=py"
```

## Optional extras

```bash
pip install "wshard[bf16]"    # ml_dtypes for bfloat16 channels
pip install "wshard[hdf5]"    # h5py for HDF5 import bridge
pip install "wshard[torch]"   # PyTorch tensor adapters
pip install "wshard[dev]"     # pytest, build tools
```

## Status

Beta. Looking for external users to test it against real robot/RL/world-model
pipelines. Bug reports welcome at https://github.com/phenomenon0/wshard/issues.

## License

MIT.
