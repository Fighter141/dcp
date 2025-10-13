# DCP

DCP is a context parallel training library designed for dynamic model input lengths and attention masks. It introduces fine-grained blockwise partitioning of both data and computation, enables flexible mapping of data and computation blocks to any device, and optimizes such mapping through a hypergraph partitioning framework.

## Installation

### Using Docker (Recommended)

The easiest way to set up DCP with all dependencies is using Docker. 

```bash
cd docker

# For standard environments
docker build --build-arg AWS=false -t dcp:latest .

# For AWS environments with EFA network support
docker build --build-arg AWS=true -t dcp:latest .

# Run the container
docker run --gpus all -it dcp:latest
```

> **Note:** See [`docker/Dockerfile`](docker/Dockerfile) for the complete installation process used to generate the container for our experiments.

### Manual Installation

DCP requires the following dependencies:

- **Custom PyTorch**: A custom branch that supports `all_to_all_single` with zero-byte send/recv operations
  ```bash
  git clone --recursive -b alltoallv https://github.com/chenyu-jiang/pytorch.git
  cd pytorch && pip install -e .
  ```

- **Custom FlashAttention**: Forked from version 2.6.3, supports specifying attention masks with ranges (limited to at most two ranges per sequence)
  ```bash
  git clone --recursive -b dcp https://github.com/chenyu-jiang/flash-attention.git
  cd flash-attention && pip install -e . --no-build-isolation
  ```
  > **Note:** This installs the custom FlashAttention as `dcp_flash_attn` to avoid overriding the original FlashAttention package.

- **DCP Library**:
  ```bash
  git clone https://github.com/chenyu-jiang/dcp.git
  cd dcp && pip install -e . --no-build-isolation
  ```

- **Hypergraph Partitioners**: `mtkahypar`, `kahypar`, `PaToH`, and `pypatoh`

For detailed installation steps, please refer to the [`docker/Dockerfile`](docker/Dockerfile).

## Quick Start

Below is a pseudo-code example demonstrating how to integrate DCP into a training pipeline. For a complete implementation example, see [`benchmark/mlm/monkey_patch.py`](benchmark/mlm/monkey_patch.py) and [`benchmark/mlm/pretrain_gpt.py`](benchmark/mlm/pretrain_gpt.py), which show how DCP can be integrated with Megatron-LM.

```python
# When defining models
from dcp.runtime.flash_attention.executor import DCPAttention, AttentionExecutor

class TransformerLayer(...):
    def forward(..., dcp_executor):
        ...
        # Replace attention implementation with DCPAttention
        core_attn_out = DCPAttention.apply(dcp_executor, q, kv)
        ...

# Define a mask function
def mask_fn(seqlens, ...):
    ...
    return mask

# In training script
from dcp.data.dataloader import DCPDataLoader

dcp_dataloader = DCPDataLoader(dataset, mask_fn)
# dcp_group is a communicator that connects all devices
# (e.g., torch.distributed.ProcessGroup)
dcp_executor = AttentionExecutor(group=dcp_group)

# Training iterations
for (local_data, execution_plan) in dcp_dataloader:
    # Set execution plan and create buffers
    dcp_executor.prepare(execution_plan)
    # Execute model
    loss = model(local_data, dcp_executor)
    ...
```

## Citation

If you find DCP helpful in your work, we would appreciate a citation to our paper:

```bibtex
@inproceedings{jiang2025dcp,
  author = {Jiang, Chenyu and Cai, Zhenkun and Tian, Ye and Jia, Zhen and Wang, Yida and Wu, Chuan},
  title = {DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism},
  booktitle = {Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  series = {SOSP '25},
  year = {2025},
  pages = {221–236}
}
```

## Artifact Evaluation

Please refer to [this document](docs/artifact_evaluation.md) for SOSP'25 Artifact Evaluation.
