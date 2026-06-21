### Setup Guide

1. Follow the official executorch and qualcomm backend build guide
2. Run the following commands to test HexFlashAttention inference:

```bash
cd $EXECUTORCH_ROOT
python backends/qualcomm/op_packages/HexFlashAttentionV1/src/tests/test_ops.py \
-b build-android -m SM8750 --arch 79  --op_id 1001 -s {DEVICE_ID} -H  "" --debug --kv_n 64
```

`kv_n` flag is optional to specify the KV length: `64 * kv_n`