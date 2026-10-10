# Needle-in-a-Haystack (gkamradt method) — KVMem retrieval (launcher config, traced)

- 上下文(prompt tokens, 实测): **252728**
- 针数: **12**  命中: **12**  召回率: **100.0%**
- 首次 prefill 墙钟: **583.9 s** (≈433 tok/s)
- 提问时 reasoning_effort: `none`
- 服务端配置: -c 262144, --kvmem-budget 36864, --kvmem-gen-reserve 16384, --kvmem-action-lrfu l=0.50 p=0.90, --kvmem-score-agg max, q8_0 KV, MTP draft; attention window verified 36791 tokens (14.6%)

## 逐深度结果

| 深度 | 针(期望值) | 结果 | 该轮 prompt | 生成 | 墙钟 | 热力 |
|---:|---|---|---:|---:|---:|---|
| 0.00% | `BLUEBIRD-100` | ✅ HIT | 252760 | 7 | 3.2s | `####################` |
| 9.09% | `ORCHID-107` | ✅ HIT | 252798 | 7 | 3.1s | `####################` |
| 18.18% | `CEDAR-114` | ✅ HIT | 252835 | 6 | 3.0s | `####################` |
| 27.27% | `MAPLE-121` | ✅ HIT | 252871 | 6 | 3.1s | `####################` |
| 36.36% | `PINE-128` | ✅ HIT | 252907 | 6 | 3.2s | `####################` |
| 45.45% | `FALCON-135` | ✅ HIT | 252944 | 7 | 3.2s | `####################` |
| 54.55% | `AMBER-142` | ✅ HIT | 252981 | 6 | 3.1s | `####################` |
| 63.64% | `QUARTZ-149` | ✅ HIT | 253017 | 7 | 3.0s | `####################` |
| 72.73% | `LANTERN-156` | ✅ HIT | 253054 | 7 | 3.3s | `####################` |
| 81.82% | `COPPER-163` | ✅ HIT | 253091 | 7 | 3.2s | `####################` |
| 90.91% | `WILLOW-170` | ✅ HIT | 253129 | 7 | 3.3s | `####################` |
| 100.00% | `SABLE-177` | ✅ HIT | 253166 | 6 | 3.4s | `####################` |

## 深度 → 召回

```
  0.00%  HIT   BLUEBIRD-100
  9.09%  HIT   ORCHID-107
 18.18%  HIT   CEDAR-114
 27.27%  HIT   MAPLE-121
 36.36%  HIT   PINE-128
 45.45%  HIT   FALCON-135
 54.55%  HIT   AMBER-142
 63.64%  HIT   QUARTZ-149
 72.73%  HIT   LANTERN-156
 81.82%  HIT   COPPER-163
 90.91%  HIT   WILLOW-170
100.00%  HIT   SABLE-177
```

## 未命中

无——全部深度均命中。

