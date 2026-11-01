Benchmark in comparison to `Arrays.sort()` on an object array.

---

## Key:
|Input name|Description|
|:---:|:---:|
| `RANDOM` | Random shuffle of objects with values in interval [0:n) |
| `SAWTOOTH` | Four sorted runs of equal size |
| `SORTED` | Sorted array containing with values in interval [0:n) |
| `REVERSED` | Reverse-sorted array |
| `REVERSED_SEGMENTS` | Reverse-sorted array with n/4 distinct values |

```
# Run complete. Total time: 00:13:14

REMEMBER: The numbers below are just data. To gain reusable insights, you need to follow up on
why the numbers are the way they are. Use profilers (see -prof, -lprof), design factorial
experiments, perform baseline and negative tests that provide experimental control, make sure
the benchmarking environment is safe on JVM/OS/HW level, ask for reviews from the domain experts.
Do not assume the numbers tell you what you want them to tell.

NOTE: Current JVM experimentally supports Compiler Blackholes, and they are in use. Please exercise
extra caution when trusting the results, look into the generated code to check the benchmark still
works, and factor in a small probability of new VM bugs. Additionally, while comparisons between
different JVMs are already problematic, the performance difference caused by different Blackhole
modes can be very significant. Please make sure you use the consistent Blackhole mode for comparisons.

Benchmark                  (distribution)  (size)  Mode  Cnt   Score    Error  Units
SortBenchmark.javaSort             RANDOM    1000  avgt   16   0.041 ±  0.001  ms/op
SortBenchmark.javaSort             RANDOM   10000  avgt   16   1.010 ±  0.015  ms/op
SortBenchmark.javaSort             RANDOM  100000  avgt   16  13.902 ±  0.263  ms/op
SortBenchmark.javaSort           SAWTOOTH    1000  avgt   16   0.008 ±  0.001  ms/op
SortBenchmark.javaSort           SAWTOOTH   10000  avgt   16   0.077 ±  0.001  ms/op
SortBenchmark.javaSort           SAWTOOTH  100000  avgt   16   0.864 ±  0.013  ms/op
SortBenchmark.javaSort             SORTED    1000  avgt   16   0.001 ±  0.001  ms/op
SortBenchmark.javaSort             SORTED   10000  avgt   16   0.012 ±  0.001  ms/op
SortBenchmark.javaSort             SORTED  100000  avgt   16   0.144 ±  0.003  ms/op
SortBenchmark.javaSort           REVERSED    1000  avgt   16   0.003 ±  0.001  ms/op
SortBenchmark.javaSort           REVERSED   10000  avgt   16   0.026 ±  0.001  ms/op
SortBenchmark.javaSort           REVERSED  100000  avgt   16   0.290 ±  0.003  ms/op
SortBenchmark.javaSort  REVERSED_SEGMENTS    1000  avgt   16   0.029 ±  0.001  ms/op
SortBenchmark.javaSort  REVERSED_SEGMENTS   10000  avgt   16   0.307 ±  0.003  ms/op
SortBenchmark.javaSort  REVERSED_SEGMENTS  100000  avgt   16   3.039 ±  0.035  ms/op
SortBenchmark.miauSort             RANDOM    1000  avgt   16   0.043 ±  0.001  ms/op
SortBenchmark.miauSort             RANDOM   10000  avgt   16   1.067 ±  0.015  ms/op
SortBenchmark.miauSort             RANDOM  100000  avgt   16  14.307 ±  0.200  ms/op
SortBenchmark.miauSort           SAWTOOTH    1000  avgt   16   0.012 ±  0.001  ms/op
SortBenchmark.miauSort           SAWTOOTH   10000  avgt   16   0.083 ±  0.001  ms/op
SortBenchmark.miauSort           SAWTOOTH  100000  avgt   16   1.073 ±  0.012  ms/op
SortBenchmark.miauSort             SORTED    1000  avgt   16   0.001 ±  0.001  ms/op
SortBenchmark.miauSort             SORTED   10000  avgt   16   0.008 ±  0.001  ms/op
SortBenchmark.miauSort             SORTED  100000  avgt   16   0.109 ±  0.002  ms/op
SortBenchmark.miauSort           REVERSED    1000  avgt   16   0.003 ±  0.001  ms/op
SortBenchmark.miauSort           REVERSED   10000  avgt   16   0.032 ±  0.001  ms/op
SortBenchmark.miauSort           REVERSED  100000  avgt   16   0.325 ±  0.002  ms/op
SortBenchmark.miauSort  REVERSED_SEGMENTS    1000  avgt   16   0.005 ±  0.001  ms/op
SortBenchmark.miauSort  REVERSED_SEGMENTS   10000  avgt   16   0.054 ±  0.001  ms/op
SortBenchmark.miauSort  REVERSED_SEGMENTS  100000  avgt   16   0.578 ±  0.005  ms/op
```
