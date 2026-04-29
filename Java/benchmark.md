Will redo this later, but this is on 100,000 random integers:
```
# JMH version: 1.37
# VM version: JDK 21.0.8, OpenJDK 64-Bit Server VM, 21.0.8+9-LTS
# VM invoker: C:\Users\miau\.p2\pool\plugins\org.eclipse.justj.openjdk.hotspot.jre.full.win32.x86_64_21.0.8.v20250724-1412\jre\bin\java.exe
# VM options: -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8 -XX:+ShowCodeDetailsInExceptionMessages
# Blackhole mode: compiler (auto-detected, use -Djmh.blackhole.autoDetect=false to disable)
# Warmup: 5 iterations, 10 s each
# Measurement: 5 iterations, 10 s each
# Timeout: 10 min per iteration
# Threads: 1 thread, will synchronize iterations
# Benchmark mode: Average time, time/op
# Benchmark: bench.SortBench.javaSort

# Run progress: 0.00% complete, ETA 00:16:40
# Fork: 1 of 5
# Warmup Iteration   1: 14.179 ms/op
# Warmup Iteration   2: 13.941 ms/op
# Warmup Iteration   3: 13.965 ms/op
# Warmup Iteration   4: 13.913 ms/op
# Warmup Iteration   5: 13.963 ms/op
Iteration   1: 13.923 ms/op
Iteration   2: 14.000 ms/op
Iteration   3: 13.987 ms/op
Iteration   4: 13.959 ms/op
Iteration   5: 13.932 ms/op

# Run progress: 10.00% complete, ETA 00:15:03
# Fork: 2 of 5
# Warmup Iteration   1: 14.334 ms/op
# Warmup Iteration   2: 14.071 ms/op
# Warmup Iteration   3: 14.062 ms/op
# Warmup Iteration   4: 14.037 ms/op
# Warmup Iteration   5: 14.005 ms/op
Iteration   1: 14.045 ms/op
Iteration   2: 14.032 ms/op
Iteration   3: 14.029 ms/op
Iteration   4: 14.085 ms/op
Iteration   5: 14.068 ms/op

# Run progress: 20.00% complete, ETA 00:13:23
# Fork: 3 of 5
# Warmup Iteration   1: 14.393 ms/op
# Warmup Iteration   2: 14.056 ms/op
# Warmup Iteration   3: 14.114 ms/op
# Warmup Iteration   4: 14.070 ms/op
# Warmup Iteration   5: 14.157 ms/op
Iteration   1: 14.110 ms/op
Iteration   2: 14.084 ms/op
Iteration   3: 14.079 ms/op
Iteration   4: 14.164 ms/op
Iteration   5: 14.122 ms/op

# Run progress: 30.00% complete, ETA 00:11:42
# Fork: 4 of 5
# Warmup Iteration   1: 14.867 ms/op
# Warmup Iteration   2: 14.337 ms/op
# Warmup Iteration   3: 14.094 ms/op
# Warmup Iteration   4: 14.171 ms/op
# Warmup Iteration   5: 14.187 ms/op
Iteration   1: 14.190 ms/op
Iteration   2: 14.096 ms/op
Iteration   3: 14.229 ms/op
Iteration   4: 14.230 ms/op
Iteration   5: 14.224 ms/op

# Run progress: 40.00% complete, ETA 00:10:02
# Fork: 5 of 5
# Warmup Iteration   1: 14.562 ms/op
# Warmup Iteration   2: 14.176 ms/op
# Warmup Iteration   3: 14.228 ms/op
# Warmup Iteration   4: 14.262 ms/op
# Warmup Iteration   5: 14.224 ms/op
Iteration   1: 14.123 ms/op
Iteration   2: 14.160 ms/op
Iteration   3: 14.198 ms/op
Iteration   4: 14.275 ms/op
Iteration   5: 14.345 ms/op


Result "bench.SortBench.javaSort":
  14.108 ±(99.9%) 0.081 ms/op [Average]
  (min, avg, max) = (13.923, 14.108, 14.345), stdev = 0.108
  CI (99.9%): [14.026, 14.189] (assumes normal distribution)


# JMH version: 1.37
# VM version: JDK 21.0.8, OpenJDK 64-Bit Server VM, 21.0.8+9-LTS
# VM invoker: C:\Users\miau\.p2\pool\plugins\org.eclipse.justj.openjdk.hotspot.jre.full.win32.x86_64_21.0.8.v20250724-1412\jre\bin\java.exe
# VM options: -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8 -XX:+ShowCodeDetailsInExceptionMessages
# Blackhole mode: compiler (auto-detected, use -Djmh.blackhole.autoDetect=false to disable)
# Warmup: 5 iterations, 10 s each
# Measurement: 5 iterations, 10 s each
# Timeout: 10 min per iteration
# Threads: 1 thread, will synchronize iterations
# Benchmark mode: Average time, time/op
# Benchmark: bench.SortBench.miauSort

# Run progress: 50.00% complete, ETA 00:08:22
# Fork: 1 of 5
# Warmup Iteration   1: 13.619 ms/op
# Warmup Iteration   2: 13.443 ms/op
# Warmup Iteration   3: 13.468 ms/op
# Warmup Iteration   4: 13.482 ms/op
# Warmup Iteration   5: 13.472 ms/op
Iteration   1: 14.042 ms/op
Iteration   2: 13.552 ms/op
Iteration   3: 13.491 ms/op
Iteration   4: 13.488 ms/op
Iteration   5: 13.533 ms/op

# Run progress: 60.00% complete, ETA 00:06:41
# Fork: 2 of 5
# Warmup Iteration   1: 13.720 ms/op
# Warmup Iteration   2: 13.595 ms/op
# Warmup Iteration   3: 13.542 ms/op
# Warmup Iteration   4: 13.617 ms/op
# Warmup Iteration   5: 13.601 ms/op
Iteration   1: 13.610 ms/op
Iteration   2: 13.562 ms/op
Iteration   3: 13.612 ms/op
Iteration   4: 13.613 ms/op
Iteration   5: 13.567 ms/op

# Run progress: 70.00% complete, ETA 00:05:01
# Fork: 3 of 5
# Warmup Iteration   1: 13.677 ms/op
# Warmup Iteration   2: 13.546 ms/op
# Warmup Iteration   3: 13.596 ms/op
# Warmup Iteration   4: 13.596 ms/op
# Warmup Iteration   5: 13.494 ms/op
Iteration   1: 13.487 ms/op
Iteration   2: 13.484 ms/op
Iteration   3: 13.482 ms/op
Iteration   4: 13.492 ms/op
Iteration   5: 13.457 ms/op

# Run progress: 80.00% complete, ETA 00:03:20
# Fork: 4 of 5
# Warmup Iteration   1: 13.639 ms/op
# Warmup Iteration   2: 13.497 ms/op
# Warmup Iteration   3: 13.529 ms/op
# Warmup Iteration   4: 13.485 ms/op
# Warmup Iteration   5: 13.505 ms/op
Iteration   1: 13.523 ms/op
Iteration   2: 13.513 ms/op
Iteration   3: 13.508 ms/op
Iteration   4: 13.519 ms/op
Iteration   5: 13.524 ms/op

# Run progress: 90.00% complete, ETA 00:01:40
# Fork: 5 of 5
# Warmup Iteration   1: 13.641 ms/op
# Warmup Iteration   2: 13.462 ms/op
# Warmup Iteration   3: 13.484 ms/op
# Warmup Iteration   4: 13.482 ms/op
# Warmup Iteration   5: 13.528 ms/op
Iteration   1: 13.554 ms/op
Iteration   2: 13.556 ms/op
Iteration   3: 13.516 ms/op
Iteration   4: 13.482 ms/op
Iteration   5: 13.515 ms/op


Result "bench.SortBench.miauSort":
  13.547 ±(99.9%) 0.084 ms/op [Average]
  (min, avg, max) = (13.457, 13.547, 14.042), stdev = 0.112
  CI (99.9%): [13.464, 13.631] (assumes normal distribution)


# Run complete. Total time: 00:16:44

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

Benchmark           Mode  Cnt   Score   Error  Units
SortBench.javaSort  avgt   25  14.108 ± 0.081  ms/op
SortBench.miauSort  avgt   25  13.547 ± 0.084  ms/op
```
