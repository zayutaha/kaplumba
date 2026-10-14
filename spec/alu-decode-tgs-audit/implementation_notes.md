# Implementation notes — investigation only

## D0 — 2026-10-14

- User confirmed scope before spec creation; later explicitly requested tg/s
  investigation and a small test proving whether implementation merits a trial.
- GitHub tip fetched: `7ddf20f901ae7ad4d1e7342ca71677e266bbe847`.
  Local initial HEAD `819a551` had a pre-existing exl3.rs edit; later another
  actor changed the shared tree (model.rs dirty). Neither was edited/staged here.
- Read AGENTS.md, MEMORY.md, spec/alu-boost, decode-19tok-repacked-250gb,
  decode-20tok-crux, decode-25tok-peak-bandwidth and mlock-clean-mmap evidence,
  plus actual pinned kernel, dispatch, generation, encoder and model source.
- Findings changed the planned first experiment: scalar preparation in still-ulong
  QKV split-K, rather than re-implementing the already-32-bit non-split contig4.
- Standalone tests are entirely outside the working repository. The archive
  snapshot has no user code modifications. Reproduction source is embedded below
  so cleanup/OS scratch removal does not erase evidence.
- Probe acquires the real GPU flock file nonblocking and releases it on exit;
  gpuq itself remains unchanged and takes no lock. No full model run. Memory
  reports at GPU runs were 0.1–0.9 GiB unused with existing swap. These runs
  are preliminary model-free evidence, not production throughput acceptance.
- Initial direct vector-helper substitution failed raw FP32 equality. A later
  scalar state-preparation substitution kept original scalar MCG and succeeded.
  Root cause of vector numerics remains undiagnosed; no waiver inferred.
- Longer warmup was justified by the initial first-shape clock ramp. A/A controls
  added to quantify noise. Preserve all samples, including outliers.
- No runtime performance counters, register allocation or disassembly collected.
  No full checkpoint, finalized FP16 or generated-token identity tests executed.
  Do not claim those gates passed. Implementation tasks T1–T8 remain unchecked.
- Swift compiler warned that fastMathEnabled is deprecated and shape need not
  be mutable; neither was a shader skip or compile failure. A source-slicing
  marker initially used `static inline` for the generic decoder, corrected to
  the pinned source's actual `static float` before execution.
- A diagnostic output-buffer flush was added after an initial assertion failure
  hid stdout. This changes logging only; no invalid elapsed-time result accepted.

## Exact reproduction commands

Extract the fenced source blocks below into files in a new scratch directory;
replace AUDIT with that absolute directory. Fetch/archive the same revision.
Never execute these inside the project code tree.

```sh
AUDIT=$(mktemp -d /tmp/kaplumba-alu-audit.XXXXXX)
git archive 7ddf20f901ae7ad4d1e7342ca71677e266bbe847 | tar -x -C "$AUDIT"
# Save source blocks below as "$AUDIT/extract_audit.rs" and "$AUDIT/qkv_probe.swift".
rustc -O "$AUDIT/extract_audit.rs" -o "$AUDIT/extract_audit"
"$AUDIT/extract_audit"
"$AUDIT/extract_audit" nowrap  # expected exit 1
"$AUDIT/extract_audit" lsb     # expected exit 1
"$AUDIT/extract_audit" omit8   # expected exit 1
swiftc -O "$AUDIT/qkv_probe.swift" -o "$AUDIT/qkv_probe"
scripts/gpuq status
PROBE_SCALAR=1 scripts/gpuq try -- "$AUDIT/qkv_probe" "$AUDIT"
# Vector-helper variant: expected equality failure, not benchmark success.
scripts/gpuq status
scripts/gpuq try -- "$AUDIT/qkv_probe" "$AUDIT"
# Exit 75 means GPU busy; do useful CPU work and retry later.
```

The failed vector run below used `v0[0] += 1` as sabotage. The final source also
adds 1 to acc0, so it still must fail, but its corrupted-output count may differ.
The short-warmup log predates the longer warmup and A/A control additions.


## Independent CPU source

```rust
fn current(w: &[u32;24], lane: usize, mutant: &str) -> [u32;8] {
    let bit = ((lane*8+1)*3+768-16)%768;
    let j=bit/32; let s=bit%32;
    let wa=w[j]; let wb=if mutant=="nowrap" && j==23 {0} else {w[(j+1)%24]};
    let hi=if mutant=="lsb" {(wa>>s)|(wb<<(32-s))} else {(wa<<s)|(wb>>(32-s))};
    let lo=wb<<s;
    let mut out=[0;8];
    for i in 0..6 {out[i]=(hi>>(16-3*i))&65535;}
    out[6]=((hi<<2)|(lo>>30))&65535;
    out[7]=((hi<<5)|(lo>>27))&65535;
    if mutant=="omit8" {out[7]=0;}
    out
}
fn reference(w: &[u32;24], lane: usize) -> [u32;8] {
    let start=((lane*8+1)*3+768-16)%768;
    std::array::from_fn(|i| {let mut v=0;for b in 0..16 {let pos=(start+i*3+b)%768;v=(v<<1)|((w[pos/32]>>(31-pos%32))&1);}v})
}
fn main() {
    let mutant=std::env::args().nth(1).unwrap_or_default();
    let shifts:std::collections::BTreeSet<_>=(0..32).map(|l|((l*8+1)*3+768-16)%32).collect();
    println!("actual lane shifts: {:?}; contig4 = 4 tiles, 8 states/lane/tile",shifts);
    let mut checks=0;
    let mut check=|w:&[u32;24], label:&str| {for lane in 0..32 {let expected=reference(w,lane);let got=current(w,lane,&mutant);if expected!=got {let i=(0..8).find(|&i|expected[i]!=got[i]).unwrap();eprintln!("FAIL {label} lane={lane} state={i} expected={:04x} got={:04x} mutant={mutant}",expected[i],got[i]);std::process::exit(1);}checks+=8;}};
    for bit in 0..768 {let mut w=[0;24];w[bit/32]=1<<(31-bit%32);check(&w,&format!("basis bit {bit}"));}
    for value in [0, u32::MAX, 0x55555555, 0xaaaaaaaa, 0x80000000,1] {check(&[value;24],"uniform");}
    let mut seed=0x8765432101234567u64;
    for _ in 0..4096 {let mut w=[0;24];for x in &mut w {seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;*x=seed as u32;}check(&w,"random");}
    println!("PASS {checks} state comparisons against independent ring-bit reader; basis sweep proves linear extraction for all ring contents at actual lane offsets (CPU only)");
}
```

## Standalone GPU probe source

```swift
import Foundation
import Metal
import Darwin
setbuf(stdout, nil)
let root = CommandLine.arguments[1]
let lockPath = ProcessInfo.processInfo.environment["KAPLUMBA_GPU_LOCK_FILE"] ?? NSHomeDirectory()+"/.kaplumba/gpu.lock"
let lockFD = open(lockPath, O_RDWR|O_CREAT, mode_t(0o600))
guard lockFD >= 0 && flock(lockFD, LOCK_EX|LOCK_NB) == 0 else { print("GPU busy"); exit(75) }
defer { flock(lockFD, LOCK_UN); close(lockFD) }
let src = try String(contentsOfFile: root+"/crates/kernels/src/linear_repacked.metal", encoding: .utf8)
let lin = try String(contentsOfFile: root+"/crates/kernels/src/linear.metal", encoding: .utf8)
func slice(_ s:String,_ a:String,_ b:String)->String { let start=s.range(of:a)!.lowerBound; let end=s.range(of:b,range:start..<s.endIndex)!.lowerBound; return String(s[start..<end]) }
let mcg = slice(lin,"static inline float decode_state_mcg(","static float decode_state(")
let helper = slice(src,"static inline void extract_v_mcg_32(","static inline void decode_and_accum_4tile_2col(")
let original = slice(src,"kernel void linear_gemv_ishizuki_k3_fused2_sk2(","kernel void linear_gemv_ishizuki_k3_fused3(")
let baseline = original.replacingOccurrences(of:"linear_gemv_ishizuki_k3_fused2_sk2",with:"probe_a")
var candidate = original.replacingOccurrences(of:"linear_gemv_ishizuki_k3_fused2_sk2",with:"probe_b")
let mainStart = candidate.range(of:"        float v0[8], v1[8], v2[8], v3[8];")!.lowerBound
let mainEnd = candidate.range(of:"        acc0 += p00",range:mainStart..<candidate.endIndex)!.lowerBound
let mainReplacement = "        float v0[8], v1[8], v2[8], v3[8];\n" + (0..<4).map { "        extract_v_mcg_32(w\($0)a, w\($0)b, sh0, 32u-sh0, v\($0));\n" }.joined()
candidate.replaceSubrange(mainStart..<mainEnd,with:mainReplacement)
let tailStart = candidate.range(of:"        float v[8];")!.lowerBound
let tailEnd = candidate.range(of:"        acc0 += p0",range:tailStart..<candidate.endIndex)!.lowerBound
candidate.replaceSubrange(tailStart..<tailEnd,with:"        float v[8];\n        extract_v_mcg_32(a0, a1, sh0, 32u-sh0, v);\n")
candidate = candidate.replacingOccurrences(of:"const ulong",with:"const uint")
let re = try NSRegularExpression(pattern:"        const uint A[0-3] = .*;\\n")
candidate = re.stringByReplacingMatches(in:candidate,range:NSRange(candidate.startIndex...,in:candidate),withTemplate:"")
let scalarHelper = """
static inline uint probe_state32(uint wa, uint wb, uint s, uint i) {
    const uint H = (wa << s) | (wb >> (32u-s));
    const uint L = wb << s;
    if (i < 6u) return (H >> (16u-3u*i)) & 0xffffu;
    if (i == 6u) return ((H << 2u) | (L >> 30u)) & 0xffffu;
    return ((H << 5u) | (L >> 27u)) & 0xffffu;
}
"""
if ProcessInfo.processInfo.environment["PROBE_SCALAR"] != nil {
    candidate = original.replacingOccurrences(of:"linear_gemv_ishizuki_k3_fused2_sk2",with:"probe_b")
    for t in 0..<4 {
        candidate = candidate.replacingOccurrences(of:"decode_state_mcg(uint(A\(t) >> (64u - rad)) & 0xffffu)",with:"decode_state_mcg(probe_state32(w\(t)a, w\(t)b, sh0, i))")
    }
    candidate = candidate.replacingOccurrences(of:"decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu)",with:"decode_state_mcg(probe_state32(a0, a1, sh0, i))")
    candidate = candidate.replacingOccurrences(of:"v[i] = decode_state_mcg(probe_state32(w0a, w0b, sh0, i));",with:"v[i] = decode_state_mcg(probe_state32(a0, a1, sh0, i));")
    candidate = candidate.replacingOccurrences(of:"const ulong",with:"const uint")
    candidate = re.stringByReplacingMatches(in:candidate,range:NSRange(candidate.startIndex...,in:candidate),withTemplate:"")
}
var sabotage = candidate.replacingOccurrences(of:"probe_b",with:"probe_bad").replacingOccurrences(of:"extract_v_mcg_32(w0a, w0b, sh0, 32u-sh0, v0);",with:"extract_v_mcg_32(w0a, w0b, sh0, 32u-sh0, v0); v0[0] += 1.0f;")
sabotage = sabotage.replacingOccurrences(of:"acc0 += p00",with:"acc0 += 1.0f; acc0 += p00")
let msl = "#include <metal_stdlib>\nusing namespace metal;\n"+mcg+helper+scalarHelper+baseline+candidate+sabotage
try msl.write(toFile:root+"/qkv_variants.metal",atomically:true,encoding:.utf8)
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else { fatalError("Metal unavailable") }
let options=MTLCompileOptions(); options.fastMathEnabled=true
let lib=try device.makeLibrary(source:msl,options:options)
let pipes=try ["probe_a","probe_b","probe_bad"].map { try device.makeComputePipelineState(function:lib.makeFunction(name:$0)!) }
func median(_ x:[Double])->Double { let s=x.sorted(); return (s[(s.count-1)/2]+s[s.count/2])/2 }
func experiment(_ K:Int,_ Na:Int,_ Nb:Int) throws {
    let weightBytes=K*(Na+Nb)*3/8
    var seed:UInt64=0x8765432101234567
    func random()->UInt32 {seed ^= seed<<13;seed ^= seed>>7;seed ^= seed<<17;return UInt32(truncatingIfNeeded:seed)}
    func weights(_ n:Int)->MTLBuffer {let b=device.makeBuffer(length:n,options:.storageModeShared)!;let p=b.contents().bindMemory(to:UInt32.self,capacity:n/4);for i in 0..<n/4 {p[i]=random()};return b}
    let wa=weights(K*Na*3/8), wb=weights(K*Nb*3/8)
    let x=device.makeBuffer(length:K*2,options:.storageModeShared)!
    let xp=x.contents().bindMemory(to:UInt16.self,capacity:K)
    for i in 0..<K {xp[i]=Float16(Float(Int(random()%2001)-1000)/1024).bitPattern}
    let outputs=(0..<3).map {_ in [device.makeBuffer(length:Na*8,options:.storageModeShared)!,device.makeBuffer(length:Nb*8,options:.storageModeShared)!]}
    var shape=[UInt32(K),UInt32(Na),UInt32(Nb)]
    func run(_ arm:Int,_ repeats:Int)->Double {
        let cb=queue.makeCommandBuffer()!;let enc=cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipes[arm]);enc.setBuffer(wa,offset:0,index:0);enc.setBuffer(wb,offset:0,index:1);enc.setBuffer(x,offset:0,index:2)
        enc.setBuffer(outputs[arm][0],offset:0,index:3);enc.setBuffer(outputs[arm][1],offset:0,index:4)
        shape.withUnsafeBytes {enc.setBytes($0.baseAddress!,length:$0.count,index:5)}
        for _ in 0..<repeats {enc.dispatchThreadgroups(MTLSize(width:(Na+Nb)/16*2,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))}
        enc.endEncoding();cb.commit();cb.waitUntilCompleted()
        guard cb.status == .completed else {fatalError("command buffer failed: \(String(describing:cb.error))")}
        return (cb.gpuEndTime-cb.gpuStartTime)*1000/Double(repeats)
    }
    for a in 0..<3 {_=run(a,1)}
    var bad=0, mismatch=0
    for col in 0..<2 {let n=(col==0 ? Na:Nb)*2;let a=outputs[0][col].contents().bindMemory(to:UInt32.self,capacity:n);let b=outputs[1][col].contents().bindMemory(to:UInt32.self,capacity:n);let c=outputs[2][col].contents().bindMemory(to:UInt32.self,capacity:n)
        for i in 0..<n {if a[i] != b[i] {mismatch += 1};if a[i] != c[i] {bad += 1}}
    }
    print("CHECK K=\(K) Na=\(Na) Nb=\(Nb) fp32_words=\((Na+Nb)*2) candidate_differences=\(mismatch) sabotaged_differences=\(bad)")
    guard mismatch==0 && bad>0 else {fatalError("bit equality or shader sabotage gate failed")}
    for _ in 0..<128 {_=run(0,8);_=run(1,8)}
    var controls=[Double]()
    for _ in 0..<16 {let a1=run(0,8),b1=run(0,8),b2=run(0,8),a2=run(0,8);controls.append(100*((a1+a2)/(b1+b2)-1))}
    print(String(format:"CONTROL_AA median_pct=%.3f min=%.3f max=%.3f",median(controls),controls.min()!,controls.max()!))
    var aTimes=[Double](),bTimes=[Double](),gains=[Double]()
    for block in 0..<16 {let a1=run(0,8),b1=run(1,8),b2=run(1,8),a2=run(0,8);let a=(a1+a2)/2,b=(b1+b2)/2;aTimes.append(a);bTimes.append(b);gains.append(100*(a/b-1));print(String(format:"PAIR %02d A_ms=%.6f B_ms=%.6f gain_pct=%.3f",block,a,b,100*(a/b-1)))}
    print(String(format:"RESULT bytes=%d A_median_ms=%.6f B_median_ms=%.6f paired_gain_median_pct=%.3f gain_min=%.3f gain_max=%.3f positive_pairs=%d/16",weightBytes,median(aTimes),median(bTimes),median(gains),gains.min()!,gains.max()!,gains.filter{$0>0}.count))
}
print("SCALAR_MODE \(ProcessInfo.processInfo.environment["PROBE_SCALAR"] != nil)"); print("DEVICE \(device.name); exact pinned shader bodies; synthetic data; warmed repeated-buffer GPU timing; fastMath=true; NOT DRAM bandwidth or tg/s")
try experiment(5376,8192,4096)
try experiment(5376,16384,2048)
try experiment(592,256,128)
```

## Failed vector-helper run

```text
DEVICE Apple M4 Pro; exact pinned shader bodies; synthetic data; warmed repeated-buffer GPU timing; fastMath=true; NOT DRAM bandwidth or tg/s
CHECK K=5376 Na=8192 Nb=4096 fp32_words=24576 candidate_differences=6607 sabotaged_differences=18895
qkv_probe/qkv_probe.swift:63: Fatal error: bit equality or shader sabotage gate failed
```

## Initial short-warmup scalar run

```text
SCALAR_MODE true
DEVICE Apple M4 Pro; exact pinned shader bodies; synthetic data; warmed repeated-buffer GPU timing; fastMath=true; NOT DRAM bandwidth or tg/s
CHECK K=5376 Na=8192 Nb=4096 fp32_words=24576 candidate_differences=0 sabotaged_differences=12288
PAIR 00 A_ms=0.209706 B_ms=0.239445 gain_pct=-12.420
PAIR 01 A_ms=0.207633 B_ms=0.228839 gain_pct=-9.267
PAIR 02 A_ms=0.215812 B_ms=0.199508 gain_pct=8.172
PAIR 03 A_ms=0.184708 B_ms=0.179021 gain_pct=3.177
PAIR 04 A_ms=0.185518 B_ms=0.176576 gain_pct=5.065
PAIR 05 A_ms=0.173891 B_ms=0.161589 gain_pct=7.613
PAIR 06 A_ms=0.166846 B_ms=0.160703 gain_pct=3.823
PAIR 07 A_ms=0.164122 B_ms=0.160477 gain_pct=2.272
PAIR 08 A_ms=0.158513 B_ms=0.153172 gain_pct=3.487
PAIR 09 A_ms=0.152854 B_ms=0.148706 gain_pct=2.790
PAIR 10 A_ms=0.181964 B_ms=0.144852 gain_pct=25.621
PAIR 11 A_ms=0.150039 B_ms=0.139734 gain_pct=7.374
PAIR 12 A_ms=0.142060 B_ms=0.135117 gain_pct=5.138
PAIR 13 A_ms=0.139727 B_ms=0.134578 gain_pct=3.826
PAIR 14 A_ms=0.138370 B_ms=0.133375 gain_pct=3.745
PAIR 15 A_ms=0.139099 B_ms=0.135534 gain_pct=2.630
RESULT bytes=24772608 A_median_ms=0.165484 B_median_ms=0.156824 paired_gain_median_pct=3.784 gain_min=-12.420 gain_max=25.621 positive_pairs=14/16
CHECK K=5376 Na=16384 Nb=2048 fp32_words=36864 candidate_differences=0 sabotaged_differences=18432
PAIR 00 A_ms=0.205570 B_ms=0.198172 gain_pct=3.733
PAIR 01 A_ms=0.207128 B_ms=0.198000 gain_pct=4.610
PAIR 02 A_ms=0.206245 B_ms=0.198906 gain_pct=3.689
PAIR 03 A_ms=0.205534 B_ms=0.198500 gain_pct=3.544
PAIR 04 A_ms=0.210297 B_ms=0.199141 gain_pct=5.602
PAIR 05 A_ms=0.207297 B_ms=0.198922 gain_pct=4.210
PAIR 06 A_ms=0.214362 B_ms=0.198518 gain_pct=7.981
PAIR 07 A_ms=0.207167 B_ms=0.200940 gain_pct=3.099
PAIR 08 A_ms=0.207008 B_ms=0.199536 gain_pct=3.744
PAIR 09 A_ms=0.206839 B_ms=0.199359 gain_pct=3.752
PAIR 10 A_ms=0.207008 B_ms=0.199555 gain_pct=3.735
PAIR 11 A_ms=0.207203 B_ms=0.199271 gain_pct=3.981
PAIR 12 A_ms=0.206141 B_ms=0.198672 gain_pct=3.759
PAIR 13 A_ms=0.206055 B_ms=0.198422 gain_pct=3.847
PAIR 14 A_ms=0.206185 B_ms=0.198682 gain_pct=3.776
PAIR 15 A_ms=0.206797 B_ms=0.198576 gain_pct=4.140
RESULT bytes=37158912 A_median_ms=0.206923 B_median_ms=0.198794 paired_gain_median_pct=3.768 gain_min=3.099 gain_max=7.981 positive_pairs=16/16
CHECK K=592 Na=256 Nb=128 fp32_words=768 candidate_differences=0 sabotaged_differences=192
PAIR 00 A_ms=0.003039 B_ms=0.003073 gain_pct=-1.102
PAIR 01 A_ms=0.003031 B_ms=0.003026 gain_pct=0.172
PAIR 02 A_ms=0.003026 B_ms=0.003008 gain_pct=0.606
PAIR 03 A_ms=0.003018 B_ms=0.002987 gain_pct=1.046
PAIR 04 A_ms=0.002990 B_ms=0.002969 gain_pct=0.702
PAIR 05 A_ms=0.003005 B_ms=0.003000 gain_pct=0.174
PAIR 06 A_ms=0.003206 B_ms=0.002977 gain_pct=7.699
PAIR 07 A_ms=0.003422 B_ms=0.003422 gain_pct=0.000
PAIR 08 A_ms=0.003289 B_ms=0.003026 gain_pct=8.692
PAIR 09 A_ms=0.002982 B_ms=0.003039 gain_pct=-1.885
PAIR 10 A_ms=0.003018 B_ms=0.003099 gain_pct=-2.605
PAIR 11 A_ms=0.003016 B_ms=0.003034 gain_pct=-0.601
PAIR 12 A_ms=0.003026 B_ms=0.003029 gain_pct=-0.086
PAIR 13 A_ms=0.003047 B_ms=0.002987 gain_pct=2.005
PAIR 14 A_ms=0.002984 B_ms=0.003000 gain_pct=-0.521
PAIR 15 A_ms=0.003065 B_ms=0.003021 gain_pct=1.466
RESULT bytes=85248 A_median_ms=0.003026 B_median_ms=0.003023 paired_gain_median_pct=0.173 gain_min=-2.605 gain_max=8.692 positive_pairs=10/16
```

## Longer-warmup scalar run and A/A controls

```text
SCALAR_MODE true
DEVICE Apple M4 Pro; exact pinned shader bodies; synthetic data; warmed repeated-buffer GPU timing; fastMath=true; NOT DRAM bandwidth or tg/s
CHECK K=5376 Na=8192 Nb=4096 fp32_words=24576 candidate_differences=0 sabotaged_differences=12288
CONTROL_AA median_pct=0.006 min=-1.160 max=5.031
PAIR 00 A_ms=0.139766 B_ms=0.133969 gain_pct=4.327
PAIR 01 A_ms=0.139659 B_ms=0.134406 gain_pct=3.908
PAIR 02 A_ms=0.139513 B_ms=0.135424 gain_pct=3.019
PAIR 03 A_ms=0.138479 B_ms=0.133901 gain_pct=3.419
PAIR 04 A_ms=0.145711 B_ms=0.134693 gain_pct=8.180
PAIR 05 A_ms=0.139669 B_ms=0.136589 gain_pct=2.255
PAIR 06 A_ms=0.139227 B_ms=0.135039 gain_pct=3.101
PAIR 07 A_ms=0.139831 B_ms=0.134870 gain_pct=3.678
PAIR 08 A_ms=0.146297 B_ms=0.134128 gain_pct=9.073
PAIR 09 A_ms=0.139487 B_ms=0.134875 gain_pct=3.419
PAIR 10 A_ms=0.154667 B_ms=0.134591 gain_pct=14.916
PAIR 11 A_ms=0.139690 B_ms=0.134672 gain_pct=3.726
PAIR 12 A_ms=0.139299 B_ms=0.135195 gain_pct=3.036
PAIR 13 A_ms=0.139995 B_ms=0.134380 gain_pct=4.178
PAIR 14 A_ms=0.139060 B_ms=0.133344 gain_pct=4.287
PAIR 15 A_ms=0.139125 B_ms=0.135326 gain_pct=2.808
RESULT bytes=24772608 A_median_ms=0.139664 B_median_ms=0.134682 paired_gain_median_pct=3.702 gain_min=2.255 gain_max=14.916 positive_pairs=16/16
CHECK K=5376 Na=16384 Nb=2048 fp32_words=36864 candidate_differences=0 sabotaged_differences=18432
CONTROL_AA median_pct=-0.040 min=-11.392 max=3.476
PAIR 00 A_ms=0.205956 B_ms=0.197435 gain_pct=4.316
PAIR 01 A_ms=0.206349 B_ms=0.198029 gain_pct=4.202
PAIR 02 A_ms=0.206214 B_ms=0.197581 gain_pct=4.369
PAIR 03 A_ms=0.205854 B_ms=0.197943 gain_pct=3.997
PAIR 04 A_ms=0.206378 B_ms=0.199836 gain_pct=3.274
PAIR 05 A_ms=0.205953 B_ms=0.199565 gain_pct=3.201
PAIR 06 A_ms=0.205443 B_ms=0.198313 gain_pct=3.595
PAIR 07 A_ms=0.206117 B_ms=0.201677 gain_pct=2.202
PAIR 08 A_ms=0.206070 B_ms=0.200299 gain_pct=2.881
PAIR 09 A_ms=0.205549 B_ms=0.200086 gain_pct=2.731
PAIR 10 A_ms=0.207005 B_ms=0.199227 gain_pct=3.904
PAIR 11 A_ms=0.206794 B_ms=0.198023 gain_pct=4.429
PAIR 12 A_ms=0.206292 B_ms=0.199318 gain_pct=3.499
PAIR 13 A_ms=0.206565 B_ms=0.199240 gain_pct=3.677
PAIR 14 A_ms=0.206445 B_ms=0.199448 gain_pct=3.508
PAIR 15 A_ms=0.206659 B_ms=0.199328 gain_pct=3.678
RESULT bytes=37158912 A_median_ms=0.206253 B_median_ms=0.199279 paired_gain_median_pct=3.636 gain_min=2.202 gain_max=4.429 positive_pairs=16/16
CHECK K=592 Na=256 Nb=128 fp32_words=768 candidate_differences=0 sabotaged_differences=192
CONTROL_AA median_pct=-0.643 min=-7.317 max=5.721
PAIR 00 A_ms=0.003503 B_ms=0.003521 gain_pct=-0.518
PAIR 01 A_ms=0.003521 B_ms=0.003531 gain_pct=-0.295
PAIR 02 A_ms=0.003221 B_ms=0.003497 gain_pct=-7.893
PAIR 03 A_ms=0.003529 B_ms=0.003451 gain_pct=2.264
PAIR 04 A_ms=0.003307 B_ms=0.003263 gain_pct=1.357
PAIR 05 A_ms=0.003307 B_ms=0.003508 gain_pct=-5.716
PAIR 06 A_ms=0.003534 B_ms=0.003510 gain_pct=0.668
PAIR 07 A_ms=0.003279 B_ms=0.003344 gain_pct=-1.947
PAIR 08 A_ms=0.003312 B_ms=0.003297 gain_pct=0.474
PAIR 09 A_ms=0.003508 B_ms=0.003487 gain_pct=0.597
PAIR 10 A_ms=0.003510 B_ms=0.003513 gain_pct=-0.074
PAIR 11 A_ms=0.003344 B_ms=0.003503 gain_pct=-4.535
PAIR 12 A_ms=0.003531 B_ms=0.003505 gain_pct=0.743
PAIR 13 A_ms=0.003336 B_ms=0.003049 gain_pct=9.394
PAIR 14 A_ms=0.003531 B_ms=0.003188 gain_pct=10.784
PAIR 15 A_ms=0.003539 B_ms=0.003518 gain_pct=0.592
RESULT bytes=85248 A_median_ms=0.003505 B_median_ms=0.003500 paired_gain_median_pct=0.533 gain_min=-7.893 gain_max=10.784 positive_pairs=9/16
```
