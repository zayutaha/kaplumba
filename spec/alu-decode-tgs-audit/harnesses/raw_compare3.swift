import Foundation
import Metal
import Darwin
setbuf(stdout,nil)
let root=CommandLine.arguments[1],lockPath=ProcessInfo.processInfo.environment["KAPLUMBA_GPU_LOCK_FILE"] ?? NSHomeDirectory()+"/.kaplumba/gpu.lock"
let fd=open(lockPath,O_RDWR|O_CREAT,mode_t(0o600));guard fd>=0 && flock(fd,LOCK_EX|LOCK_NB)==0 else{print("GPU busy");exit(75)};defer{flock(fd,LOCK_UN);close(fd)}
let src=try String(contentsOfFile:root+"/crates/kernels/src/linear_repacked.metal",encoding:.utf8),old=try String(contentsOfFile:root+"/base_linear_repacked.metal",encoding:.utf8),lin=try String(contentsOfFile:root+"/crates/kernels/src/linear.metal",encoding:.utf8)
func slice(_ s:String,_ a:String,_ b:String)->String{let x=s.range(of:a)!.lowerBound,y=s.range(of:b,range:x..<s.endIndex)!.lowerBound;return String(s[x..<y])}
let mcg=slice(lin,"static inline float decode_state_mcg(","static float decode_state(")
let helpers=slice(src,"static inline void extract_v_mcg_32(","static inline void extract_v_mcg_fast(")
let start="kernel void linear_gemv_ishizuki_k3_fused3_sk2(",end="kernel void linear_gemv_ishizuki_k3_colmajor_stream("
let base=slice(old,start,end).replacingOccurrences(of:"linear_gemv_ishizuki_k3_fused3_sk2",with:"probe3_a")
let active=slice(src,start,end);guard active.contains("extract_v_mcg_32_scalar(t0[j0], t0[j1], sh0, rsh0, v);") else{fatalError("fused3 does not use scalar decoder")}
let cand=active.replacingOccurrences(of:"linear_gemv_ishizuki_k3_fused3_sk2",with:"probe3_b")
let bad=cand.replacingOccurrences(of:"probe3_b",with:"probe3_bad").replacingOccurrences(of:"extract_v_mcg_32_scalar(t0[j0], t0[j1], sh0, rsh0, v);",with:"extract_v_mcg_32_scalar(t0[j0], t0[j1], sh0, rsh0, v); v[0] += 1.0f;")
let msl="#include <metal_stdlib>\nusing namespace metal;\n"+mcg+helpers+base+cand+bad
try msl.write(toFile:root+"/raw_compare3.metal",atomically:true,encoding:.utf8)
guard let d=MTLCreateSystemDefaultDevice(),let q=d.makeCommandQueue() else{fatalError("Metal unavailable")};let lib=try d.makeLibrary(source:msl,options:nil),ps=try ["probe3_a","probe3_b","probe3_bad"].map{try d.makeComputePipelineState(function:lib.makeFunction(name:$0)!)}
func median(_ a:[Double])->Double{let s=a.sorted();return(s[(s.count-1)/2]+s[s.count/2])/2}
func test(_ K:Int,_ ns:[Int])throws{var seed:UInt64=0x123456789abcdef;func rnd()->UInt32{seed ^= seed<<13;seed ^= seed>>7;seed ^= seed<<17;return UInt32(truncatingIfNeeded:seed)}
 func packed(_ n:Int)->MTLBuffer{let b=d.makeBuffer(length:n,options:.storageModeShared)!;let p=b.contents().bindMemory(to:UInt32.self,capacity:n/4);for i in 0..<n/4{p[i]=rnd()};return b}
 let w=ns.map{packed(K*$0*3/8)},x=d.makeBuffer(length:K*2,options:.storageModeShared)!;let xp=x.contents().bindMemory(to:UInt16.self,capacity:K);for i in 0..<K{xp[i]=Float16(Float(Int(rnd()%2001)-1000)/1024).bitPattern};let out=(0..<3).map{_ in ns.map{d.makeBuffer(length:$0*8,options:.storageModeShared)!}};let shape:[UInt32]=[UInt32(K),UInt32(ns[0]),UInt32(ns[1]),UInt32(ns[2]),0]
 func run(_ arm:Int,_ reps:Int)->Double{let cb=q.makeCommandBuffer()!,e=cb.makeComputeCommandEncoder()!;e.setComputePipelineState(ps[arm]);for i in 0..<3{e.setBuffer(w[i],offset:0,index:i);e.setBuffer(out[arm][i],offset:0,index:i+4)};e.setBuffer(x,offset:0,index:3);shape.withUnsafeBytes{e.setBytes($0.baseAddress!,length:$0.count,index:7)};for _ in 0..<reps{e.dispatchThreadgroups(MTLSize(width:ns.reduce(0,+)/16*2,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))};e.endEncoding();cb.commit();cb.waitUntilCompleted();guard cb.status == .completed else{fatalError("GPU cmd failed \(String(describing:cb.error))")};return(cb.gpuEndTime-cb.gpuStartTime)*1000/Double(reps)}
 for a in 0..<3{_=run(a,1)};var diff=0,badDiff=0;for c in 0..<3{let n=ns[c]*2,A=out[0][c].contents().bindMemory(to:UInt32.self,capacity:n),B=out[1][c].contents().bindMemory(to:UInt32.self,capacity:n),C=out[2][c].contents().bindMemory(to:UInt32.self,capacity:n);for i in 0..<n{if A[i] != B[i]{diff+=1};if A[i] != C[i]{badDiff+=1}}};print("CHECK3 K=\(K) Ns=\(ns) words=\(ns.reduce(0,+)*2) candidate_differences=\(diff) sabotage_differences=\(badDiff)");guard diff==0 && badDiff>0 else{fatalError("fused3 raw bit/sabotage gate failed")}
 for _ in 0..<96{_=run(0,8);_=run(1,8)};var gains=[Double](),aa=[Double](),at=[Double](),bt=[Double]();for _ in 0..<16{let a=run(0,8),b=run(1,8),c=run(1,8),z=run(0,8);at.append((a+z)/2);bt.append((b+c)/2);gains.append(100*((a+z)/(b+c)-1))};for _ in 0..<8{let a=run(0,8),b=run(0,8),c=run(0,8),z=run(0,8);aa.append(100*((a+z)/(b+c)-1))};print(String(format:"TIMING3 A=%.6f B=%.6f gain=%.3f%% spread=[%.3f,%.3f] wins=%d/16 AA=%.3f%% [% .3f,% .3f]",median(at),median(bt),median(gains),gains.min()!,gains.max()!,gains.filter{$0>0}.count,median(aa),aa.min()!,aa.max()!));print("GAINS3 \(gains.map{String(format:"%.3f",$0)}.joined(separator:","))")
}
print("DEVICE \(d.name); pinned 64-bit fused3 vs actual scalar fused3, raw split partials, synthetic")
try test(5376,[8192,4096,4096]);try test(5376,[16384,2048,2048]);try test(592,[256,128,128])
