//
//  main.swift
//  FDM-Metal
//
//  Created by 조일현 on 2023/02/27.
//


import MetalKit



//FDM parameters
let start =  CFAbsoluteTimeGetCurrent()
var nx : Int = 256
var ny : Int = 256
let t : Float = 1000.0
let diviN : Int = 8
let divi : Int = nx/diviN
let dx : Float = 1.0
let dy : Float = 1.0

let dt : Float = 0.1
let bndL : Float = 0.0
let bndR : Float = 100.0
let initV : Float = 50.0
var stepT = Int(t/dt)

// matrix
var u = [Float](repeating: initV, count: nx*ny)
var uR = [Float](repeating: initV, count: nx*ny)
var temp = [Float](repeating: 0.0, count: nx*ny)

// Boundary condition
for dy in 0..<nx{
    u[dy*ny] = bndL
    u[dy*ny+(nx-1)] = bndR
}
uR = u

//GPU work setup
guard
let device  = MTLCreateSystemDefaultDevice(),
let commandQueue = device.makeCommandQueue(),
let defaultLibrary = device.makeDefaultLibrary(),
let kernelFunction = defaultLibrary.makeFunction(name: "fdmKernel"),
let errorFunction = defaultLibrary.makeFunction(name: "errorKernel")
else {fatalError()}

let computePipelineState = try device.makeComputePipelineState(function: kernelFunction)

let computePipelineState2 = try device.makeComputePipelineState(function: errorFunction)


//create Buffers
var fdmBuffer : MTLBuffer = device.makeBuffer(bytes: &u, length: MemoryLayout<Float>.stride * u.count,  options : .storageModeShared)!
var fdmBuffer2 : MTLBuffer = device.makeBuffer(bytes: &uR,length: MemoryLayout<Float>.stride * uR.count, options : .storageModeShared)!
var tempBuffer = device.makeBuffer(bytes: &temp, length: MemoryLayout<Float>.stride * temp.count,options : .storageModeShared)
var errorBuffer = device.makeBuffer(bytes: &temp, length: MemoryLayout<Float>.stride * temp.count,options : .storageModeShared)!
var eSumBuffer = device.makeBuffer(bytes: &nx, length: MemoryLayout<Int>.stride ,options : .storageModeShared)!

let threadsPerThreadgroup = MTLSize(width:nx/divi, height:ny/divi , depth: 1)
var threadGroupSizeIsMultipleOfThreadExecutionWidth: Bool { true }
let threadGroupCount = MTLSize(width:divi, height: divi, depth: 1)



let startTime = CFAbsoluteTimeGetCurrent()

var count : Int = 0
var conv = "not converged"
var sum : Float = 0.0
var errorsum : Int = 0


//iterate
for _ in 0..<stepT{
    guard
        let commandBuffer : MTLCommandBuffer = commandQueue.makeCommandBuffer(),
        let commandBuffer2 : MTLCommandBuffer = commandQueue.makeCommandBuffer(),
        let commandBuffer3 : MTLCommandBuffer = commandQueue.makeCommandBuffer(),
        let computeEncoder : MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder(),
        let computeEncoder2 : MTLComputeCommandEncoder = commandBuffer3.makeComputeCommandEncoder(),
        let blitCommandEncoder = commandBuffer2.makeBlitCommandEncoder()
            
    else {fatalError()}
    
    
    computeEncoder.setComputePipelineState(computePipelineState)
    
    //set Buffers to computeEncoder
    computeEncoder.setBuffer(fdmBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(fdmBuffer2, offset: 0, index: 1)
    computeEncoder.setBuffer(errorBuffer, offset: 0, index: 2)
    computeEncoder.setBytes(&nx, length: MemoryLayout<Int>.stride, index: 4)
    computeEncoder.dispatchThreadgroups(threadGroupCount, threadsPerThreadgroup: threadsPerThreadgroup)
    computeEncoder.endEncoding()
    commandBuffer.commit()
  //  commandBuffer.waitUntilCompleted()
    
    
    sum = 0.0
    
    
    let rawpointer4 = eSumBuffer.contents()
  
     let datau4 = rawpointer4.bindMemory(to: Int.self, capacity: 1)
     datau4.pointee = 0
    
    
 //computeEncoder2 for error
    
    computeEncoder2.setComputePipelineState(computePipelineState2)

    computeEncoder2.setBuffer(fdmBuffer, offset: 0, index: 0)
    computeEncoder2.setBuffer(fdmBuffer2, offset: 0, index: 1)
    computeEncoder2.setBuffer(eSumBuffer, offset: 0, index: 5)
    computeEncoder2.setBytes(&nx, length: MemoryLayout<Int>.stride, index: 4)
    computeEncoder2.dispatchThreadgroups(threadGroupCount, threadsPerThreadgroup: threadsPerThreadgroup)
    computeEncoder2.endEncoding()
    commandBuffer3.commit()
    commandBuffer3.waitUntilCompleted()
    
    

   
    sum = Float(datau4.pointee)/(1.0E7)
 
    if (sum<1.0E-5){
        conv = "converged at"
        break;
    }
    count += 1
    
    blitCommandEncoder.copy(from: fdmBuffer2,
                            sourceOffset: 0,
                            to: fdmBuffer,
                            destinationOffset: 0,
                            size:
                                MemoryLayout<Int>.stride*u.count/2)
    
    blitCommandEncoder.endEncoding()
    
    commandBuffer2.commit()
 //   commandBuffer2.waitUntilCompleted()
}
//capture GPU time
print("GPU conversion time:", CFAbsoluteTimeGetCurrent() - startTime)

let rawpointer2 = fdmBuffer2.contents()
var datau2 = rawpointer2.bindMemory(to: Float.self, capacity: u.count)


for i in 0..<nx{
    for j in 0..<ny{
        print(datau2[i*nx+j], terminator: " ")
    }
    print("\n")
}

print(conv, count,"diff sum:",sum)

