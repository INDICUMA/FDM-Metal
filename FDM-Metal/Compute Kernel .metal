//
//  Compute Kernel .metal
//  FDM-Metal
//
//  Created by 조일현 on 2023/02/27.
//

#include <metal_stdlib>
using namespace metal;



kernel void fdmKernel(
    device float *a [[ buffer(0) ]],
    device float *b [[ buffer(1) ]],
    device float *temp [[ buffer(2) ]],
                      
   const uint2 id [[ thread_position_in_grid ]],

    const device int  &stepT [[buffer(3)]],
    const device uint  &nx [[buffer(4)]])

{
    if ((id.x <= (nx-3)) && (id.y <= (nx-3))) {
        temp[((id.x+1)*nx+id.y+1)]  = (a[(id.x+1+1)*nx+id.y+1]+a[(id.x+1-1)*nx+id.y+1]+a[(id.x+1)*nx+id.y+1+1]+a[(id.x+1)*nx+id.y-1+1]-4*a[(id.x+1)*nx+id.y+1]) ;
        b[(id.x+1)*nx+id.y+1] = a[(id.x+1)*nx+id.y+1] + float(0.1 /(pow(1.0, 2)))*temp[(id.x+1)*nx+id.y+1];
    }
}


kernel void errorKernel(
    device float *a [[ buffer(0) ]],
    device float *b [[ buffer(1) ]],
    const uint2 id [[ thread_position_in_grid ]],
    device atomic_uint  &datau4 [[buffer(5)]],
    const device uint  &nx [[buffer(4)]])
{
    if ( id.x == 0){
        b[id.y] = b[nx+id.y];
        b[nx*(nx-1)+id.y] = b[nx*(nx-2)+id.y];
    }
        atomic_fetch_add_explicit(&datau4, uint((1.0E7)*abs(b[(id.x)*nx+id.y] - a[(id.x)*nx+id.y])), memory_order_relaxed);
}
