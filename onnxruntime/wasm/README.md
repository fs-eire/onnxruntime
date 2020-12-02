# ONNX Runtime for WebAssembly

Currently, only MLAS can be built to WASM. More parts in ORT will be built into WASM in future.

## HOW TO BUILD:

### Before build

1. Install Node.js 14.x
2. Syncup git submodules (cmake/external/emsdk)
3. Perform one-time setup (This will be implicit called by build.cmd. It takes some time to download.)
    - `emsdk install latest`
    - `emsdk activate latest`

### Building

- Build WebAssembly MVP
   - call `build.cmd`
   - call `build.cmd debug` for debug build
- Build WebAssembly SIMD128
   - call `build.cmd simd`
   - call `build.cmd simd debug` for debug build

### Test

- Use Node.js to launch.
   - call `test.cmd`

### Output

Files `out_wasm_main.*` will be outputted.