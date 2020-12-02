# ONNX Runtime for WebAssembly

Currently, only MLAS can be built to WASM. More parts in ORT will be built into WASM in future.

## HOW TO BUILD:

### Before build

1. Install Node.js 14.x
2. Syncup git submodules (cmake/external/emsdk)
3. Perform one-time setup
    - `emsdk install latest`
    - `emsdk activate latest`

### Building

1. call `build.cmd`

### Test

1. Use Node.js to launch.
   - call `test.cmd`

### Output

Files `out_wasm_main.*` will be outputted.