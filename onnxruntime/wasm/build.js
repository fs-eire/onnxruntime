#!/usr/bin/env node

'use strict';

const child_process = require('child_process');
const fs = require('fs');
const path = require('path');

const ORT_FOLDER = path.normalize(path.join(__dirname, '../..'));
const EMSDK_FOLDER = path.normalize(path.join(ORT_FOLDER, 'cmake/external/emsdk'));
const EMSDK_BIN = path.join(EMSDK_FOLDER, 'emsdk');
const EMCC_BIN = path.normalize(path.join(EMSDK_FOLDER, 'upstream/emscripten/em++'));

const INC_SEARCH_FOLDERS = [
    `${ORT_FOLDER}/onnxruntime/core/mlas/inc`, // mlas.h
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib`, // mlasi.h
    `${ORT_FOLDER}/include/onnxruntime`
];

const SOURCE_FILES = [
    `${ORT_FOLDER}/onnxruntime/test/mlas/unittest.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/threading.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/platform.cpp`,
    `${ORT_FOLDER}/onnxruntime/core/mlas/lib/sgemm.cpp`,
];

let args = `
${INC_SEARCH_FOLDERS.map(i => `-I${path.normalize(i)}`).join(' ')}
-DEIGEN_MPL2_ONLY                                             
-DMLAS_NO_ONNXRUNTIME_THREADPOOL                              
-std=c++14                                                    
-s WASM=1                                                     
-s NO_EXIT_RUNTIME=0                                          
-s ALLOW_MEMORY_GROWTH=1                                      
-s SAFE_HEAP=0                                                
-s MODULARIZE=1                                               
-s SAFE_HEAP_LOG=0                                            
-s STACK_OVERFLOW_CHECK=0                                     
-s EXPORT_ALL=0                                               
-o out_wasm_main.js                                           
-s "EXPORTED_FUNCTIONS=[_main]"                     
${SOURCE_FILES.map(path.normalize).join(' ')}
`
    ;

if (process.argv.indexOf('simd') !== -1) {
    args += `
-msimd128 
${path.normalize(`${ORT_FOLDER}/onnxruntime/core/mlas/lib/wasm_simd/sgemmc.cpp`)}
`
} else {
    args += `
${path.normalize(`${ORT_FOLDER}/onnxruntime/core/mlas/lib/wasm/sgemmc.cpp`)}
`
}

if (process.argv.indexOf('debug') !== -1) {
    args += `
-s VERBOSE=1
-s ASSERTIONS=1
-g4 
`
} else {
    args += `
-s VERBOSE=0
-s ASSERTIONS=0
-O3                                                          
`
}


if (!fs.existsSync(EMCC_BIN)) {

    // One-time installation: 'emsdk install latest'

    const install = child_process.spawnSync(`${EMSDK_BIN} install latest`, { shell: true, stdio: 'inherit', cwd: EMSDK_FOLDER });
    if (install.status !== 0) {
        if (install.error) {
            console.error(install.error);
        }
        process.exit(install.status === null ? undefined : install.status);
    }

    // 'emsdk activate latest'

    const activate = child_process.spawnSync(`${EMSDK_BIN} activate latest`, { shell: true, stdio: 'inherit', cwd: EMSDK_FOLDER });
    if (activate.status !== 0) {
        if (activate.error) {
            console.error(activate.error);
        }
        process.exit(activate.status === null ? undefined : activate.status);
    }
}

console.log(`${EMCC_BIN} ${args.split('\n').map(i => i.trim()).filter(i => i !== '').join(' ')}`);
const emccBuild = child_process.spawnSync(EMCC_BIN, args.split('\n').map(i => i.trim()), { shell: true, stdio: 'inherit', cwd: __dirname });

if (emccBuild.error) {
    console.error(emccBuild.error);
    process.exit(emccBuild.status === null ? undefined : emccBuild.status);
}
