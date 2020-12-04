const wasm_factory = require('./out_wasm_main');

wasm_factory().then((o) => {
    setTimeout( () => {
        console.log('----------------start----------------');
        o.__Z18mlas_unittest_mainv();
    }, 2000);
});