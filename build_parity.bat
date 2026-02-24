:: build parity tests
::

:: build static

:: call build --config RelWithDebInfo --parallel --use_webgpu --build_dir build_parity\static --skip_tests --build_shared_lib --target onnxruntime
:: copy /Y build_parity\static\Release\Release\onnxruntime.* E:\pg\2026-01-18\runtime_static\

:: build dynamic

:: call build --config RelWithDebInfo --build_dir build_parity\generic --enable_generic_interface --build_shared_lib --skip_tests --target onnxruntime

call build --config RelWithDebInfo --use_webgpu shared_lib --build_dir build_parity\shared --skip_tests --target onnxruntime_providers_webgpu

:: copy /Y build_parity\generic\Release\Release\onnxruntime.* E:\pg\2026-01-18\runtime_ep\
copy /Y build_parity\shared\Release\Release\onnxruntime_providers_webgpu.* E:\pg\2026-01-18\runtime_ep\


@REM How to build WebGPU EP API parity:

@REM build baseline:
@REM onnxruntime (branch: main)
@REM build --config Release --parallel --use_webgpu --build_dir build_parity\static --skip_tests --build_shared_lib --target onnxruntime
@REM artifacts: onnxruntime.dll
@REM onnxruntime-genai (branch: main)
@REM build --config Release --skip_tests --skip_wheel --ort_home <ort_home>
@REM artifacts: onnxruntime-genai.dll, model_benchmark.exe

@REM build my change:
@REM onnxruntime (branch: ep-api, https://github.com/fs-eire/onnxruntime)
@REM build --config Release --parallel --use_webgpu shared_lib --build_dir build_parity\shared --skip_tests --target onnxruntime_providers_webgpu
@REM artifacts: onnxruntime_providers_webgpu.dll
@REM build --config Release --parallel --build_dir build_parity\generic --enable_generic_interface --build_shared_lib --skip_tests --target onnxruntime
@REM artifacts: onnxruntime.dll
@REM onnxruntime-genai (branch: fs-eire/ep-api, https://github.com/fs-eire/onnxruntime-genai)
@REM build --config Release --skip_tests --skip_wheel --ort_home <ort_home>
@REM artifacts: onnxruntime-genai.dll, model_benchmark.exe
