// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

module.exports = {
    manifest: {
        'linux/x64': [
            {
                type: 'nuget',
                index: 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/index.json',
                package: 'onnxruntime-linux-x64-gpu',
                version: '0.0.0',
                files: {
                    'libonnxruntime_providers_cuda.so': 'path_in_tgz/libonnxruntime_providers_cuda.so',
                    'libonnxruntime_providers_shared.so': 'path_in_tgz/libonnxruntime_providers_shared.so',
                    'libonnxruntime_providers_tensorrt.so': 'path_in_tgz/libonnxruntime_providers_tensorrt.so',
                }
            },
        ],
    },
};
