// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// init JSEP
Module['jsepInit'] = (backend, alloc, free, copy, copyAsync, createKernel, releaseKernel, runKernel) => {
  Module.jsepBackend = backend;
  Module.jsepAlloc = alloc;
  Module.jsepFree = free;
  Module.jsepCopy = copy;
  Module.jsepCopyAsync = copyAsync;
  Module.jsepCreateKernel = createKernel;
  Module.jsepReleaseKernel = releaseKernel;
  Module.jsepRunKernel = runKernel;

  // This is a simplified version of cwrap() with options.async === true (-sASYNCIFY=1)
  // It removes some overhead in cwarp() and ccall() that we don't need.
  //
  // Currently in JSEP build, we only use this for the following functions:
  // - OrtRun()
  // - OrtRunWithBinding()
  // - OrtBindInput()
  const jsepWrapAsync = (func) => {
    return (...args) => {
      const previousAsync = Asyncify.currData;
      const ret = func(...args);
      if (Asyncify.currData != previousAsync) {
        return Asyncify.whenDone();
      }
      return ret;
    };
  };

  // This is a wrapper for OrtRun() and OrtRunWithBinding() to ensure that Promises are handled correctly.
  const runAsync = (runAsyncFunc) => {
    return async (...args) => {
      try {
        if (Module.jsepSessionState) {
          throw new Error('Session already started');
        }

        const state = Module.jsepSessionState = {sessionHandle: args[0], errors: []};

        const ret = await runAsyncFunc(...args);

        if (Module.jsepSessionState !== state) {
          throw new Error('Session mismatch');
        }

        backend['flush']();

        const errorPromises = state.errors;
        if (errorPromises.length > 0) {
          let errors = await Promise.all(errorPromises);
          errors = errors.filter(e => e);
          if (errors.length > 0) {
            throw new Error(errors.join('\n'));
          }
        }

        return ret;
      } finally {
        Module.jsepSessionState = null;
      }
    };
  };

  Module['_OrtRun'] = runAsync(jsepWrapAsync(Module['_OrtRun']));
  Module['_OrtRunWithBinding'] = runAsync(jsepWrapAsync(Module['_OrtRunWithBinding']));
  Module['_OrtBindInput'] = jsepWrapAsync(Module['_OrtBindInput']);

  Module['jsepRegisterBuffer'] = (sessionId, index, buffer, size) => {
    return backend['registerBuffer'](sessionId, index, buffer, size);
  };

  Module['jsepUnregisterBuffers'] = sessionId => {
    backend['unregisterBuffers'](sessionId);
  };

  Module['jsepGetBuffer'] = (dataId) => {
    return backend['getBuffer'](dataId);
  };

  Module['jsepCreateDownloader'] = (gpuBuffer, size, type) => {
    return backend['createDownloader'](gpuBuffer, size, type);
  };

  Module['jsepCreateDisposer'] = (gpuBuffer) => {
    return backend['createDisposer'](gpuBuffer);
  };
};
