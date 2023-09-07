// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, InferenceSession, Tensor} from 'onnxruntime-common';

import {SerializableModeldata, SerializableSessionMetadata, SerializableTensorMetadata, TensorMetadata} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {dataLocationStringToEnum, getTensorElementSize, logLevelStringToEnum, tensorDataTypeEnumToString, tensorDataTypeStringToEnum, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {getInstance} from './wasm-factory';
import {allocWasmString, checkLastError} from './wasm-utils';

/**
 * get the input/output count of the session.
 * @param sessionHandle the handle representing the session. should be non-zero.
 * @returns a tuple including 2 numbers, representing the input count and output count.
 */
const getSessionInputOutputCount = (sessionHandle: number): [number, number] => {
  const wasm = getInstance();
  const stack = wasm.stackSave();
  try {
    const dataOffset = wasm.stackAlloc(8);
    const errorCode = wasm._OrtGetInputOutputCount(sessionHandle, dataOffset, dataOffset + 4);
    if (errorCode !== 0) {
      checkLastError('Can\'t get session input/output count.');
    }
    return [wasm.HEAP32[dataOffset / 4], wasm.HEAP32[dataOffset / 4 + 1]];
  } finally {
    wasm.stackRestore(stack);
  }
};

/**
 * initialize ORT environment.
 * @param numThreads SetGlobalIntraOpNumThreads(numThreads)
 * @param loggingLevel CreateEnv(static_cast<OrtLoggingLevel>(logging_level))
 */
const initOrt = (numThreads: number, loggingLevel: number): void => {
  const errorCode = getInstance()._OrtInit(numThreads, loggingLevel);
  if (errorCode !== 0) {
    checkLastError('Can\'t initialize onnxruntime.');
  }
};

/**
 * intialize runtime environment.
 * @param env passed in the environment config object.
 */
export const initRuntime = async(env: Env): Promise<void> => {
  // init ORT
  initOrt(env.wasm.numThreads!, logLevelStringToEnum(env.logLevel));

  if (!BUILD_DEFS.DISABLE_WEBGPU) {
    // init JSEP if available

    // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
    const initJsep = require('./jsep/init').init;
    await initJsep(getInstance(), env);
  }
};

/**
 * valid data locations for input/output tensors.
 */
type SupportedTensorDataLocationForInputOutput = 'cpu'|'cpu-pinned'|'gpu-buffer';

type IOBindingState = {
  /**
   * cached data for input tensors. value is the tensor data. null means the data is not bound. Should be the same
   * length as inputNames.
   */
  inputs: Array<[data: number, location: SupportedTensorDataLocationForInputOutput]|null>;

  /**
   * cached data for output tensors. value is the tensor data. null means the data is not bound. Should be the same
   * length as outputNames.
   */
  outputs: Array<[data: number, location: SupportedTensorDataLocationForInputOutput]|null>;

  /**
   * the handle of IO binding.
   */
  readonly handle: number;

  /**
   * the preferred location for each output tensor.
   *
   * value is one of 'cpu', 'cpu-pinned', 'gpu-buffer'.
   */
  readonly outputPreferredLocations: readonly SupportedTensorDataLocationForInputOutput[];

  /**
   * enum value of the preferred location for each output tensor.
   */
  readonly outputPreferredLocationsEncoded: readonly number[];
};

/**
 *  tuple elements are: InferenceSession ID; inputNamesUTF8Encoded; outputNamesUTF8Encoded; bindingState
 */
type SessionMetadata = [
  inferenceSessionId: number, inputNamesUTF8Encoded: number[], outputNamesUTF8Encoded: number[],
  bindingState: IOBindingState|null
];

const activeSessions = new Map<number, SessionMetadata>();

/**
 * allocate the memory and memcpy the model bytes, preparing for creating an instance of InferenceSession.
 * @returns a 2-elements tuple - the pointer and size of the allocated buffer
 */
export const createSessionAllocate = (model: Uint8Array): [number, number] => {
  const wasm = getInstance();
  const modelDataOffset = wasm._malloc(model.byteLength);
  if (modelDataOffset === 0) {
    throw new Error(`Can't create a session. failed to allocate a buffer of size ${model.byteLength}.`);
  }
  wasm.HEAPU8.set(model, modelDataOffset);
  return [modelDataOffset, model.byteLength];
};

/**
 * create an inference session using the prepared buffer containing the model data.
 * @param modelData a 2-elements tuple containing the pointer and size of the model data buffer.
 * @param options an optional session options object.
 * @returns a 3-elements tuple containing [session handle, input names, output names]
 */
export const createSessionFinalize =
    (modelData: SerializableModeldata, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const wasm = getInstance();

      let sessionHandle = 0;
      let sessionOptionsHandle = 0;
      let ioBindingHandle = 0;
      let allocs: number[] = [];
      const inputNamesUTF8Encoded = [];
      const outputNamesUTF8Encoded = [];

      try {
        [sessionOptionsHandle, allocs] = setSessionOptions(options);

        sessionHandle = wasm._OrtCreateSession(modelData[0], modelData[1], sessionOptionsHandle);
        if (sessionHandle === 0) {
          checkLastError('Can\'t create a session.');
        }

        const [inputCount, outputCount] = getSessionInputOutputCount(sessionHandle);

        const inputNames = [];
        const outputNames = [];
        const outputPreferredLocations: SupportedTensorDataLocationForInputOutput[] = [];
        for (let i = 0; i < inputCount; i++) {
          const name = wasm._OrtGetInputName(sessionHandle, i);
          if (name === 0) {
            checkLastError('Can\'t get an input name.');
          }
          inputNamesUTF8Encoded.push(name);
          inputNames.push(wasm.UTF8ToString(name));
        }
        for (let i = 0; i < outputCount; i++) {
          const name = wasm._OrtGetOutputName(sessionHandle, i);
          if (name === 0) {
            checkLastError('Can\'t get an output name.');
          }
          outputNamesUTF8Encoded.push(name);
          outputNames.push(wasm.UTF8ToString(name));

          const location = typeof options?.preferredOutputLocation === 'string' ?
              options.preferredOutputLocation :
              options?.preferredOutputLocation?.[name] ?? 'cpu';
          if (location !== 'cpu' && location !== 'cpu-pinned' && location !== 'gpu-buffer') {
            throw new Error(`Not supported preferred output location: ${location}.`);
          }
          outputPreferredLocations.push(location);
        }

        // use IO binding only when at least one output is preffered to be on GPU.
        let bindingState: IOBindingState|null = null;
        if (outputPreferredLocations.some(l => l === 'gpu-buffer')) {
          ioBindingHandle = wasm._OrtCreateBinding(sessionHandle);
          if (ioBindingHandle === 0) {
            checkLastError('Can\'t create IO binding.');
          }

          bindingState = {
            handle: ioBindingHandle,
            inputs: new Array(inputCount).fill(null),
            outputs: new Array(outputCount).fill(null),
            outputPreferredLocations,
            outputPreferredLocationsEncoded: outputPreferredLocations.map(l => dataLocationStringToEnum(l)),
          };
        }

        activeSessions.set(sessionHandle, [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded, bindingState]);
        return [sessionHandle, inputNames, outputNames];
      } catch (e) {
        inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
        outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));

        if (ioBindingHandle !== 0) {
          wasm._OrtReleaseBinding(ioBindingHandle);
        }

        if (sessionHandle !== 0) {
          wasm._OrtReleaseSession(sessionHandle);
        }
        throw e;
      } finally {
        wasm._free(modelData[0]);
        if (sessionOptionsHandle !== 0) {
          wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
        }
        allocs.forEach(alloc => wasm._free(alloc));
      }
    };


/**
 * create an instance of InferenceSession.
 * @returns the metadata of InferenceSession. 0-value handle for failure.
 */
export const createSession =
    (model: Uint8Array, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const modelData: SerializableModeldata = createSessionAllocate(model);
      return createSessionFinalize(modelData, options);
    };

export const releaseSession = (sessionId: number): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot release session. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded, ioBindingState] = session;

  if (ioBindingState) {
    wasm._OrtReleaseBinding(ioBindingState.handle);
  }

  wasm.jsepUnregisterBuffers?.(sessionHandle);

  inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  wasm._OrtReleaseSession(sessionHandle);
  activeSessions.delete(sessionId);
};

const prepareInputOutputTensor =
    (tensor: TensorMetadata|null, tensorHandles: number[], allocs: number[], rawDatas: number[], sessionId: number,
     index: number): void => {
      if (!tensor) {
        tensorHandles.push(0);
        rawDatas.push(0);
        return;
      }

      const wasm = getInstance();

      const dataType = tensor[0];
      const dims = tensor[1];
      const location = tensor[3];

      let rawData: number;
      let dataByteLength: number;

      if (dataType === 'string' && location === 'gpu-buffer') {
        throw new Error('String tensor is not supported on GPU.');
      }

      if (location === 'gpu-buffer') {
        const gpuBuffer = tensor[2].gpuBuffer as GPUBuffer;
        const elementSizeInBytes = getTensorElementSize(tensorDataTypeStringToEnum(dataType))!;
        dataByteLength = dims.reduce((a, b) => a * b, 1) * elementSizeInBytes;
        rawData = wasm.jsepRegisterBuffer(sessionId, index, gpuBuffer, dataByteLength);
      } else {
        const data = tensor[2];

        if (Array.isArray(data)) {
          // string tensor
          dataByteLength = 4 * data.length;
          rawData = wasm._malloc(dataByteLength);
          allocs.push(rawData);
          let dataIndex = rawData / 4;
          for (let i = 0; i < data.length; i++) {
            if (typeof data[i] !== 'string') {
              throw new TypeError(`tensor data at index ${i} is not a string`);
            }
            wasm.HEAPU32[dataIndex++] = allocWasmString(data[i], allocs);
          }
        } else {
          dataByteLength = data.byteLength;
          rawData = wasm._malloc(dataByteLength);
          allocs.push(rawData);
          wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, dataByteLength), rawData);
        }
      }

      const stack = wasm.stackSave();
      const dimsOffset = wasm.stackAlloc(4 * dims.length);
      try {
        let dimIndex = dimsOffset / 4;
        dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
        const tensor = wasm._OrtCreateTensor(
            tensorDataTypeStringToEnum(dataType), rawData, dataByteLength, dimsOffset, dims.length,
            dataLocationStringToEnum(location));
        if (tensor === 0) {
          checkLastError(`Can't create tensor for input/output. session=${sessionId}, index=${index}.`);
        }
        tensorHandles.push(tensor);
        rawDatas.push(rawData);
      } finally {
        wasm.stackRestore(stack);
      }
    };

/**
 * perform inference run
 */
export const run = async(
    sessionId: number, inputIndices: number[], inputTensors: TensorMetadata[], outputIndices: number[],
    outputTensors: Array<TensorMetadata|null>, options: InferenceSession.RunOptions): Promise<TensorMetadata[]> => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot run inference. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded, ioBindingState] = session;

  const inputCount = inputIndices.length;
  const outputCount = outputIndices.length;

  let runOptionsHandle = 0;
  let runOptionsAllocs: number[] = [];

  const inputTensorHandles: number[] = [];
  const outputTensorHandles: number[] = [];
  const inputOutputAllocs: number[] = [];

  const beforeRunStack = wasm.stackSave();
  const inputValuesOffset = wasm.stackAlloc(inputCount * 4);
  const inputNamesOffset = wasm.stackAlloc(inputCount * 4);
  const outputValuesOffset = wasm.stackAlloc(outputCount * 4);
  const outputNamesOffset = wasm.stackAlloc(outputCount * 4);

  try {
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);

    const inputsRawData: number[] = [];
    const outputsRawData: number[] = [];

    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      prepareInputOutputTensor(
          inputTensors[i], inputTensorHandles, inputOutputAllocs, inputsRawData, sessionId, inputIndices[i]);
    }

    // create output tensors
    for (let i = 0; i < outputCount; i++) {
      prepareInputOutputTensor(
          outputTensors[i], outputTensorHandles, inputOutputAllocs, outputsRawData, sessionId,
          inputCount + outputIndices[i]);
    }

    // jsepOnRunStart is only available when JSEP is enabled.
    wasm.jsepOnRunStart?.(sessionId);

    try {
      let inputValuesIndex = inputValuesOffset / 4;
      let inputNamesIndex = inputNamesOffset / 4;
      let outputValuesIndex = outputValuesOffset / 4;
      let outputNamesIndex = outputNamesOffset / 4;
      for (let i = 0; i < inputCount; i++) {
        wasm.HEAPU32[inputValuesIndex++] = inputTensorHandles[i];
        wasm.HEAPU32[inputNamesIndex++] = inputNamesUTF8Encoded[inputIndices[i]];
      }
      for (let i = 0; i < outputCount; i++) {
        wasm.HEAPU32[outputValuesIndex++] = outputTensorHandles[i];
        wasm.HEAPU32[outputNamesIndex++] = outputNamesUTF8Encoded[outputIndices[i]];
      }

      if (ioBindingState) {
        const {handle, inputs, outputs, outputPreferredLocations, outputPreferredLocationsEncoded} = ioBindingState;

        if (inputs.length !== inputCount) {
          throw new Error(`input count from feeds is expected to be always equal to model's input: ${inputCount}.`);
        }

        // use IO binding
        for (let i = 0; i < inputCount; i++) {
          const index = inputIndices[i];
          const location = inputTensors[i][3];
          const boundInput = inputs[index];
          if (!boundInput || boundInput[0] !== inputsRawData[i] || boundInput[1] !== location) {
            // input is not bound or bound to a different tensor.
            const errorCode = wasm._OrtBindInput(handle, inputNamesUTF8Encoded[index], inputTensorHandles[i]);
            if (errorCode !== 0) {
              checkLastError(`Can't bind input[${i}] for session=${sessionId}.`);
            }
            inputs[index] = [inputsRawData[i], location];
          }
        }

        const processedOutputIndices = new Set<number>();

        // process pre-allocated outputs
        for (let i = 0; i < outputCount; i++) {
          const index = outputIndices[i];
          const location = outputTensors[i]?.[3];  // undefined means output is not pre-allocated.
          const boundOutput = outputs[index];

          if (location) {
            // output is pre-allocated. skip preferred location.
            if (boundOutput && boundOutput[0] === outputsRawData[i] && boundOutput[1] === location) {
              // output is bound to the same tensor. skip.
              continue;
            }

            // output is bound to a different tensor.
            const errorCode = wasm._OrtBindOutput(handle, outputNamesUTF8Encoded[index], outputTensorHandles[i], 0);
            if (errorCode !== 0) {
              checkLastError(`Can't bind pre-allocated output[${i}] for session=${sessionId}.`);
            }
            outputs[index] = [outputsRawData[i], location];
          } else {
            // output is not pre-allocated. reset preferred location.
            const errorCode =
                wasm._OrtBindOutput(handle, outputNamesUTF8Encoded[index], 0, outputPreferredLocationsEncoded[index]);
            if (errorCode !== 0) {
              checkLastError(`Can't bind output[${i}] to ${outputPreferredLocations[i]} for session=${sessionId}.`);
            }
            outputs[index] = [0, outputPreferredLocations[i]];
          }

          processedOutputIndices.add(index);
        }

        // process preferred location for unused outputs
        for (let i = 0; i < outputs.length; i++) {
          // if outputs[i] is null, it's either the initial state or bound to a location. this means no active tensor
          // is bound to the output.
          if (!processedOutputIndices.has(i) && outputs[i]) {
            const errorCode =
                wasm._OrtBindOutput(handle, outputNamesUTF8Encoded[i], 0, outputPreferredLocationsEncoded[i]);
            if (errorCode !== 0) {
              checkLastError(`Can't bind output[${i}] to ${outputPreferredLocations[i]} for session=${sessionId}.`);
            }
            outputs[i] = null;
          }
        }
      }

      let errorCode = ioBindingState ?
          wasm._OrtRunWithBinding(
              sessionHandle, ioBindingState.handle, outputCount, outputValuesOffset, runOptionsHandle) :
          wasm._OrtRun(
              sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
              outputValuesOffset, runOptionsHandle);


      const runPromise = wasm.jsepRunPromise;
      if (runPromise) {
        // jsepRunPromise is a Promise object. It is only available when JSEP is enabled.
        //
        // OrtRun() is a synchrnous call, but it internally calls async functions. Emscripten's ASYNCIFY allows it to
        // work in this way. However, OrtRun() does not return a promise, so when code reaches here, it is earlier than
        // the async functions are finished.
        //
        // To make it work, we created a Promise and resolve the promise when the C++ code actually reaches the end of
        // OrtRun(). If the promise exists, we need to await for the promise to be resolved.
        errorCode = await runPromise;
      }

      if (errorCode !== 0) {
        checkLastError('failed to call OrtRun().');
      }
    } finally {
      const jsepOnRunEnd = wasm.jsepOnRunEnd;
      if (jsepOnRunEnd) {
        // jsepOnRunEnd is only available when JSEP is enabled.
        //
        // it returns a promise, which is resolved or rejected when the following async functions are finished:
        // - errors thrown from backend.computeKernel()
        // - collecting GPU validation errors.
        await jsepOnRunEnd(sessionId);
      }
    }

    const output: TensorMetadata[] = [];

    for (let i = 0; i < outputCount; i++) {
      const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];
      if (tensor === outputTensorHandles[i]) {
        // output tensor is pre-allocated. no need to copy data.
        output.push(outputTensors[i]!);
        continue;
      }

      const beforeGetTensorDataStack = wasm.stackSave();
      // stack allocate 4 pointer value
      const tensorDataOffset = wasm.stackAlloc(4 * 4);

      let type: Tensor.Type|undefined, dataOffset = 0;
      try {
        const errorCode = wasm._OrtGetTensorData(
            tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
        if (errorCode !== 0) {
          checkLastError(`Can't access output tensor data on index ${i}.`);
        }
        let tensorDataIndex = tensorDataOffset / 4;
        const dataType = wasm.HEAPU32[tensorDataIndex++];
        dataOffset = wasm.HEAPU32[tensorDataIndex++];
        const dimsOffset = wasm.HEAPU32[tensorDataIndex++];
        const dimsLength = wasm.HEAPU32[tensorDataIndex++];
        const dims = [];
        for (let i = 0; i < dimsLength; i++) {
          dims.push(wasm.HEAPU32[dimsOffset / 4 + i]);
        }
        wasm._OrtFree(dimsOffset);

        const size = dims.reduce((a, b) => a * b, 1);
        type = tensorDataTypeEnumToString(dataType);

        const preferredLocation = ioBindingState?.outputPreferredLocations[outputIndices[i]];

        if (type === 'string') {
          if (preferredLocation === 'gpu-buffer') {
            throw new Error('String tensor is not supported on GPU.');
          }
          const stringData: string[] = [];
          let dataIndex = dataOffset / 4;
          for (let i = 0; i < size; i++) {
            const offset = wasm.HEAPU32[dataIndex++];
            const maxBytesToRead = i === size - 1 ? undefined : wasm.HEAPU32[dataIndex] - offset;
            stringData.push(wasm.UTF8ToString(offset, maxBytesToRead));
          }
          output.push([type, dims, stringData, 'cpu']);
        } else {
          if (preferredLocation === 'gpu-buffer') {
            const gpuBuffer = wasm.jsepGetBuffer(dataOffset);
            const elementSize = getTensorElementSize(dataType);
            if (elementSize === undefined) {
              throw new Error(`Unsupported data type: ${dataType}`);
            }
            output.push([
              type, dims, {gpuBuffer, download: wasm.jsepCreateDownloader(gpuBuffer, size * elementSize, type)},
              'gpu-buffer'
            ]);
          } else {
            const typedArrayConstructor = tensorTypeToTypedArrayConstructor(type);
            const data = new typedArrayConstructor(size);
            new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + data.byteLength));
            output.push([type, dims, data, 'cpu']);
          }
        }
      } finally {
        wasm.stackRestore(beforeGetTensorDataStack);
        if (type === 'string' && dataOffset) {
          wasm._free(dataOffset);
        }
        wasm._OrtReleaseTensor(tensor);
      }
    }

    return output;
  } finally {
    wasm.stackRestore(beforeRunStack);

    inputTensorHandles.forEach(v => wasm._OrtReleaseTensor(v));
    outputTensorHandles.forEach(v => wasm._OrtReleaseTensor(v));
    inputOutputAllocs.forEach(p => wasm._free(p));

    if (runOptionsHandle !== 0) {
      wasm._OrtReleaseRunOptions(runOptionsHandle);
    }
    runOptionsAllocs.forEach(p => wasm._free(p));
  }
};

/**
 * end profiling
 */
export const endProfiling = (sessionId: number): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error('invalid session id');
  }
  const sessionHandle = session[0];

  // profile file name is not used yet, but it must be freed.
  const profileFileName = wasm._OrtEndProfiling(sessionHandle);
  if (profileFileName === 0) {
    checkLastError('Can\'t get an profile file name.');
  }
  wasm._OrtFree(profileFileName);
};

export const extractTransferableBuffers = (tensors: readonly SerializableTensorMetadata[]): ArrayBufferLike[] => {
  const buffers: ArrayBufferLike[] = [];
  for (const tensor of tensors) {
    const data = tensor[2];
    if (!Array.isArray(data) && 'buffer' in data) {
      buffers.push(data.buffer);
    }
  }
  return buffers;
};
