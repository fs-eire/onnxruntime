// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #region File type declarations

/**
 * A string that represents a file's URL or path.
 *
 * Path is vailable only in onnxruntime-node or onnxruntime-web running in Node.js.
 */
export type FileUrlOrPath = string;

/**
 * A Blob object that represents a file.
 */
export type FileBlob = Blob;

/**
 * A Uint8Array, ArrayBuffer or SharedArrayBuffer object that represents a file content.
 *
 * When it is an ArrayBuffer or SharedArrayBuffer, the whole buffer is assumed to be the file content.
 */
export type FileData = Uint8Array|ArrayBufferLike;

/**
 * Represents a file that can be loaded by the ONNX Runtime JavaScript API.
 */
export type FileType = FileUrlOrPath|FileBlob|FileData;

// /**
//  * A tuple that represents a file with its checksum.
//  */
// export type FileWithChecksumType = [file: FileType, checksum: string];

// #endregion File type declarations

// #region Model file and external data file

export interface ExternalDataFileDescription {
  data: FileType;
  path?: string;
}

/**
 * Represents an external data file.
 *
 * When
 */
export type ExternalDataFileType = ExternalDataFileDescription|FileUrlOrPath;

// #endregion File type declarations


export interface OnnxModelOptions {
  /**
   * Specifying one file or multiple files that represents the external data.
   */
  data?: ExternalDataFile|ReadonlyArray<ExternalDataFile>;
}

export interface OnnxModelDescription extends OnnxModelOptions {
  /**
   * Specifying one file or multiple files that represents the model graph.
   *
   * When it is a single file, it should be either an ONNX format(.onnx) or ORT format(.ort) model file.
   *
   * When it is multiple files, it should be an array of files. Each file is a chunk of a part of the model data.
   */
  model: FileType|readonly FileType[];
}
