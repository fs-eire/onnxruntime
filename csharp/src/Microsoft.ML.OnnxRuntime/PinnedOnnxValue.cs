using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// A lightweighted class to wrap a numberic tensor OrtValue.
    /// When used as model outputs, the passed in shape have to match the actual output value.
    ///
    /// This class is a part of prototype and may be changed in future version.
    /// </summary>
    public class PinnedOnnxValue : IDisposable
    {
        public MemoryHandle PinnedMemory { get; protected set; }
        public IntPtr Value { get; protected set; }

        internal PinnedOnnxValue(MemoryHandle pinnedMemory, ulong sizeInBytes, long[] shape, TensorElementType elementType)
        {
            PinnedMemory = pinnedMemory;
            Value = default;
            unsafe
            {
                var status = NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                    NativeMemoryInfo.DefaultInstance.Handle,
                    (IntPtr)pinnedMemory.Pointer,
                    (UIntPtr)sizeInBytes,
                    shape,
                    (UIntPtr)shape.Length,
                    elementType,
                    out IntPtr value);
                NativeApiStatus.VerifySuccess(status);
                Value = value;
            }
        }

        public static PinnedOnnxValue Create(Memory<int> data, long[] shape)
        {
            return Create(data.Pin(), data.Length * sizeof(int), shape, TensorElementType.Int32);
        }
        public static PinnedOnnxValue Create(Memory<float> data, long[] shape)
        {
            return Create(data.Pin(), data.Length * sizeof(float), shape, TensorElementType.Float);
        }
        public static PinnedOnnxValue Create(Memory<long> data, long[] shape)
        {
            return Create(data.Pin(), data.Length * sizeof(long), shape, TensorElementType.Int64);
        }

        // TODO: support other element types

        // TODO: deal with unsupported OrtValue types, including string tensor, sequences and maps.

        private static PinnedOnnxValue Create(MemoryHandle pinnedMemory, int length, long[] shape, TensorElementType elementType)
        {
            try
            {
                return new PinnedOnnxValue(pinnedMemory, (ulong)length, shape, elementType);
            }
            catch
            {
                pinnedMemory.Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            ((IDisposable)PinnedMemory).Dispose();

            if (Value != IntPtr.Zero)
            {
                NativeMethods.OrtReleaseValue(Value);
            }
        }
    }
}
