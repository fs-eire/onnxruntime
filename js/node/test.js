ort = require('.');
ort.env.logLevel = 'verbose';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function printMetadataTable(label, metadata) {
    const idxW = String(metadata.length - 1).length;
    const nameW = Math.max(...metadata.map(m => m.name.length), 4);
    const typeW = Math.max(...metadata.map(m => m.type.length), 4);

    const header = `${'#'.padStart(idxW)}  ${'Name'.padEnd(nameW)}  ${'Type'.padEnd(typeW)}  Shape`;
    const sep = '-'.repeat(header.length + 10);

    console.log(`\n ==== ${label} (${metadata.length}) ====`);
    console.log(header);
    console.log(sep);
    metadata.forEach((m, i) => {
        const shapeStr = `[${m.shape.join(', ')}]`;
        console.log(`${String(i).padStart(idxW)}  ${m.name.padEnd(nameW)}  ${m.type.padEnd(typeW)}  ${shapeStr}`);
    });
}

// ---------------------------------------------------------------------------
// CreateKvCache – derive layer count / head count / head dim from the session
//                 metadata and return a mutable cache object.
// ---------------------------------------------------------------------------

function CreateKvCache(session, maxSeqLen) {
    // Discover KV structure from input metadata
    const kvInputs = session.inputMetadata.filter(m => m.name.startsWith('past_key_values.'));
    const numLayers = kvInputs.length / 2; // key + value per layer

    // Use the first KV input shape to get numHeads and headDim
    // shape: [batch_size, numHeads, past_sequence_length, headDim]
    const firstKv = kvInputs[0];
    const numHeads = typeof firstKv.shape[1] === 'number' ? firstKv.shape[1] : 8;
    const headDim = typeof firstKv.shape[3] === 'number' ? firstKv.shape[3] : 128;

    console.log(`\nKV cache config: ${numLayers} layers, ${numHeads} heads, dim=${headDim}, maxSeqLen=${maxSeqLen}`);

    return {
        numLayers,
        numHeads,
        headDim,
        maxSeqLen,
        pastSeqLen: 0,     // how many tokens are already cached
        layers: Array.from({ length: numLayers }, () => ({ key: null, value: null })),
    };
}

// ---------------------------------------------------------------------------
// run – execute one forward pass (prefill or decode).
//
//   opts.B        – batch size
//   opts.seq_len  – number of new tokens (prompt length for prefill, 1 for decode)
//   opts.kv_cache – object returned by CreateKvCache (mutated in-place)
// ---------------------------------------------------------------------------

async function run(session, { B, seq_len, kv_cache }) {
    const { numLayers, numHeads, headDim, pastSeqLen } = kv_cache;
    const totalSeqLen = pastSeqLen + seq_len;

    // ---- Build feeds -------------------------------------------------------

    // input_ids – dummy token ids (all ones)
    const inputIds = new ort.Tensor(
        'int64',
        new BigInt64Array(B * seq_len).fill(1n),
        [B, seq_len],
    );

    // attention_mask – all ones (attend to every position)
    const attentionMask = new ort.Tensor(
        'int64',
        new BigInt64Array(B * totalSeqLen).fill(1n),
        [B, totalSeqLen],
    );

    const feeds = {
        'input_ids': inputIds,
        'attention_mask': attentionMask,
    };

    // Track temporary feed tensors so we can dispose them after run
    const tempFeedTensors = [inputIds, attentionMask];

    // past_key_values – either from cache or zero-sized for the first run
    for (let i = 0; i < numLayers; i++) {
        if (kv_cache.layers[i].key) {
            feeds[`past_key_values.${i}.key`] = kv_cache.layers[i].key;
            feeds[`past_key_values.${i}.value`] = kv_cache.layers[i].value;
        } else {
            const elems = B * numHeads * pastSeqLen * headDim; // 0 on first call
            const kTensor = new ort.Tensor(
                'float16', new Uint16Array(elems), [B, numHeads, pastSeqLen, headDim]);
            const vTensor = new ort.Tensor(
                'float16', new Uint16Array(elems), [B, numHeads, pastSeqLen, headDim]);
            feeds[`past_key_values.${i}.key`] = kTensor;
            feeds[`past_key_values.${i}.value`] = vTensor;
            tempFeedTensors.push(kTensor, vTensor);
        }
    }

    // ---- Run ---------------------------------------------------------------
    console.log(`\nRunning inference: B=${B}, seq_len=${seq_len}, past=${pastSeqLen}, total=${totalSeqLen}`);
    const t0 = performance.now();
    const results = await session.run(feeds);
    const elapsed = (performance.now() - t0).toFixed(1);
    console.log(`Inference done in ${elapsed} ms`);

    // ---- Dispose temporary feed tensors (no longer needed after run) -------
    for (const t of tempFeedTensors) {
        t.dispose();
    }

    // ---- Update KV cache (dispose old GPU tensors, keep new ones) ----------
    for (let i = 0; i < numLayers; i++) {
        // Dispose old KV tensors from the previous run (GPU-backed)
        if (kv_cache.layers[i].key) {
            kv_cache.layers[i].key.dispose();
            kv_cache.layers[i].value.dispose();
        }
        kv_cache.layers[i].key = results[`present.${i}.key`];
        kv_cache.layers[i].value = results[`present.${i}.value`];
    }
    kv_cache.pastSeqLen = totalSeqLen;

    // ---- Report logits & dispose -------------------------------------------
    const logits = results['logits'];
    console.log(`Logits shape: [${logits.dims}], type: ${logits.type}`);
    logits.dispose();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    let session;
    try {
        const sessionOptions = {
            executionProviders: ['webgpu'],
            logSeverityLevel: 0, // Verbose
        };
        session = await ort.InferenceSession.create(
            'E:\\pg\\learn-llamacpp\\models\\model_builder\\qwen3-4b-instruct-2507-pruned\\model.onnx',
            sessionOptions);

        printMetadataTable('Model Inputs', session.inputMetadata);
        printMetadataTable('Model Outputs', session.outputMetadata);

        // run prefill
        const kv_cache = CreateKvCache(session, 4096);
        await run(session, {
            B: 1,
            seq_len: 1024,
            kv_cache
        });

        // // run one iteration of generation
        // await run(session, {
        //     B: 1,
        //     seq_len: 1,
        //     kv_cache
        // });

        // Dispose remaining KV cache tensors
        for (const layer of kv_cache.layers) {
            if (layer.key) layer.key.dispose();
            if (layer.value) layer.value.dispose();
        }
    }
    catch (e) {
        console.error(e);
    } finally {
        if (session) {
            await session.release();
        }
    }
}

main();
