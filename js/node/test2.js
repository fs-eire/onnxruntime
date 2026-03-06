ort = require('.');
ort.env.logLevel = 'verbose';

async function main() {
    let session;
    try {
        // wait 5 seconds to allow debugger to attach
        await new Promise(resolve => setTimeout(resolve, 5000));

        const sessionOptions = {
            executionProviders: ['webgpu'],
            logSeverityLevel: 0, // Verbose
        };
        session = await ort.InferenceSession.create(
            'D:\\code\\onnxruntime\\js\\test\\data\\node\\opset19\\test_acos\\model.onnx',
            sessionOptions);

        await session.run({
            'x': new ort.Tensor('float32', new Float32Array(3 * 4 * 5), [3, 4, 5])
        });

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
