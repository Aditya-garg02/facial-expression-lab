import { readFileSync } from 'node:fs';

export default {
  plugins: [{
    name: 'browser-lab-assets',
    apply: 'build',
    generateBundle() {
      const assets = {
        'detector.onnx': 'models/face_detection_yunet_2023mar.onnx',
        'profile.json': 'artifacts/inference_profile.json',
      };
      for (const name of ['ort.wasm.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
        assets[name] = 'node_modules/onnxruntime-web/dist/' + name;
      }
      for (const [name, path] of Object.entries(assets)) {
        this.emitFile({type: 'asset', fileName: 'browser-lab/' + name, source: readFileSync(path)});
      }
    },
  }],
};
