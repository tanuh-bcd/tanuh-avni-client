/**
 * EdgeModelService unit tests.
 *
 * Verifies the registry-driven, lazy-load-once design described in the plan
 * (~/.claude/plans/composed-tumbling-bachman.md):
 *   • init() reads the registry once via TFLiteModule.getRegistry; the Promise is awaited
 *     by any subsequent inference call.
 *   • Each modelKey is loaded exactly once per app lifetime (no double load).
 *   • Plain vs encrypted asset types route to loadModel vs loadEncryptedModel.
 *   • Override blocks are JSON-stringified before crossing the bridge.
 */

const buildRegistry = (override) => ({
    defaultModel: 'oral-cancer-v1',
    models: {
        'oral-cancer-v1': {
            asset: {
                type: 'plain',
                path: 'models/oral-cancer-v1.tflite',
                encryptionKey: null,
                sha256OfPlaintext: null,
            },
            override: override ?? {
                input: {
                    type: 'image', width: 224, height: 224, channels: 3,
                    layout: 'CHW', dtype: 'float32',
                    normalization: { scale: 1 / 255, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }
                },
                output: { shape: [1, 2], dtype: 'float32', labels: ['oral_normal', 'oral_scc'] }
            }
        },
        'encrypted-model': {
            asset: {
                type: 'encrypted',
                path: 'models/encrypted-model.bin',
                encryptionKey: 'YmFzZTY0a2V5',
                sha256OfPlaintext: 'abcd1234',
            }
        }
    }
});

jest.mock('react-native', () => ({
    NativeModules: {
        TFLiteModule: {
            getRegistry: jest.fn(),
            loadModel: jest.fn(() => Promise.resolve(true)),
            loadEncryptedModel: jest.fn(() => Promise.resolve(true)),
            runInference: jest.fn(() => Promise.resolve([0.9, 0.1])),
            runInferenceOnImage: jest.fn(() => Promise.resolve([0.9, 0.1])),
        },
    },
}));

// The @Service decorator needs a no-op so the module loads outside the app container.
jest.mock('../../src/framework/bean/Service', () => () => (target) => target);

import {NativeModules} from 'react-native';
import EdgeModelService from '../../src/service/EdgeModelService';

describe('EdgeModelService', () => {
    let service;

    beforeEach(() => {
        jest.clearAllMocks();
        NativeModules.TFLiteModule.getRegistry.mockResolvedValue(buildRegistry());
        service = new EdgeModelService(null, null);
        service.init();
    });

    it('reads the registry once at init via TFLiteModule.getRegistry', async () => {
        await service._registryReady;
        expect(NativeModules.TFLiteModule.getRegistry).toHaveBeenCalledTimes(1);
    });

    it('loads a plain model on first inference call and routes via loadModel', async () => {
        await service.runInference('oral-cancer-v1', [1.0, 2.0]);

        expect(NativeModules.TFLiteModule.loadModel).toHaveBeenCalledTimes(1);
        const [modelKey, assetPath, overrideJson] = NativeModules.TFLiteModule.loadModel.mock.calls[0];
        expect(modelKey).toBe('oral-cancer-v1');
        expect(assetPath).toBe('models/oral-cancer-v1.tflite');
        expect(JSON.parse(overrideJson).input.width).toBe(224);
    });

    it('does not reload the model on subsequent inference calls', async () => {
        await service.runInference('oral-cancer-v1', [1.0]);
        await service.runInference('oral-cancer-v1', [2.0]);
        await service.runInferenceOnImage('oral-cancer-v1', '/tmp/x.jpg');

        expect(NativeModules.TFLiteModule.loadModel).toHaveBeenCalledTimes(1);
    });

    it('routes encrypted asset entries to loadEncryptedModel with key and sha256', async () => {
        await service.runInference('encrypted-model', [1.0]);

        expect(NativeModules.TFLiteModule.loadEncryptedModel).toHaveBeenCalledTimes(1);
        expect(NativeModules.TFLiteModule.loadModel).not.toHaveBeenCalled();
        const [modelKey, path, key, sha, override] = NativeModules.TFLiteModule.loadEncryptedModel.mock.calls[0];
        expect(modelKey).toBe('encrypted-model');
        expect(path).toBe('models/encrypted-model.bin');
        expect(key).toBe('YmFzZTY0a2V5');
        expect(sha).toBe('abcd1234');
        expect(override).toBeNull();
    });

    it('forwards inputData and modelKey to runInference correctly', async () => {
        await service.runInference('oral-cancer-v1', [1.0, 2.0]);

        expect(NativeModules.TFLiteModule.runInference).toHaveBeenCalledWith('oral-cancer-v1', [1.0, 2.0]);
    });

    it('returns the output array from the native module', async () => {
        const result = await service.runInference('oral-cancer-v1', [1.0]);

        expect(result).toEqual([0.9, 0.1]);
    });

    it('throws a clear error when modelKey is not in the registry', async () => {
        await expect(service.runInference('does-not-exist', [1.0]))
            .rejects.toThrow("no entry for modelKey 'does-not-exist'");
    });

    it('propagates load errors from the native module', async () => {
        NativeModules.TFLiteModule.loadModel.mockRejectedValueOnce(
            new Error('TFLITE_LOAD_ERROR: model file not found')
        );

        await expect(service.runInference('oral-cancer-v1', [1.0])).rejects.toThrow('model file not found');
    });

    it('propagates inference errors from the native module', async () => {
        await service.runInference('oral-cancer-v1', [1.0]);  // load first
        NativeModules.TFLiteModule.runInference.mockRejectedValueOnce(
            new Error('TFLITE_INFERENCE_ERROR: shape mismatch')
        );

        await expect(service.runInference('oral-cancer-v1', [1.0])).rejects.toThrow('shape mismatch');
    });

    it('surfaces registry load failure on first inference call (not at init)', async () => {
        NativeModules.TFLiteModule.getRegistry.mockRejectedValueOnce(new Error('asset not found'));
        const failingService = new EdgeModelService(null, null);
        failingService.init();

        await expect(failingService.runInference('oral-cancer-v1', [1.0])).rejects.toThrow('asset not found');
    });
});
