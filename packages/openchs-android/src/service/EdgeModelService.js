import BaseService from "./BaseService";
import Service from "../framework/bean/Service";
import {NativeModules} from "react-native";

/**
 * EdgeModelService — JS surface for on-device TFLite inference.
 *
 * Overall design (~/.claude/plans/composed-tumbling-bachman.md):
 *   • The native module (`TFLiteModule`) is generic: a `modelKey` selects which model
 *     to use; preprocessing parameters come from a per-flavour `assets/models/registry.json`.
 *   • This service caches the registry on app boot, then lazy-loads each model on first
 *     use. Once loaded the interpreter stays for the app lifetime *until* the OS evicts
 *     it under memory pressure — at which point the next inference call self-heals via
 *     the native side's cached load-args.
 *   • Plain or AES-GCM-encrypted assets are both supported; the registry entry's
 *     `asset.type` field selects the load path.
 *
 * Rule usage:
 *   await params.services.edgeModelService.runInferenceOnImage('oral-cancer-v1', imagePath)
 *   await params.services.edgeModelService.runInference('oral-cancer-v1', floatArray)
 */
@Service("edgeModelService")
class EdgeModelService extends BaseService {
    constructor(db, context) {
        super(db, context);
        this._registry = null;
        this._registryReady = null;
        this._loaded = new Set();
    }

    /**
     * BeanRegistry calls init() synchronously at app boot. We can't block here, but we
     * can kick off the registry read and stash the Promise — any subsequent inference
     * call will await this before consulting `_registry`. Failures are surfaced lazily
     * (on the first inference call), not at app boot, so a missing or malformed registry
     * doesn't break the rest of the app.
     */
    init() {
        this._registryReady = NativeModules.TFLiteModule.getRegistry()
            .then(parsed => { this._registry = parsed; })
            .catch(e => {
                console.error('EdgeModelService: failed to load assets/models/registry.json', e);
                throw e;
            });
    }

    async runInference(modelKey, inputData) {
        await this._ensureLoaded(modelKey);
        return NativeModules.TFLiteModule.runInference(modelKey, inputData);
    }

    /**
     * Run inference on an image file path. Native handles decode → resize → normalise →
     * layout-transpose, all driven by the resolved ModelContract. `imagePath` is an
     * absolute path on the device (e.g. from react-native-image-picker, with `file://`
     * stripped).
     */
    async runInferenceOnImage(modelKey, imagePath) {
        await this._ensureLoaded(modelKey);
        return NativeModules.TFLiteModule.runInferenceOnImage(modelKey, imagePath);
    }

    /**
     * Lazy-load the interpreter for `modelKey` exactly once per app lifetime. Idempotent:
     * if the native side has evicted the interpreter under memory pressure it self-heals
     * via its cached load-args, so we don't re-issue the load call here.
     */
    async _ensureLoaded(modelKey) {
        await this._registryReady;
        if (this._loaded.has(modelKey)) return;

        const entry = this._registry?.models?.[modelKey];
        if (!entry) {
            throw new Error(`EdgeModelService: no entry for modelKey '${modelKey}' in assets/models/registry.json`);
        }
        const overrideJson = entry.override ? JSON.stringify(entry.override) : null;

        if (entry.asset?.type === 'encrypted') {
            await NativeModules.TFLiteModule.loadEncryptedModel(
                modelKey,
                entry.asset.path,
                entry.asset.encryptionKey,
                entry.asset.sha256OfPlaintext,
                overrideJson
            );
        } else {
            await NativeModules.TFLiteModule.loadModel(
                modelKey,
                entry.asset.path,
                overrideJson
            );
        }
        this._loaded.add(modelKey);
    }
}

export default EdgeModelService;
