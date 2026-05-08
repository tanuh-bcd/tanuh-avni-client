import BaseService from "./BaseService";
import {CustomCardConfig} from 'avni-models';
import Service from "../framework/bean/Service";
import FileSystem from "../model/FileSystem";
import General from "../utility/General";
import SettingsService from "./SettingsService";
import AuthService from "./AuthService";
import MessageService from "./MessageService";
import RNFetchBlob from "react-native-blob-util";
import fs from 'react-native-fs';
import {IDP_PROVIDERS} from "../model/IdpProviders";

@Service("customCardConfigService")
class CustomCardConfigService extends BaseService {

    constructor(db, context) {
        super(db, context);
    }

    getSchema() {
        return CustomCardConfig.schema.name;
    }

    downloadHtmlFiles() {
        const configs = this.getAllNonVoided();
        const settingsService = this.getService(SettingsService);
        const serverUrl = settingsService.getSettings().serverURL;
        const targetDir = FileSystem.getCustomCardConfigsDir();

        configs.forEach((config) => {
            if (!config.htmlFileS3Key) return;
            const sourceUrl = `${serverUrl}/customCardConfigFile/${config.htmlFileS3Key}`;
            const targetFilePath = `${targetDir}/${config.htmlFileS3Key}`;
            General.logDebug("CustomCardConfigService", `Downloading ${sourceUrl} to ${targetFilePath}`);
            this.downloadFromUrl(sourceUrl, targetFilePath);
        });
    }

    async downloadFromUrl(url, targetFilePath) {
        const [token, idpType] = await Promise.all([
            this.getService(AuthService).getAuthProviderService().getAuthToken(),
            this.getService(SettingsService).getSettings().idpType
        ]);
        const headers = idpType === IDP_PROVIDERS.NONE ? {'USER-NAME': token} : {'AUTH-TOKEN': token};
        return RNFetchBlob
            .config({fileCache: true, path: targetFilePath})
            .fetch('GET', url, headers);
    }

    async readHtmlTemplate(config) {
        if (!config.htmlFileS3Key) return "";
        const targetPath = `${FileSystem.getCustomCardConfigsDir()}/${config.htmlFileS3Key}`;
        const exists = await fs.exists(targetPath);
        if (!exists) {
            const settingsService = this.getService(SettingsService);
            const sourceUrl = `${settingsService.getSettings().serverURL}/customCardConfigFile/${config.htmlFileS3Key}`;
            General.logDebug("CustomCardConfigService", `Template missing, fetching ${sourceUrl}`);
            await this.downloadFromUrl(sourceUrl, targetPath);
        }
        return fs.readFile(targetPath, "utf8");
    }

    resolveTranslations(config) {
        const defaults = config && typeof config.getTranslations === 'function'
            ? config.getTranslations()
            : {};
        const i18n = this.getService(MessageService).getI18n();
        // Resolve per access so the template can reference any org translation key, not just keys declared on the card.
        // Precedence: org translation (any locale) → card English default → key.
        return new Proxy({}, {
            get(_target, key) {
                if (typeof key !== 'string') return undefined;
                const translated = i18n.t(key);
                if (translated !== key) return translated;
                if (defaults[key] != null) return defaults[key];
                return key;
            }
        });
    }
}

export default CustomCardConfigService;
