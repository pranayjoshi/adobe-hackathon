import {defineConfig} from 'vite';
import {resolve} from 'node:path';
export default defineConfig({build:{target:"esnext",rollupOptions:{input:{index:resolve(import.meta.dirname,'index.html'),sandbox:resolve(import.meta.dirname,'src/sandbox.js')},external:['add-on-sdk-document-sandbox','express-document-sdk','https://new.express.adobe.com/static/add-on-sdk/sdk.js'],output:{entryFileNames:'[name].js',format:'es'}}}});
