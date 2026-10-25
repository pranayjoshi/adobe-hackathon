import sandbox from 'add-on-sdk-document-sandbox';
import {editor,constants,fonts} from 'express-document-sdk';
import {createAdapter} from './adapter.js';
sandbox.instance.runtime.exposeApi(createAdapter({editor,constants,fonts}));
