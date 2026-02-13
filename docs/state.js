/**
 * State module: undo history, config cache, and shared app state.
 * Centralizes all mutable state so other modules can read/write it
 * without circular dependencies.
 */

import { loadConfig } from './gl_utils.js';

// ─── Rule History (undo stack) ───────────────────────────────────────────────

export class RuleHistory {
    constructor(maxSize = 200) {
        this.stack = []; // Array of {rule: number[], seed: number}
        this.maxSize = maxSize;
    }

    push(rule, seed) {
        this.stack.push({ rule: Array.from(rule), seed });
        if (this.stack.length > this.maxSize) {
            this.stack.shift();
        }
    }

    pop() {
        if (this.stack.length <= 1) return null;
        this.stack.pop(); // remove current
        return this.stack[this.stack.length - 1]; // return previous (now current)
    }

    current() {
        return this.stack.length > 0 ? this.stack[this.stack.length - 1] : null;
    }

    get length() {
        return this.stack.length;
    }
}

// ─── Config cache (for preset hover preview) ────────────────────────────────

const configCache = new Map();

export async function fetchConfig(name) {
    if (configCache.has(name)) return configCache.get(name);
    const path = `physics_configs/${name}.json`;
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${path}`);
    const data = await response.json();
    const config = loadConfig(data);
    configCache.set(name, config);
    return config;
}

// ─── Shared app state ────────────────────────────────────────────────────────

export function createAppState() {
    return {
        mouseMode: 'select',   // 'select' or 'draw'
        mouseDown: false,
        mousePos: { x: 0, y: 0 },
        prevMousePos: { x: 0, y: 0 },
        paused: false,

        // Dropdown/preview state
        dropdownOpen: false,
        previewActive: false,
        previewBaseConfig: null,
        previewGeneration: 0,

        // Fancy camera state
        fancyCamera: false,
        camera: { posX: 0, posY: 0, zoom: 1.0 },
        cameraKeys: { w: false, a: false, s: false, d: false, q: false, e: false },
    };
}
