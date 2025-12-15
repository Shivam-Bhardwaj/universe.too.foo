/**
 * Runtime environment detection and requirements.
 * Phase 0.2: Define supported runtime environment for "one path" architecture.
 */

export interface RuntimeInfo {
    /** WebGPU is available and functional */
    webgpu: boolean;
    /** WebGL2 is available (fallback) */
    webgl2: boolean;
    /** User agent string */
    userAgent: string;
    /** Browser name (best guess) */
    browser: string;
    /** Browser version (best guess, may be null) */
    version: string | null;
    /** Whether this runtime meets minimum requirements */
    meetsRequirements: boolean;
    /** Error message if requirements not met */
    errorMessage: string | null;
}

/**
 * Minimum requirements for production use:
 * - WebGPU: Chrome 113+, Edge 113+, or Firefox 110+ (experimental flag)
 * - WebGL2: Fallback for older browsers (not recommended for production)
 */
const MIN_CHROME_VERSION = 113;
const MIN_EDGE_VERSION = 113;
const MIN_FIREFOX_VERSION = 110; // Requires experimental flag
const MIN_SAFARI_VERSION = 18; // Safari 18+ (macOS Sequoia/iOS 18+)

/**
 * Detect runtime environment and check requirements.
 */
export function detectRuntime(): RuntimeInfo {
    const ua = navigator.userAgent;
    const hasWebGpu = 'gpu' in navigator;
    
    // Check WebGL2 support
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    const hasWebGl2 = gl !== null;

    // Parse browser and version
    let browser = 'unknown';
    let version: string | null = null;

    // Chrome/Chromium
    const chromeMatch = ua.match(/Chrome\/(\d+)/);
    if (chromeMatch && !ua.includes('Edg')) {
        browser = 'chrome';
        version = chromeMatch[1];
    }
    // Edge
    else if (ua.includes('Edg')) {
        const edgeMatch = ua.match(/Edg\/(\d+)/);
        browser = 'edge';
        version = edgeMatch ? edgeMatch[1] : null;
    }
    // Firefox
    else if (ua.includes('Firefox')) {
        const firefoxMatch = ua.match(/Firefox\/(\d+)/);
        browser = 'firefox';
        version = firefoxMatch ? firefoxMatch[1] : null;
    }
    // Safari
    else if (ua.includes('Safari') && !ua.includes('Chrome')) {
        // Safari version is tricky; check for Version/XX.X
        const safariMatch = ua.match(/Version\/(\d+)/);
        browser = 'safari';
        version = safariMatch ? safariMatch[1] : null;
    }

    // Check if requirements are met
    let meetsRequirements = false;
    let errorMessage: string | null = null;

    if (hasWebGpu) {
        // WebGPU is available - check browser version
        const versionNum = version ? parseInt(version, 10) : null;
        
        if (browser === 'chrome' && versionNum !== null && versionNum >= MIN_CHROME_VERSION) {
            meetsRequirements = true;
        } else if (browser === 'edge' && versionNum !== null && versionNum >= MIN_EDGE_VERSION) {
            meetsRequirements = true;
        } else if (browser === 'firefox' && versionNum !== null && versionNum >= MIN_FIREFOX_VERSION) {
            // Firefox WebGPU requires experimental flag
            meetsRequirements = true;
            errorMessage = 'Firefox WebGPU requires experimental flag: dom.webgpu.enabled = true';
        } else if (browser === 'safari' && versionNum !== null && versionNum >= MIN_SAFARI_VERSION) {
            meetsRequirements = true;
        } else if (hasWebGpu) {
            // WebGPU detected but version unknown or too old
            meetsRequirements = false;
            errorMessage = `WebGPU detected but browser version may be too old. ` +
                `Required: Chrome ${MIN_CHROME_VERSION}+, Edge ${MIN_EDGE_VERSION}+, ` +
                `Firefox ${MIN_FIREFOX_VERSION}+ (experimental), Safari ${MIN_SAFARI_VERSION}+`;
        }
    } else if (hasWebGl2) {
        // WebGL2 fallback - warn but allow
        meetsRequirements = true;
        errorMessage = 'WebGPU not available. Using WebGL2 fallback (reduced performance). ' +
            'For best experience, use Chrome 113+, Edge 113+, or Safari 18+.';
    } else {
        // No GPU support
        meetsRequirements = false;
        errorMessage = 'No GPU rendering support detected. ' +
            'WebGPU or WebGL2 is required. Please use a modern browser.';
    }

    return {
        webgpu: hasWebGpu,
        webgl2: hasWebGl2,
        userAgent: ua,
        browser,
        version,
        meetsRequirements,
        errorMessage,
    };
}

/**
 * Get a user-friendly error message for display in the UI.
 */
export function getRuntimeErrorMessage(info: RuntimeInfo): string {
    if (info.meetsRequirements) {
        if (info.errorMessage) {
            // Warning but still works
            return info.errorMessage;
        }
        return '';
    }
    
    // Hard failure
    return info.errorMessage || 'Unsupported browser. Please use Chrome 113+, Edge 113+, or Safari 18+.';
}
