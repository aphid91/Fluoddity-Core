/**
 * Logger module: on-screen error display + console output.
 * Provides a clean API for surfacing shader compilation errors
 * and other diagnostics to the user.
 */

export function createLogger(errorDisplayEl) {
    return {
        error(msg) {
            errorDisplayEl.textContent = msg;
            errorDisplayEl.style.display = 'block';
            console.error(msg);
        },
        warn(msg) {
            console.warn(msg);
        },
        clear() {
            errorDisplayEl.textContent = '';
            errorDisplayEl.style.display = 'none';
        },
    };
}
