import { UniverseClient } from './client';
import { InputHandler, LocalInputHandler } from './input';
import { HUD } from './hud';
import { fetchCell, fetchManifest, fetchPackIndex, parseCell } from './dataset';
import { LocalCamera } from './camera';
import { WebGpuSplatRenderer } from './webgpu_splats';
import { WebGlSplatRenderer } from './webgl_splats';
import { detectRuntime, getRuntimeErrorMessage } from './runtime';
import { LandmarksManager } from './landmarks';
import { determineRegime, ScaleRegime } from './scale_system';
import { ClientTimeController, J2000_JD } from './time_controller';
import { estimateStellarVelocity, propagateStarFull } from './stellar_motion';

type Dom = {
    video: HTMLCanvasElement;
    videoContainer: HTMLDivElement;
    loading: HTMLDivElement;
    clickPrompt: HTMLDivElement;
    status: HTMLDivElement;
    buffering: HTMLDivElement;
};

function resetVideoCanvas(dom: Dom): HTMLCanvasElement {
    // A canvas cannot switch context types (webgl/webgpu/2d). When switching modes,
    // we must replace the canvas element to get a fresh context.
    const old = dom.video;
    const parent = dom.videoContainer;

    const next = document.createElement('canvas');
    next.id = old.id || 'video';

    // Replace if possible; otherwise just append and remove old.
    if (old.parentElement === parent) {
        parent.replaceChild(next, old);
    } else {
        parent.appendChild(next);
        try {
            old.remove();
        } catch {
            // ignore
        }
    }

    dom.video = next;
    return next;
}

async function runDatasetMode(dom: Dom, params: URLSearchParams): Promise<boolean> {
    const video = resetVideoCanvas(dom);
    const { videoContainer, loading, clickPrompt, status, buffering } = dom;

    // Phase 0.2: Check runtime requirements
    const runtime = detectRuntime();
    console.log('[RUNTIME]', {
        browser: runtime.browser,
        version: runtime.version,
        webgpu: runtime.webgpu,
        webgl2: runtime.webgl2,
        meetsRequirements: runtime.meetsRequirements,
    });

    if (!runtime.meetsRequirements && !runtime.webgl2) {
        // Hard failure: no GPU support at all
        loading.classList.remove('hidden');
        loading.innerHTML = `
            <div style="color: #f00">Unsupported Browser</div>
            <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                ${getRuntimeErrorMessage(runtime)}
            </div>
            <div style="font-size: 12px; margin-top: 20px; opacity: 0.5">
                Required: Chrome 113+, Edge 113+, Safari 18+, or Firefox 110+ (experimental)
            </div>
        `;
        status.textContent = '● UNSUPPORTED';
        status.className = 'disconnected';
        return false;
    }

    // UI state
    status.textContent = '● DATASET MODE';
    status.className = 'connected';
    buffering.classList.add('hidden');
    loading.classList.remove('hidden');

    // HUD
    const hud = new HUD();

    // Debug overlay (query-param gated): `?debugOverlay=1` (or `?debug=1`)
    const debugOverlayEnabled =
        params.get('debugOverlay') === '1' ||
        params.get('debugOverlay')?.toLowerCase() === 'true' ||
        params.get('debug') === '1' ||
        params.get('debug')?.toLowerCase() === 'true';

    const debugOverlayEl: HTMLDivElement | null = debugOverlayEnabled
        ? (() => {
              const el = document.createElement('div');
              el.id = 'debug-overlay';
              el.style.position = 'absolute';
              el.style.left = '20px';
              el.style.bottom = '20px';
              el.style.padding = '10px 12px';
              el.style.borderRadius = '10px';
              el.style.border = '1px solid rgba(0, 170, 255, 0.25)';
              el.style.background = 'rgba(0, 0, 0, 0.55)';
              el.style.color = 'rgba(207, 239, 255, 0.95)';
              el.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace';
              el.style.fontSize = '12px';
              el.style.lineHeight = '1.35';
              el.style.pointerEvents = 'none';
              el.style.zIndex = '90';
              el.textContent = 'debug overlay…';
              document.getElementById('overlay')?.appendChild(el);
              return el;
          })()
        : null;

    // Canvas sizing
    const dpr = window.devicePixelRatio || 1;
    const resizeCanvas = () => {
        const rect = videoContainer.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (video.width !== w || video.height !== h) {
            video.width = w;
            video.height = h;
        }
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Local camera + input
    const camera = new LocalCamera();
    // Initialize floating origin at the camera start position for precision.
    camera.origin = { ...camera.position };
    const input = new LocalInputHandler(camera);
    const flightControls = input.getFlightControls();

    // Time controller for ±100,000 year time travel
    const timeController = new ClientTimeController(J2000_JD);

    // Crosshair always visible for targeting
    const crosshairEl = document.getElementById('crosshair');
    if (crosshairEl) {
        crosshairEl.classList.add('always-visible');
    }

    // Make canvas focusable for keyboard input
    video.setAttribute('tabindex', '0');
    video.style.outline = 'none';

    // Targeting state (updated later once solar bodies exist)
    type TargetableBody = { name: string; pos: [number, number, number]; radius_m: number };
    let targetBody: TargetableBody | null = null;
    let pickBodies: TargetableBody[] = [];

    const teleportToBodyCenter = (b: TargetableBody) => {
        // Teleport to a sensible viewing distance outside the body (not to its center).
        const viewDist = Math.max(1e7, b.radius_m * 20);
        const toTarget = {
            x: b.pos[0] - camera.position.x,
            y: b.pos[1] - camera.position.y,
            z: b.pos[2] - camera.position.z,
        };
        const len = Math.sqrt(toTarget.x * toTarget.x + toTarget.y * toTarget.y + toTarget.z * toTarget.z);
        const dir =
            len > 1e-6
                ? { x: toTarget.x / len, y: toTarget.y / len, z: toTarget.z / len }
                : { x: 0, y: 0, z: -1 };

        camera.setPosition({
            x: b.pos[0] - dir.x * viewDist,
            y: b.pos[1] - dir.y * viewDist,
            z: b.pos[2] - dir.z * viewDist,
        });
        camera.lookAt({ x: b.pos[0], y: b.pos[1], z: b.pos[2] });

        // Immediately rebase so GPU-relative positions stay stable after a teleport.
        camera.origin = { ...camera.position };
        recomputeRelativePositions();
        uploadSplats();
    };

    const pickBodyAt = (clientX: number, clientY: number): TargetableBody | null => {
        if (pickBodies.length === 0) return null;
        const rect = video.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;

        const u = camera.cameraUniform(video.width / video.height);
        const viewProj = u.subarray(32, 48);

        const THRESH_PX = 42;
        const THRESH2 = THRESH_PX * THRESH_PX;

        let best: { b: TargetableBody; d2: number } | null = null;
        for (const b of pickBodies) {
            const rx = b.pos[0] - camera.origin.x;
            const ry = b.pos[1] - camera.origin.y;
            const rz = b.pos[2] - camera.origin.z;

            const clipX = viewProj[0] * rx + viewProj[4] * ry + viewProj[8] * rz + viewProj[12];
            const clipY = viewProj[1] * rx + viewProj[5] * ry + viewProj[9] * rz + viewProj[13];
            const clipW = viewProj[3] * rx + viewProj[7] * ry + viewProj[11] * rz + viewProj[15];
            if (clipW <= 0) continue;

            const ndcX = clipX / clipW;
            const ndcY = clipY / clipW;
            const sx = (ndcX * 0.5 + 0.5) * rect.width;
            const sy = (1.0 - (ndcY * 0.5 + 0.5)) * rect.height;

            const dx = sx - x;
            const dy = sy - y;
            const d2 = dx * dx + dy * dy;
            if (d2 > THRESH2) continue;

            if (!best || d2 < best.d2) best = { b, d2 };
        }
        return best ? best.b : null;
    };

    // ---------------------------------------------------------------------
    // Unified Flight Controls: mouse-steer + scroll-throttle (desktop)
    //                         1-finger drag + pinch throttle (mobile)
    // ---------------------------------------------------------------------
    clickPrompt.classList.add('hidden'); // No pointer lock needed

    // Desktop:
    // - Flight mode: click-drag to steer (mouse deltas), click to jump
    // - Orbit mode: click-drag to orbit
    let orbitDragActive = false;
    let orbitLastX = 0;
    let orbitLastY = 0;
    let flightDragActive = false;
    let flightDragMoved2 = 0;
    let lastClickTime = 0;

    video.addEventListener('mousemove', (e) => {
        const mode = flightControls.getMode();

        if (mode === 'orbitFocus' && orbitDragActive) {
            const rect = video.getBoundingClientRect();
            const dx = e.clientX - orbitLastX;
            const dy = e.clientY - orbitLastY;
            orbitLastX = e.clientX;
            orbitLastY = e.clientY;
            flightControls.updateOrbitFromDrag(dx, dy, rect.width, rect.height);
            return;
        }

        if (mode === 'flight' && flightDragActive) {
            // Delta-based steering (only while dragging): avoids edge/continuous-rotation weirdness.
            flightControls.updateSteerFromMouse(e.movementX, e.movementY);
            flightDragMoved2 += e.movementX * e.movementX + e.movementY * e.movementY;
        }
    });

    video.addEventListener('mouseup', (e) => {
        orbitDragActive = false;

        if (!flightDragActive) return;
        flightDragActive = false;

        // Treat short drags as clicks (jump-to-target).
        const CLICK_DIST2 = 8 * 8;
        if (flightDragMoved2 > CLICK_DIST2) return;

        const picked = pickBodyAt(e.clientX, e.clientY);
        if (!picked) return;

        const now = performance.now();
        const isDoubleClick = now - lastClickTime < 300;
        lastClickTime = now;

        if (isDoubleClick) {
            // Double-click: teleport to center (instant)
            teleportToBodyCenter(picked);
        } else {
            // Single click: start jump to target
            flightControls.startJump(
                { x: picked.pos[0], y: picked.pos[1], z: picked.pos[2] },
                picked.name,
                picked.radius_m,
            );
        }
    });
    video.addEventListener('mouseleave', () => {
        orbitDragActive = false;
        flightDragActive = false;
    });

    // Desktop: scroll wheel controls throttle (flight) or zoom (orbit)
    video.addEventListener(
        'wheel',
        (e) => {
            e.preventDefault();
            video.focus();
            const mode = flightControls.getMode();
            if (mode === 'orbitFocus') flightControls.updateOrbitRadius(e.deltaY, false);
            else flightControls.updateThrottleFromWheel(e.deltaY);
        },
        { passive: false },
    );

    // Mobile: 1-finger drag steers, pinch throttles
    let touchSteerId: number | null = null;
    let touchLastX = 0;
    let touchLastY = 0;
    let touchStartX = 0;
    let touchStartY = 0;
    let touchMoved2 = 0;
    let touchDownTimeMs = 0;
    let pinchStartDistance = 0;
    let pinchLastDistance = 0;

    const getTouchDistance = (touches: TouchList): number => {
        if (touches.length < 2) return 0;
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    };

    video.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touches = Array.from(e.touches);

        if (touches.length === 1) {
            // Single touch: start steer
            touchSteerId = touches[0].identifier;
            touchStartX = touches[0].clientX;
            touchStartY = touches[0].clientY;
            touchLastX = touches[0].clientX;
            touchLastY = touches[0].clientY;
            touchMoved2 = 0;
            touchDownTimeMs = performance.now();
            video.focus();
        } else if (touches.length === 2) {
            // Two touches: start pinch
            pinchStartDistance = getTouchDistance(e.touches);
            pinchLastDistance = pinchStartDistance;
            flightControls.updateThrottleFromPinch(pinchStartDistance, pinchStartDistance);
        }
    }, { passive: false });

    video.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touches = Array.from(e.touches);

        if (touches.length === 1 && touchSteerId === touches[0].identifier) {
            // Single touch: steer
            const dx = touches[0].clientX - touchLastX;
            const dy = touches[0].clientY - touchLastY;
            touchMoved2 += dx * dx + dy * dy;

            const mode = flightControls.getMode();
            if (mode === 'flight') {
                flightControls.updateSteerFromDrag(dx, dy);
            } else if (mode === 'orbitFocus') {
                flightControls.updateOrbitFromDrag(dx, dy, video.width, video.height);
            }

            touchLastX = touches[0].clientX;
            touchLastY = touches[0].clientY;
        } else if (touches.length === 2) {
            // Two touches: pinch throttle/zoom
            const distance = getTouchDistance(e.touches);
            const mode = flightControls.getMode();
            if (mode === 'flight') {
                flightControls.updateThrottleFromPinch(distance, pinchStartDistance);
            } else if (mode === 'orbitFocus') {
                const delta = (distance - pinchLastDistance) / Math.max(1, pinchLastDistance);
                flightControls.updateOrbitRadius(delta, true);
            }
            pinchLastDistance = distance;
        }
    }, { passive: false });

    video.addEventListener('touchend', (e) => {
        e.preventDefault();
        const touches = Array.from(e.touches);

        if (touches.length === 0) {
            // All touches ended
            if (touchSteerId !== null) {
                // Check if this was a tap (for orbit focus or teleport)
                const totalDx = touchLastX - touchStartX;
                const totalDy = touchLastY - touchStartY;
                const total2 = totalDx * totalDx + totalDy * totalDy;
                const dtMs = performance.now() - touchDownTimeMs;

                const TAP_DIST2 = 10 * 10;
                const TAP_TIME_MS = 260;
                const LONG_PRESS_MS = 350;

                if (total2 <= TAP_DIST2 && touchMoved2 <= TAP_DIST2) {
                    const picked = pickBodyAt(touchLastX, touchLastY);
                    if (picked) {
                        if (dtMs >= LONG_PRESS_MS) {
                            // Long press: teleport to center (instant)
                            teleportToBodyCenter(picked);
                        } else if (dtMs <= TAP_TIME_MS) {
                            // Tap: start jump to target
                            flightControls.startJump(
                                { x: picked.pos[0], y: picked.pos[1], z: picked.pos[2] },
                                picked.name,
                                picked.radius_m,
                            );
                        }
                    }
                }
            }
            touchSteerId = null;
            flightControls.resetPinch();
            pinchStartDistance = 0;
            pinchLastDistance = 0;
        } else if (touches.length === 1) {
            // One touch remains: switch to steer
            touchSteerId = touches[0].identifier;
            touchLastX = touches[0].clientX;
            touchLastY = touches[0].clientY;
            flightControls.resetPinch();
            pinchStartDistance = 0;
            pinchLastDistance = 0;
        }
    }, { passive: false });

    video.addEventListener('touchcancel', (e) => {
        e.preventDefault();
        touchSteerId = null;
        flightControls.resetPinch();
    }, { passive: false });

    // Desktop: mousedown begins steering drag (flight) or orbit drag (orbit mode).
    video.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        video.focus();

        const mode = flightControls.getMode();
        if (mode === 'orbitFocus') {
            // In orbit mode, click-drag rotates the orbit.
            orbitDragActive = true;
            orbitLastX = e.clientX;
            orbitLastY = e.clientY;
            return;
        }
        // In flight mode, mouse drag steers; mouseup will trigger click-to-jump if it was a short drag.
        flightDragActive = true;
        flightDragMoved2 = 0;
    });

    // Safety: never leave movement keys "stuck" if the tab loses focus.
    window.addEventListener('blur', () => {
        input.clearVirtualKeys();
        flightControls.resetThrottle();
        flightControls.cancelJump();
    });

    // Minimal mobile UI buttons
    const mobileMinUI = document.getElementById('mobile-min-ui');
    const exitOrbitBtn = mobileMinUI?.querySelector('[data-action="exitOrbit"]') as HTMLButtonElement | null;
    if (mobileMinUI) {
        mobileMinUI.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            const action = target.getAttribute('data-action');
            if (!action) return;

            switch (action) {
                case 'home':
                    flightControls.cancelJump();
                    camera.setPosition({ x: 0, y: 0, z: 1.5e11 });
                    camera.lookAt({ x: 0, y: 0, z: 0 });
                    if (flightControls.getMode() === 'orbitFocus') {
                        flightControls.exitOrbitFocus();
                    }
                    break;
                case 'reset':
                    flightControls.cancelJump();
                    camera.resetGain();
                    flightControls.resetThrottle();
                    break;
                case 'exitOrbit':
                    flightControls.exitOrbitFocus();
                    break;
            }
        });
    }

    // Renderer: prefer WebGPU, fall back to WebGL2 (still dataset-based; no video codecs)
    let rendererKind: 'webgpu' | 'webgl' = 'webgpu';
    let rendererWebGpu: WebGpuSplatRenderer | null = null;
    let rendererWebGl: WebGlSplatRenderer | null = null;

    try {
        rendererWebGpu = await WebGpuSplatRenderer.create(video);
        rendererKind = 'webgpu';
    } catch (e) {
        console.warn('[DATASET] WebGPU unavailable, trying WebGL2…', e);
        try {
            rendererWebGl = WebGlSplatRenderer.create(video);
            rendererKind = 'webgl';
        } catch (e2) {
            console.error(e2);
            loading.classList.remove('hidden');
            const runtimeMsg = getRuntimeErrorMessage(runtime);
            loading.innerHTML = `
                <div style="color: #f00">GPU Rendering Unavailable</div>
                <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                    ${runtimeMsg || 'Neither WebGPU nor WebGL2 is available in this browser.'}
                </div>
                <div style="font-size: 12px; margin-top: 20px; opacity: 0.5">
                    Browser: ${runtime.browser} ${runtime.version || 'unknown'}
                </div>
            `;
            return false;
        }
    }

    const syncResize = () => {
        resizeCanvas();
        if (rendererKind === 'webgpu') rendererWebGpu!.resize(video.width, video.height);
        else rendererWebGl!.resize(video.width, video.height);
    };
    window.addEventListener('resize', syncResize);
    syncResize();

    // Load dataset manifest + a subset of cells
    loading.innerHTML = `
        <div class="spinner"></div>
        <div>Loading dataset…</div>
        <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
            Fetching /universe/index.json
        </div>
    `;

    let manifest;
    try {
        manifest = await fetchManifest('/universe');
    } catch (e) {
        console.error(e);
        loading.classList.remove('hidden');
        loading.innerHTML = `
            <div style="color: #f00">Dataset Missing</div>
            <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                Could not fetch <code>/universe/index.json</code>.
            </div>
        `;
        return false;
    }

    // Compute dataset radius from manifest (outer radius of outermost shell)
    const computeDatasetRadius = (): number => {
        const cfg = manifest.config;
        let maxL = 0;
        for (const cell of manifest.cells) {
            if (cell.id.l > maxL) maxL = cell.id.l;
        }
        // rOuter(l) = r_min * log_base^(l+1)
        return cfg.r_min * Math.pow(cfg.log_base, maxL + 1);
    };
    const datasetRadius = computeDatasetRadius();

    // Initialize landmarks system (built-in + optional ML landmarks)
    const landmarks = new LandmarksManager();
    landmarks.loadMLLandmarks('/universe').catch(() => {
        // Optional: ML landmarks not available, no problem
    });
    console.log(`[LANDMARKS] Initialized with ${landmarks.getAll().length} built-in landmarks`);

    const cellsParam = params.get('cells');
    const splatsParam = params.get('splats');

    // For small datasets, load all cells by default (to avoid a \"dark\" view due to sparse cells).
    const totalCellsInManifest = manifest.cells.length;
    const totalBytes = (manifest as any).total_size_bytes as number | undefined;
    const smallDataset =
        typeof totalBytes === 'number' && totalBytes > 0 ? totalBytes <= 20_000_000 : totalCellsInManifest <= 2000;

    const defaultMaxCells = smallDataset ? totalCellsInManifest : 1000;
    const maxCells = Math.max(
        1,
        Math.min(
            totalCellsInManifest,
            parseInt(cellsParam ?? '', 10) || defaultMaxCells,
        ),
    );

    const defaultMaxSplats = Math.min(200_000, manifest.total_splats || 200_000);
    const maxSplats = Math.max(1, parseInt(splatsParam ?? '', 10) || defaultMaxSplats);

    // For large manifests, don't load the first N entries (which is effectively arbitrary).
    // Instead, prioritize cells closest to the current camera position.
    const selectEntriesByDistance = (pos: { x: number; y: number; z: number }) => {
        const cfg = manifest.config;
        const thetaStep = (Math.PI * 2) / cfg.n_theta;
        const phiStep = Math.PI / cfg.n_phi;

        const rInner = (l: number) => cfg.r_min * Math.pow(cfg.log_base, l);
        const rOuter = (l: number) => cfg.r_min * Math.pow(cfg.log_base, l + 1);

        const cellCenter = (id: { l: number; theta: number; phi: number }) => {
            const ri = rInner(id.l);
            const ro = rOuter(id.l);
            const r = Math.sqrt(ri * ro); // geometric mean (log shells)
            const theta = -Math.PI + (id.theta + 0.5) * thetaStep;
            const phi = (id.phi + 0.5) * phiStep;
            const sinPhi = Math.sin(phi);
            return {
                x: r * sinPhi * Math.cos(theta),
                y: r * sinPhi * Math.sin(theta),
                z: r * Math.cos(phi),
            };
        };

        const scored: Array<{ e: (typeof manifest.cells)[number]; d2: number }> = manifest.cells.map((e) => {
            const c = cellCenter(e.id);
            const dx = c.x - pos.x;
            const dy = c.y - pos.y;
            const dz = c.z - pos.z;
            return { e, d2: dx * dx + dy * dy + dz * dz };
        });

        scored.sort((a, b) => a.d2 - b.d2);

        const out: (typeof manifest.cells) = [];
        let splats = 0;
        for (const s of scored) {
            if (out.length >= maxCells) break;
            const n = s.e.splat_count || 0;
            if (n <= 0) continue;
            if (splats + n > maxSplats) continue;
            out.push(s.e);
            splats += n;
        }
        return out;
    };

    const selectEntriesByView = (pos: { x: number; y: number; z: number }, forward: { x: number; y: number; z: number }) => {
        const cfg = manifest.config;
        const thetaStep = (Math.PI * 2) / cfg.n_theta;
        const phiStep = Math.PI / cfg.n_phi;

        const rInner = (l: number) => cfg.r_min * Math.pow(cfg.log_base, l);
        const rOuter = (l: number) => cfg.r_min * Math.pow(cfg.log_base, l + 1);

        const cellCenter = (id: { l: number; theta: number; phi: number }) => {
            const ri = rInner(id.l);
            const ro = rOuter(id.l);
            const r = Math.sqrt(ri * ro); // geometric mean (log shells)
            const theta = -Math.PI + (id.theta + 0.5) * thetaStep;
            const phi = (id.phi + 0.5) * phiStep;
            const sinPhi = Math.sin(phi);
            return {
                x: r * sinPhi * Math.cos(theta),
                y: r * sinPhi * Math.sin(theta),
                z: r * Math.cos(phi),
            };
        };

        // Expand the selection cone a bit so turning doesn't constantly thrash loads.
        const margin = 1.4;
        const cosHalf = Math.cos((camera.fovY * 0.5) * margin);

        const candidates: Array<{ e: (typeof manifest.cells)[number]; dot: number; d2: number }> = [];
        for (const e of manifest.cells) {
            const c = cellCenter(e.id);
            const dx = c.x - pos.x;
            const dy = c.y - pos.y;
            const dz = c.z - pos.z;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= 1e-12) continue;
            const invD = 1.0 / Math.sqrt(d2);
            const ux = dx * invD;
            const uy = dy * invD;
            const uz = dz * invD;
            const dot = forward.x * ux + forward.y * uy + forward.z * uz;
            if (dot < cosHalf) continue;
            candidates.push({ e, dot, d2 });
        }

        // Prefer center-of-view and denser cells.
        candidates.sort((a, b) => {
            const dc = b.dot - a.dot;
            if (Math.abs(dc) > 1e-6) return dc > 0 ? 1 : -1;
            const sc = (b.e.splat_count || 0) - (a.e.splat_count || 0);
            if (sc !== 0) return sc;
            return a.d2 - b.d2;
        });

        const out: (typeof manifest.cells) = [];
        let splats = 0;
        for (const c of candidates) {
            if (out.length >= maxCells) break;
            const n = c.e.splat_count || 0;
            if (n <= 0) continue;
            if (splats + n > maxSplats) continue;
            out.push(c.e);
            splats += n;
        }
        return out;
    };

    // -------------------------------------------------------------------------
    // Solar system overlay (procedural)
    // Dataset mode is primarily the starfield dataset. We also add a tiny set of
    // procedural "anchor" bodies so the scene doesn't feel random and you can
    // actually see the Sun from solar-system distances.
    // -------------------------------------------------------------------------
    const AU_M = 1.496e11;
    const formatDistance = (meters: number): string => {
        const m = Math.max(0, meters);
        const au = m / AU_M;
        // Use AU for large distances, km for near distances to avoid misleading “0.00 AU”.
        if (au >= 0.01) return `${au.toFixed(2)} AU`;
        const km = Math.round(m / 1000);
        return `${km.toLocaleString()} km`;
    };
    type SolarBody = {
        name: string;
        pos: [number, number, number];
        radius_m: number;
        color: [number, number, number];
        opacity: number;
    };

    // NOTE: Planet positions are *approximate* (semi-major axis along +X).
    // This is just a visual reference layer; later we can drive this from ephemeris.
    const solarBodies: SolarBody[] = [
        // Make the Sun a bit larger than physical so it's obvious at 1 AU.
        { name: 'Sun', pos: [0, 0, 0], radius_m: 6.9634e8 * 4.0, color: [1.0, 0.95, 0.85], opacity: 1.0 },
        { name: 'Mercury', pos: [0.39 * AU_M, 0, 0], radius_m: 2.4397e6, color: [0.70, 0.70, 0.70], opacity: 1.0 },
        { name: 'Venus', pos: [0.72 * AU_M, 0, 0], radius_m: 6.0518e6, color: [0.95, 0.88, 0.70], opacity: 1.0 },
        { name: 'Earth', pos: [1.0 * AU_M, 0, 0], radius_m: 6.371e6, color: [0.20, 0.55, 1.00], opacity: 1.0 },
        { name: 'Mars', pos: [1.52 * AU_M, 0, 0], radius_m: 3.3895e6, color: [1.00, 0.45, 0.25], opacity: 1.0 },
        { name: 'Jupiter', pos: [5.2 * AU_M, 0, 0], radius_m: 6.9911e7, color: [0.92, 0.84, 0.72], opacity: 1.0 },
        { name: 'Saturn', pos: [9.5 * AU_M, 0, 0], radius_m: 5.8232e7, color: [0.92, 0.86, 0.68], opacity: 1.0 },
        { name: 'Uranus', pos: [19.2 * AU_M, 0, 0], radius_m: 2.5362e7, color: [0.62, 0.86, 0.92], opacity: 1.0 },
        { name: 'Neptune', pos: [30.0 * AU_M, 0, 0], radius_m: 2.4622e7, color: [0.34, 0.55, 0.98], opacity: 1.0 },

        // Spacecraft (scaled up for visibility - actual size ~10m)
        { name: 'Voyager 1', pos: [1.7e13, 5.2e12, 2.9e12], radius_m: 1e7, color: [0.9, 0.9, 1.0], opacity: 0.9 },
        { name: 'Voyager 2', pos: [1.4e13, -1.1e13, -8.3e12], radius_m: 1e7, color: [0.9, 0.9, 1.0], opacity: 0.9 },
        { name: 'New Horizons', pos: [-8.7e12, -2.0e12, -3.0e12], radius_m: 1e7, color: [1.0, 1.0, 0.9], opacity: 0.85 },
        { name: 'JWST', pos: [1.511e11, 0, 0], radius_m: 1e6, color: [1.0, 0.85, 0.6], opacity: 0.8 },
        { name: 'Parker', pos: [6.0e10, 0, 0], radius_m: 5e5, color: [1.0, 0.5, 0.2], opacity: 0.75 },
    ];

    const solarSplats = solarBodies.length;
    // Enable picking/clicking of these anchor bodies
    pickBodies = solarBodies;

    // -------------------------------------------------------------------------
    // Navigation UI: compass + labels for a few anchor bodies
    // -------------------------------------------------------------------------
    const locationContextEl = document.getElementById('location-context');
    const navDistSunEl = document.getElementById('dist-sun');
    const navSpeedEl = document.getElementById('speed');
    const navAnglesEl = document.getElementById('angles');
    const navModeEl = document.getElementById('control-mode');
    const navTargetEl = document.getElementById('target');

    const labelsRoot = document.getElementById('labels') as HTMLDivElement | null;
    const labelNames = new Set(['Sun', 'Earth', 'Mars', 'Voyager 1', 'JWST']);
    const labelEls: Array<{ body: SolarBody; el: HTMLDivElement; distEl: HTMLSpanElement }> = [];

    if (labelsRoot) {
        labelsRoot.innerHTML = '';
        for (const b of solarBodies) {
            if (!labelNames.has(b.name)) continue;
            const el = document.createElement('div');
            el.className = 'world-label hidden';
            el.innerHTML = `<span class="name">${b.name}</span><span class="dist">--</span>`;
            labelsRoot.appendChild(el);
            const distEl = el.querySelector('.dist') as HTMLSpanElement;
            labelEls.push({ body: b, el, distEl });
        }
    }

    const compassCanvas = document.getElementById('compass') as HTMLCanvasElement | null;
    const compassCtx = compassCanvas ? (compassCanvas.getContext('2d') as CanvasRenderingContext2D | null) : null;

    // -------------------------------------------------------------------------
    // Time Controls
    // -------------------------------------------------------------------------
    const timeOffsetEl = document.getElementById('time-offset');
    const timeSlider = document.getElementById('time-slider') as HTMLInputElement | null;
    const timePauseBtn = document.getElementById('time-pause-btn');
    const timeResetBtn = document.getElementById('time-reset');

    // Wire up time control buttons
    document.getElementById('time-back-1000')?.addEventListener('click', () => timeController.addYears(-1000));
    document.getElementById('time-back-100')?.addEventListener('click', () => timeController.addYears(-100));
    document.getElementById('time-fwd-100')?.addEventListener('click', () => timeController.addYears(100));
    document.getElementById('time-fwd-1000')?.addEventListener('click', () => timeController.addYears(1000));
    document.getElementById('time-slower')?.addEventListener('click', () => timeController.multiplyRate(0.5));
    document.getElementById('time-faster')?.addEventListener('click', () => timeController.multiplyRate(2.0));

    timeResetBtn?.addEventListener('click', () => {
        timeController.resetToJ2000();
        if (timeSlider) timeSlider.value = '0';
    });

    timePauseBtn?.addEventListener('click', () => {
        timeController.togglePause();
        if (timePauseBtn) timePauseBtn.textContent = timeController.isPaused() ? 'PLAY' : 'PAUSE';
    });

    // Time slider - scrub through ±100,000 years
    timeSlider?.addEventListener('input', (e) => {
        const years = parseFloat((e.target as HTMLInputElement).value);
        const jd = J2000_JD + (years * 365.25);
        timeController.setJulianDate(jd);
        timeController.setPaused(true);  // Pause when scrubbing
        if (timePauseBtn) timePauseBtn.textContent = 'PLAY';
    });

    // Update time offset display
    timeController.addListener(() => {
        if (timeOffsetEl) {
            timeOffsetEl.textContent = `J2000 ${timeController.getTimeOffsetString()}`;
        }

        // Update HUD time display
        const hudTimeEl = document.getElementById('hud-time');
        if (hudTimeEl) {
            hudTimeEl.textContent = timeController.toDateString();
        }

        // Mark buffers dirty so stars get repositioned with proper motion
        dirtyRebuild = true;
    });

    // -------------------------------------------------------------------------
    // Search/Find UI
    // -------------------------------------------------------------------------
    const searchInput = document.getElementById('search-input') as HTMLInputElement | null;
    const searchResults = document.getElementById('search-results');
    const searchCount = document.getElementById('search-count');

    const updateSearchResults = () => {
        if (!searchInput || !searchResults || !landmarks) return;

        const query = searchInput.value.trim().toLowerCase();

        if (query.length === 0) {
            searchResults.innerHTML = '';
            if (searchCount) searchCount.textContent = 'Type to search...';
            return;
        }

        const results = landmarks.search(query);

        if (searchCount) {
            searchCount.textContent = `${results.length} result${results.length === 1 ? '' : 's'}`;
        }

        // Show top 10 results
        const top = results.slice(0, 10);
        searchResults.innerHTML = top
            .map((lm) => {
                const distM = Math.sqrt(
                    (lm.pos_meters.x - camera.position.x) ** 2 +
                        (lm.pos_meters.y - camera.position.y) ** 2 +
                        (lm.pos_meters.z - camera.position.z) ** 2
                );
                const distStr = formatDistance(distM);

                return `
                <div class="search-result-item" data-landmark-id="${lm.id}">
                    <div>
                        <span class="search-result-name">${lm.name}</span>
                        <span class="search-result-type">${lm.kind}</span>
                    </div>
                    <div class="search-result-dist">${distStr} away</div>
                </div>
            `;
            })
            .join('');

        // Wire up click handlers for results
        searchResults.querySelectorAll('.search-result-item').forEach((el) => {
            el.addEventListener('click', () => {
                const landmarkId = el.getAttribute('data-landmark-id');
                if (!landmarkId) return;

                const landmark = landmarks.get(landmarkId);
                if (!landmark) return;

                // Jump to landmark
                flightControls.startJump(
                    landmark.pos_meters,
                    landmark.name,
                    landmark.radius_hint || 1e9
                );

                // Clear search
                searchInput.value = '';
                updateSearchResults();
            });
        });
    };

    searchInput?.addEventListener('input', updateSearchResults);

    // Initialize search count
    if (searchCount) {
        searchCount.textContent = 'Type to search...';
    }

    // Minimap orb (DOM/canvas)
    const navOrbEl = document.getElementById('nav-orb') as HTMLDivElement | null;
    const navOrbCanvas = document.getElementById('nav-orb-canvas') as HTMLCanvasElement | null;
    const navOrbCtx = navOrbCanvas ? (navOrbCanvas.getContext('2d') as CanvasRenderingContext2D | null) : null;

    const resizeCompass = () => {
        if (!compassCanvas || !compassCtx) return;
        const rect = compassCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (compassCanvas.width !== w || compassCanvas.height !== h) {
            compassCanvas.width = w;
            compassCanvas.height = h;
        }
    };
    window.addEventListener('resize', resizeCompass);
    resizeCompass();

    const resizeNavOrb = () => {
        if (!navOrbCanvas || !navOrbCtx) return;
        const rect = navOrbCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (navOrbCanvas.width !== w || navOrbCanvas.height !== h) {
            navOrbCanvas.width = w;
            navOrbCanvas.height = h;
        }
    };
    window.addEventListener('resize', resizeNavOrb);
    resizeNavOrb();

    if (navOrbEl) {
        navOrbEl.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            navOrbEl.classList.toggle('expanded');
            resizeNavOrb();
        });
    }

    // Warp streak canvas setup
    const warpCanvas = document.getElementById('warp-canvas') as HTMLCanvasElement | null;
    const warpCtx = warpCanvas ? (warpCanvas.getContext('2d') as CanvasRenderingContext2D | null) : null;

    const resizeWarpCanvas = () => {
        if (!warpCanvas || !warpCtx) return;
        const rect = warpCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (warpCanvas.width !== w || warpCanvas.height !== h) {
            warpCanvas.width = w;
            warpCanvas.height = h;
        }
    };
    window.addEventListener('resize', resizeWarpCanvas);
    resizeWarpCanvas();

    // Warp streak particles
    type WarpStreak = { x: number; y: number; z: number; speed: number; brightness: number };
    const warpStreaks: WarpStreak[] = [];
    const MAX_STREAKS = 150;
    for (let i = 0; i < MAX_STREAKS; i++) {
        warpStreaks.push({
            x: (Math.random() - 0.5) * 2,
            y: (Math.random() - 0.5) * 2,
            z: Math.random(),
            speed: 0.5 + Math.random() * 0.5,
            brightness: 0.3 + Math.random() * 0.7,
        });
    }

    const drawWarpStreaks = (intensity: number) => {
        if (!warpCanvas || !warpCtx) return;
        const ctx = warpCtx;
        const w = warpCanvas.width;
        const h = warpCanvas.height;

        // Clear with fade effect for motion blur
        ctx.fillStyle = `rgba(0, 0, 0, ${0.3 + intensity * 0.4})`;
        ctx.fillRect(0, 0, w, h);

        if (intensity <= 0.01) {
            warpCanvas.classList.add('hidden');
            return;
        }
        warpCanvas.classList.remove('hidden');

        const cx = w * 0.5;
        const cy = h * 0.5;

        // Update and draw streaks
        ctx.lineCap = 'round';
        for (const streak of warpStreaks) {
            // Move streak toward camera (z decreases)
            streak.z -= intensity * streak.speed * 0.08;
            if (streak.z <= 0.05) {
                // Reset to far distance
                streak.z = 1.0;
                streak.x = (Math.random() - 0.5) * 2;
                streak.y = (Math.random() - 0.5) * 2;
                streak.speed = 0.5 + Math.random() * 0.5;
                streak.brightness = 0.3 + Math.random() * 0.7;
            }

            // Project to screen (simple perspective)
            const scale = 1 / streak.z;
            const sx = cx + streak.x * scale * cx;
            const sy = cy + streak.y * scale * cy;

            // Only draw if on screen
            if (sx < 0 || sx > w || sy < 0 || sy > h) continue;

            // Streak length based on speed and intensity
            const prevZ = streak.z + intensity * streak.speed * 0.08;
            const prevScale = 1 / prevZ;
            const prevSx = cx + streak.x * prevScale * cx;
            const prevSy = cy + streak.y * prevScale * cy;

            // Color: blue-white streaks
            const alpha = streak.brightness * intensity * Math.min(1, (1 - streak.z) * 3);
            ctx.strokeStyle = `rgba(150, 200, 255, ${alpha.toFixed(2)})`;
            ctx.lineWidth = Math.max(1, 2 * intensity * (1 - streak.z));

            ctx.beginPath();
            ctx.moveTo(prevSx, prevSy);
            ctx.lineTo(sx, sy);
            ctx.stroke();
        }
    };

    const dot3 = (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) =>
        a.x * b.x + a.y * b.y + a.z * b.z;

    const mat4Mul = (a: Float32Array, b: Float32Array): Float32Array => {
        // Column-major 4x4 multiply: out = a * b
        const out = new Float32Array(16);
        for (let c = 0; c < 4; c++) {
            for (let r = 0; r < 4; r++) {
                out[c * 4 + r] =
                    a[0 * 4 + r] * b[c * 4 + 0] +
                    a[1 * 4 + r] * b[c * 4 + 1] +
                    a[2 * 4 + r] * b[c * 4 + 2] +
                    a[3 * 4 + r] * b[c * 4 + 3];
            }
        }
        return out;
    };

    // 3D sphere rotation state (for expanded minimap)
    let sphereRotX = 0;
    let sphereRotY = 0;
    let sphereDragActive = false;
    let sphereDragStartX = 0;
    let sphereDragStartY = 0;

    const drawNavOrb = () => {
        if (!navOrbCanvas || !navOrbCtx) return;
        const ctx = navOrbCtx;
        const w = navOrbCanvas.width;
        const h = navOrbCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w * 0.5;
        const cy = h * 0.5;
        const displayRadius = Math.min(w, h) * 0.42;

        // Use dataset radius as the sphere's actual radius
        const sphereRadius = datasetRadius;

        // Normalize camera position relative to dataset center (0,0,0)
        const camX = camera.position.x / sphereRadius;
        const camY = camera.position.y / sphereRadius;
        const camZ = camera.position.z / sphereRadius;
        const camDist = Math.sqrt(camX * camX + camY * camY + camZ * camZ);
        const outside = camDist > 1.0;

        // Clamp to sphere surface if outside
        const clampedDist = Math.min(1.0, camDist);
        const normX = camX / Math.max(1e-6, camDist);
        const normY = camY / Math.max(1e-6, camDist);
        const normZ = camZ / Math.max(1e-6, camDist);

        // 3D projection: simple perspective from fixed camera
        const viewDist = 2.5; // camera distance from sphere center
        const isExpanded = navOrbEl?.classList.contains('expanded');

        // Rotation: expanded view is user-rotatable, collapsed view is fixed/world-oriented.
        const rotX = isExpanded ? sphereRotX : 0.35;
        const rotY = isExpanded ? sphereRotY : 0.75;

        // Rotate camera position for display
        const cosX = Math.cos(rotX);
        const sinX = Math.sin(rotX);
        const cosY = Math.cos(rotY);
        const sinY = Math.sin(rotY);

        const rotate3D = (x: number, y: number, z: number) => {
            // Rotate around Y (azimuth)
            let x1 = x * cosY - z * sinY;
            let z1 = x * sinY + z * cosY;
            // Rotate around X (elevation)
            let y1 = y * cosX - z1 * sinX;
            let z2 = y * sinX + z1 * cosX;
            return { x: x1, y: y1, z: z2 };
        };

        // Project 3D point to 2D screen
        const project3D = (x: number, y: number, z: number) => {
            const rotated = rotate3D(x, y, z);
            const scale = viewDist / (viewDist - rotated.z);
            return {
                sx: cx + rotated.x * displayRadius * scale,
                sy: cy - rotated.y * displayRadius * scale,
                depth: rotated.z,
            };
        };

        ctx.save();

        // Draw sphere base (shaded circle)
        ctx.beginPath();
        ctx.arc(cx, cy, displayRadius, 0, Math.PI * 2);
        ctx.clip();

        // Sphere shading
        const g = ctx.createRadialGradient(cx - displayRadius * 0.25, cy - displayRadius * 0.25, displayRadius * 0.2, cx, cy, displayRadius);
        g.addColorStop(0, 'rgba(0, 170, 255, 0.18)');
        g.addColorStop(1, 'rgba(0, 0, 0, 0.05)');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);

        // Draw equator and axes (3D projected)
        ctx.strokeStyle = 'rgba(0, 170, 255, 0.12)';
        ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) * 0.01));

        // Equator circle
        const equatorSteps = 32;
        ctx.beginPath();
        for (let i = 0; i <= equatorSteps; i++) {
            const theta = (i / equatorSteps) * Math.PI * 2;
            const x = Math.cos(theta);
            const y = 0;
            const z = Math.sin(theta);
            const proj = project3D(x, y, z);
            if (i === 0) ctx.moveTo(proj.sx, proj.sy);
            else ctx.lineTo(proj.sx, proj.sy);
        }
        ctx.stroke();

        // Axes
        const axes = [
            { x: 1, y: 0, z: 0, color: 'rgba(255, 90, 90, 0.3)' },
            { x: 0, y: 1, z: 0, color: 'rgba(98, 255, 98, 0.3)' },
            { x: 0, y: 0, z: 1, color: 'rgba(90, 125, 255, 0.3)' },
        ];
        for (const axis of axes) {
            const proj = project3D(axis.x, axis.y, axis.z);
            if (proj.depth > -0.5) {
                ctx.strokeStyle = axis.color;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(proj.sx, proj.sy);
                ctx.stroke();
            }
        }

        // Draw anchor bodies (projected)
        for (const b of solarBodies) {
            const px = b.pos[0] / sphereRadius;
            const py = b.pos[1] / sphereRadius;
            const pz = b.pos[2] / sphereRadius;
            const proj = project3D(px, py, pz);

            if (proj.depth > -0.8) {
                const alpha = 0.3 + 0.4 * ((proj.depth + 1) * 0.5);
                const r = 1.5 + alpha * 1.5;
                ctx.beginPath();
                ctx.arc(proj.sx, proj.sy, r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${Math.floor(b.color[0] * 255)}, ${Math.floor(b.color[1] * 255)}, ${Math.floor(
                    b.color[2] * 255,
                )}, ${alpha})`;
                ctx.fill();
            }
        }

        // Center marker + camera position vector
        const camProj = project3D(normX * clampedDist, normY * clampedDist, normZ * clampedDist);
        const camAlpha = 0.6 + 0.4 * ((camProj.depth + 1) * 0.5);

        ctx.beginPath();
        ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(camProj.sx, camProj.sy);
        ctx.strokeStyle = outside ? 'rgba(255, 90, 90, 0.22)' : 'rgba(255, 255, 255, 0.16)';
        ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) * 0.006));
        ctx.stroke();

        ctx.shadowColor = 'rgba(255,255,255,0.5)';
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(camProj.sx, camProj.sy, 4.0 + camAlpha * 4.0, 0, Math.PI * 2);
        ctx.fillStyle = outside ? `rgba(255, 90, 90, ${camAlpha})` : `rgba(255, 255, 255, ${camAlpha})`;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Draw dataset radius indicator
        ctx.strokeStyle = 'rgba(0, 170, 255, 0.2)';
        ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) * 0.008));
        ctx.beginPath();
        ctx.arc(cx, cy, displayRadius, 0, Math.PI * 2);
        ctx.stroke();

        // If expanded, add readout
        if (isExpanded) {
            ctx.fillStyle = 'rgba(207, 239, 255, 0.9)';
            ctx.font = `${Math.max(12, Math.round(Math.min(w, h) * 0.03))}px system-ui, -apple-system, Segoe UI, sans-serif`;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            const ax = (camera.position.x / AU_M).toFixed(2);
            const ay = (camera.position.y / AU_M).toFixed(2);
            const az = (camera.position.z / AU_M).toFixed(2);
            const radiusAu = (datasetRadius / AU_M).toFixed(2);
            ctx.fillText(`Pos (AU): ${ax}, ${ay}, ${az}`, 18, 18);
            ctx.fillText(`Dataset radius: ${radiusAu} AU`, 18, 40);
            let ty = 62;
            if (outside) {
                ctx.fillText('You are outside the dataset radius', 18, ty);
                ty += 22;
            }
            ctx.fillText('Tap anywhere to close', 18, ty);
        }

        ctx.restore();

        // Outline
        ctx.beginPath();
        ctx.arc(cx, cy, displayRadius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 170, 255, 0.25)';
        ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) * 0.012));
        ctx.stroke();
    };

    // Sphere rotation drag (expanded view only)
    if (navOrbEl) {
        navOrbEl.addEventListener('mousedown', (e) => {
            if (!navOrbEl.classList.contains('expanded')) return;
            e.preventDefault();
            sphereDragActive = true;
            sphereDragStartX = e.clientX;
            sphereDragStartY = e.clientY;
        });

        document.addEventListener('mousemove', (e) => {
            if (!sphereDragActive) return;
            const dx = e.clientX - sphereDragStartX;
            const dy = e.clientY - sphereDragStartY;
            sphereRotY += dx * 0.01;
            sphereRotX = Math.max(-Math.PI * 0.4, Math.min(Math.PI * 0.4, sphereRotX - dy * 0.01));
            sphereDragStartX = e.clientX;
            sphereDragStartY = e.clientY;
        });

        document.addEventListener('mouseup', () => {
            sphereDragActive = false;
        });
    }

    const updateNavAndOverlays = (viewProj: Float32Array) => {
        // --- HUD extras (distance / speed / angles) ---
        const distSunM = Math.sqrt(
            camera.position.x * camera.position.x +
                camera.position.y * camera.position.y +
                camera.position.z * camera.position.z,
        );
        if (navDistSunEl) navDistSunEl.textContent = `${(distSunM / AU_M).toFixed(2)} AU`;

        // Update location breadcrumb
        if (locationContextEl) {
            const regime = determineRegime(distSunM);
            const distFormatted = formatDistance(distSunM);

            // Find nearest landmark
            const nearest = landmarks?.findNearest(camera.position, 1)[0];
            const nearestName = nearest ? nearest.landmark.name : 'Unknown';
            const nearestDist = nearest ? formatDistance(nearest.distance) : '';

            let breadcrumb = '';
            switch (regime) {
                case ScaleRegime.Planetary:
                case ScaleRegime.SolarSystem:
                    breadcrumb = `<span class="location-primary">${nearestName}</span>`;
                    if (nearest && nearest.distance > 0) {
                        breadcrumb += ` <span class="location-sep">•</span> ${nearestDist}`;
                    }
                    break;
                case ScaleRegime.Interstellar:
                    breadcrumb = `<span class="location-primary">Local Stellar Neighborhood</span> <span class="location-sep">•</span> ${distFormatted} from Sun`;
                    break;
                case ScaleRegime.Galactic:
                    breadcrumb = `<span class="location-primary">Milky Way</span> <span class="location-sep">•</span> ${distFormatted} from Sun`;
                    break;
                case ScaleRegime.Intergalactic:
                    breadcrumb = `<span class="location-primary">Local Group</span> <span class="location-sep">•</span> ${distFormatted} from Milky Way`;
                    break;
            }
            locationContextEl.innerHTML = breadcrumb;
        }

        const speedAuPerSec = camera.speed / AU_M;
        if (navSpeedEl) {
            const txt = Math.abs(speedAuPerSec) >= 1 ? speedAuPerSec.toFixed(2) : speedAuPerSec.toFixed(4);
            navSpeedEl.textContent = `${txt} AU/s`;
        }

        const fwd = camera.forward();
        let yawDeg = (Math.atan2(-fwd.x, -fwd.z) * 180) / Math.PI;
        if (yawDeg < 0) yawDeg += 360;
        const pitchDeg = (Math.asin(fwd.y) * 180) / Math.PI;
        if (navAnglesEl) navAnglesEl.textContent = `${yawDeg.toFixed(0)}° / ${pitchDeg.toFixed(0)}°`;

        const mode = flightControls.getMode();
        if (navModeEl) navModeEl.textContent = mode === 'flight' ? 'FLIGHT' : 'ORBIT';
        
        // Update exit orbit button visibility
        if (exitOrbitBtn) {
            exitOrbitBtn.style.display = mode === 'orbitFocus' ? 'block' : 'none';
        }

        // Orbit hint (helps explain the “locked” feeling)
        const orbitHintEl = document.getElementById('orbit-hint');
        if (orbitHintEl) orbitHintEl.classList.toggle('hidden', mode !== 'orbitFocus');
        
        // Update throttle display
        const throttle = flightControls.getThrottle();
        if (navSpeedEl) {
            const speedAuPerSec = camera.speed / AU_M;
            const throttlePct = Math.abs(throttle) * 100;
            const txt = Math.abs(speedAuPerSec) >= 1 ? speedAuPerSec.toFixed(2) : speedAuPerSec.toFixed(4);
            navSpeedEl.textContent = `${txt} AU/s (${throttlePct.toFixed(0)}%)`;
        }
        
        // Update keyboard debug
        const lastKeyEl = document.getElementById('last-key');
        if (lastKeyEl) {
            lastKeyEl.textContent = input.getLastKeyDebug() || '--';
        }

        // --- Jump HUD ---
        const jumpState = flightControls.getJumpState();
        const jumpHudEl = document.getElementById('jump-hud');
        const jumpTargetEl = document.getElementById('jump-target');
        const jumpDistanceEl = document.getElementById('jump-distance');
        if (jumpHudEl) {
            const isJumping = jumpState.phase !== 'inactive';
            jumpHudEl.classList.toggle('hidden', !isJumping);
            if (isJumping && jumpTargetEl && jumpDistanceEl) {
                jumpTargetEl.textContent = jumpState.targetName || 'Unknown';
                // Compute remaining distance to arrival (stop shell), not to target center.
                const dx = jumpState.targetPos.x - camera.position.x;
                const dy = jumpState.targetPos.y - camera.position.y;
                const dz = jumpState.targetPos.z - camera.position.z;
                const distToCenter = Math.sqrt(dx * dx + dy * dy + dz * dz);
                const remainingToArrival = Math.max(0, distToCenter - (jumpState.stopDistAbsM || 0));
                jumpDistanceEl.textContent = `${formatDistance(remainingToArrival)} remaining`;
            }
        }

        // --- Mode display update (include jump status) ---
        if (navModeEl) {
            if (jumpState.phase !== 'inactive') {
                navModeEl.textContent = `JUMP (${jumpState.phase})`;
            } else {
                navModeEl.textContent = mode === 'flight' ? 'FLIGHT' : 'ORBIT';
            }
        }

        // Convert to CSS pixels for DOM positioning.
        const wCss = video.width / dpr;
        const hCss = video.height / dpr;

        // --- Targeting (crosshair pick) ---
        // Select the solar body closest to the crosshair, within a small NDC radius.
        let bestTarget: { body: SolarBody; d2: number; distM: number } | null = null;
        const THRESH = 0.02; // NDC radius (~2% of viewport)
        const THRESH2 = THRESH * THRESH;

        for (const b of solarBodies) {
            const wp = b.pos;
            const rx = wp[0] - camera.origin.x;
            const ry = wp[1] - camera.origin.y;
            const rz = wp[2] - camera.origin.z;

            const clipX = viewProj[0] * rx + viewProj[4] * ry + viewProj[8] * rz + viewProj[12];
            const clipY = viewProj[1] * rx + viewProj[5] * ry + viewProj[9] * rz + viewProj[13];
            const clipW = viewProj[3] * rx + viewProj[7] * ry + viewProj[11] * rz + viewProj[15];
            if (clipW <= 0) continue;

            const ndcX = clipX / clipW;
            const ndcY = clipY / clipW;
            const d2 = ndcX * ndcX + ndcY * ndcY;
            if (d2 > THRESH2) continue;

            const dx = camera.position.x - wp[0];
            const dy = camera.position.y - wp[1];
            const dz = camera.position.z - wp[2];
            const distM = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (!bestTarget || d2 < bestTarget.d2) {
                bestTarget = { body: b, d2, distM };
            }
        }

        targetBody = bestTarget ? bestTarget.body : null;
        if (crosshairEl) crosshairEl.classList.toggle('target', !!targetBody);
        if (navTargetEl) {
            navTargetEl.textContent = targetBody ? `${targetBody.name} · ${formatDistance(bestTarget!.distM)}` : '--';
        }

        // --- Compass (axis triad) ---
        if (compassCanvas && compassCtx) {
            const ctx = compassCtx;
            const w = compassCanvas.width;
            const h = compassCanvas.height;
            ctx.clearRect(0, 0, w, h);

            const cx = w * 0.5;
            const cy = h * 0.5;
            const radius = Math.min(w, h) * 0.38;

            // Backplate ring
            ctx.save();
            ctx.translate(cx, cy);
            ctx.beginPath();
            ctx.arc(0, 0, radius, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(0, 170, 255, 0.25)';
            ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) * 0.015));
            ctx.stroke();

            const camR = camera.right();
            const camU = camera.up();
            const camF = camera.forward();

            const axes: Array<{ name: string; v: { x: number; y: number; z: number }; color: string }> = [
                { name: 'X', v: { x: 1, y: 0, z: 0 }, color: '#ff5a5a' },
                { name: 'Y', v: { x: 0, y: 1, z: 0 }, color: '#62ff62' },
                { name: 'Z', v: { x: 0, y: 0, z: 1 }, color: '#5a7dff' },
            ];

            ctx.font = `${Math.max(10, Math.round(Math.min(w, h) * 0.12))}px system-ui, -apple-system, Segoe UI, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            for (const a of axes) {
                const x = dot3(a.v, camR);
                const y = dot3(a.v, camU);
                const z = dot3(a.v, camF);
                const alpha = 0.35 + 0.65 * ((z + 1) * 0.5);

                const dx = x * radius;
                const dy = -y * radius;

                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(dx, dy);
                ctx.strokeStyle = a.color.replace(')', `, ${alpha})`);
                // Some browsers don't support color.replace trick on hex; set via rgba:
                ctx.strokeStyle =
                    a.name === 'X'
                        ? `rgba(255, 90, 90, ${alpha})`
                        : a.name === 'Y'
                          ? `rgba(98, 255, 98, ${alpha})`
                          : `rgba(90, 125, 255, ${alpha})`;
                ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) * 0.018));
                ctx.stroke();

                ctx.fillStyle =
                    a.name === 'X'
                        ? `rgba(255, 90, 90, ${Math.min(1, alpha + 0.2)})`
                        : a.name === 'Y'
                          ? `rgba(98, 255, 98, ${Math.min(1, alpha + 0.2)})`
                          : `rgba(90, 125, 255, ${Math.min(1, alpha + 0.2)})`;
                ctx.fillText(a.name, dx * 1.12, dy * 1.12);
            }
            ctx.restore();
        }

        // --- Sun direction indicator (always visible, edge-clamped when off-screen) ---
        const sunIndicatorEl = document.getElementById('sun-indicator');
        const sunArrowEl = document.getElementById('sun-arrow');
        const sunDistLabelEl = document.getElementById('sun-dist-label');
        if (sunIndicatorEl && sunArrowEl && sunDistLabelEl) {
            // Sun is at origin (0, 0, 0)
            const sunRx = 0 - camera.origin.x;
            const sunRy = 0 - camera.origin.y;
            const sunRz = 0 - camera.origin.z;

            // Project Sun position to clip space
            const clipX = viewProj[0] * sunRx + viewProj[4] * sunRy + viewProj[8] * sunRz + viewProj[12];
            const clipY = viewProj[1] * sunRx + viewProj[5] * sunRy + viewProj[9] * sunRz + viewProj[13];
            const clipW = viewProj[3] * sunRx + viewProj[7] * sunRy + viewProj[11] * sunRz + viewProj[15];

            // Update distance label
            sunDistLabelEl.textContent = `${(distSunM / AU_M).toFixed(2)} AU`;

            // Check if Sun is in front of camera
            if (clipW > 0) {
                const ndcX = clipX / clipW;
                const ndcY = clipY / clipW;

                // Check if on screen
                const onScreen = ndcX >= -1.0 && ndcX <= 1.0 && ndcY >= -1.0 && ndcY <= 1.0;

                if (onScreen) {
                    // Sun is on-screen: position indicator directly over it
                    const sx = (ndcX * 0.5 + 0.5) * wCss;
                    const sy = (1.0 - (ndcY * 0.5 + 0.5)) * hCss;
                    sunIndicatorEl.style.left = `${sx.toFixed(1)}px`;
                    sunIndicatorEl.style.top = `${sy.toFixed(1)}px`;
                    sunIndicatorEl.style.transform = 'translate(-50%, -50%)';
                    sunArrowEl.style.transform = 'none';
                    sunIndicatorEl.classList.add('on-screen');
                } else {
                    // Sun is off-screen: clamp to edge and show arrow
                    const edgePadding = 50;
                    const maxX = wCss - edgePadding;
                    const maxY = hCss - edgePadding;

                    // Clamp to viewport edges
                    let sx = (ndcX * 0.5 + 0.5) * wCss;
                    let sy = (1.0 - (ndcY * 0.5 + 0.5)) * hCss;
                    sx = Math.max(edgePadding, Math.min(maxX, sx));
                    sy = Math.max(edgePadding, Math.min(maxY, sy));

                    // Compute arrow rotation to point toward Sun
                    const centerX = wCss / 2;
                    const centerY = hCss / 2;
                    const dirX = (ndcX * 0.5 + 0.5) * wCss - centerX;
                    const dirY = centerY - (ndcY * 0.5 + 0.5) * hCss; // flip Y
                    const angle = Math.atan2(dirX, dirY) * 180 / Math.PI;

                    sunIndicatorEl.style.left = `${sx.toFixed(1)}px`;
                    sunIndicatorEl.style.top = `${sy.toFixed(1)}px`;
                    sunIndicatorEl.style.transform = 'translate(-50%, -50%)';
                    sunArrowEl.style.transform = `rotate(${angle.toFixed(0)}deg)`;
                    sunIndicatorEl.classList.remove('on-screen');
                }
                sunIndicatorEl.classList.remove('hidden');
            } else {
                // Sun is behind camera: show at bottom of screen pointing backward
                const edgePadding = 50;
                sunIndicatorEl.style.left = `${(wCss / 2).toFixed(1)}px`;
                sunIndicatorEl.style.top = `${(hCss - edgePadding).toFixed(1)}px`;
                sunIndicatorEl.style.transform = 'translate(-50%, -50%)';
                sunArrowEl.style.transform = 'rotate(180deg)';
                sunIndicatorEl.classList.remove('on-screen');
                sunIndicatorEl.classList.remove('hidden');
            }
        }

        // --- World labels (Sun/Earth/Mars) ---
        if (labelEls.length > 0) {
            for (const l of labelEls) {
                const wp = l.body.pos;
                const rx = wp[0] - camera.origin.x;
                const ry = wp[1] - camera.origin.y;
                const rz = wp[2] - camera.origin.z;

                // clip = viewProj * [rel, 1]
                const clipX = viewProj[0] * rx + viewProj[4] * ry + viewProj[8] * rz + viewProj[12];
                const clipY = viewProj[1] * rx + viewProj[5] * ry + viewProj[9] * rz + viewProj[13];
                const clipW = viewProj[3] * rx + viewProj[7] * ry + viewProj[11] * rz + viewProj[15];

                if (clipW <= 0) {
                    l.el.classList.add('hidden');
                    l.el.classList.remove('active');
                    continue;
                }

                const ndcX = clipX / clipW;
                const ndcY = clipY / clipW;

                // Hide if off-screen (keep simple; we can add edge-clamping later).
                if (ndcX < -1.05 || ndcX > 1.05 || ndcY < -1.05 || ndcY > 1.05) {
                    l.el.classList.add('hidden');
                    l.el.classList.remove('active');
                    continue;
                }

                const sx = (ndcX * 0.5 + 0.5) * wCss;
                const sy = (1.0 - (ndcY * 0.5 + 0.5)) * hCss;

                l.el.style.left = `${sx.toFixed(1)}px`;
                l.el.style.top = `${sy.toFixed(1)}px`;
                l.el.classList.remove('hidden');

                const dx = camera.position.x - wp[0];
                const dy = camera.position.y - wp[1];
                const dz = camera.position.z - wp[2];
                const distM = Math.sqrt(dx * dx + dy * dy + dz * dz);
                l.distEl.textContent = formatDistance(distM);
                l.el.classList.toggle('active', !!targetBody && targetBody.name === l.body.name);
            }
        }

        // --- Minimap orb ---
        drawNavOrb();
    };

    // ---------------------------------------------------------------------------------
    // On-demand streaming (dataset mode)
    // - Maintain a small in-memory cache of decoded cells
    // - Select target cells around the camera (budgeted)
    // - Rebuild GPU instance buffer only when the set changes (throttled)
    // ---------------------------------------------------------------------------------
    type LoadedCell = {
        file: string;
        centroidX: number;
        centroidY: number;
        centroidZ: number;
        splats14: Float32Array;
        splatCount: number;
        lastUsed: number;
        bytes: number;
    };

    const loaded = new Map<string, LoadedCell>();
    const inflight = new Set<string>();

    let desiredEntries: (typeof manifest.cells) = [];
    let desiredSet = new Set<string>();

    // Render buffers (dynamic)
    let finalWorldPos = new Float64Array(0);
    let finalGpuSplats = new Float32Array(0);
    let lastRebuildMs = 0;
    let dirtyRebuild = true;
    let hasAnyStars = false;

    const selectionMode = (params.get('select') ?? 'distance').toLowerCase();

    const computeTargetEntries = () => {
        // Small datasets: load everything (fast + avoids "dark" sparse view).
        if (smallDataset) return manifest.cells.slice(0, maxCells);
        if (selectionMode === 'view') return selectEntriesByView(camera.position, camera.forward());
        return selectEntriesByDistance(camera.position);
    };

    const sameSet = (a: Set<string>, b: Set<string>) => {
        if (a.size !== b.size) return false;
        for (const x of a) if (!b.has(x)) return false;
        return true;
    };

    const updateUi = (loadedInViewCells: number, targetCells: number, loadedInViewSplats: number, targetSplats: number) => {
        hud.updateDatasetProgress(loadedInViewCells, targetCells, loadedInViewSplats, targetSplats);
        if (!hasAnyStars) {
            loading.classList.remove('hidden');
            loading.innerHTML = `
                <div class="spinner"></div>
                <div>Loading dataset…</div>
                <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
                    Cells: ${loadedInViewCells}/${targetCells} · Splats: ${loadedInViewSplats}/${targetSplats}
                </div>
            `;
        }
    };

    const fillSolar = (worldPos: Float64Array, gpu: Float32Array, base: number) => {
        for (let i = 0; i < solarBodies.length; i++) {
            const b = solarBodies[i];
            const idx = base + i;

            const dst3 = idx * 3;
            worldPos[dst3 + 0] = b.pos[0];
            worldPos[dst3 + 1] = b.pos[1];
            worldPos[dst3 + 2] = b.pos[2];

            const dst16 = idx * 16;
            // pos (relative to floating origin)
            gpu[dst16 + 0] = (b.pos[0] - camera.origin.x) as number;
            gpu[dst16 + 1] = (b.pos[1] - camera.origin.y) as number;
            gpu[dst16 + 2] = (b.pos[2] - camera.origin.z) as number;
            gpu[dst16 + 3] = 0;

            gpu[dst16 + 4] = b.radius_m;
            gpu[dst16 + 5] = b.radius_m;
            gpu[dst16 + 6] = b.radius_m;
            gpu[dst16 + 7] = 0;

            gpu[dst16 + 8] = 0;
            gpu[dst16 + 9] = 0;
            gpu[dst16 + 10] = 0;
            gpu[dst16 + 11] = 1;

            gpu[dst16 + 12] = b.color[0];
            gpu[dst16 + 13] = b.color[1];
            gpu[dst16 + 14] = b.color[2];
            gpu[dst16 + 15] = b.opacity;
        }
    };

    const uploadSplats = () => {
        if (rendererKind === 'webgpu') rendererWebGpu!.setSplats(finalGpuSplats);
        else rendererWebGl!.setSplats(finalGpuSplats);
    };

    const recomputeRelativePositions = () => {
        const n = Math.floor(finalGpuSplats.length / 16);
        for (let i = 0; i < n; i++) {
            const src3 = i * 3;
            const dst16 = i * 16;
            finalGpuSplats[dst16 + 0] = (finalWorldPos[src3 + 0] - camera.origin.x) as number;
            finalGpuSplats[dst16 + 1] = (finalWorldPos[src3 + 1] - camera.origin.y) as number;
            finalGpuSplats[dst16 + 2] = (finalWorldPos[src3 + 2] - camera.origin.z) as number;
        }
    };

    const rebuildBuffers = (nowMs: number) => {
        // Throttle rebuilds so we don't spam GPU uploads while streaming.
        const MIN_REBUILD_MS = 200;
        if (!dirtyRebuild) return;
        if (nowMs - lastRebuildMs < MIN_REBUILD_MS) return;

        // Target splats budget for stars (solar bodies are always included).
        const targetStarSplats = desiredEntries.reduce((sum, e) => sum + (e.splat_count || 0), 0);
        const targetSplats = Math.min(maxSplats, targetStarSplats);

        // Count how many splats we can actually render from loaded cells.
        let starCount = 0;
        let loadedInViewCells = 0;
        for (const e of desiredEntries) {
            const c = loaded.get(e.file_name);
            if (!c) continue;
            loadedInViewCells++;
            const n = Math.min(e.splat_count || 0, c.splatCount, targetSplats - starCount);
            if (n <= 0) continue;
            starCount += n;
            if (starCount >= targetSplats) break;
        }

        const totalInstances = starCount + solarSplats;
        const worldPos = new Float64Array(totalInstances * 3);
        const gpu = new Float32Array(totalInstances * 16);

        // Fill stars
        let cursor = 0;
        for (const e of desiredEntries) {
            if (cursor >= starCount) break;
            const c = loaded.get(e.file_name);
            if (!c) continue;
            const n = Math.min(e.splat_count || 0, c.splatCount, starCount - cursor);
            if (n <= 0) continue;

            const splats14 = c.splats14;
            for (let i = 0; i < n; i++) {
                const src = i * 14;
                const splatIdx = cursor + i;

                const dst3 = splatIdx * 3;
                const baseX = c.centroidX + splats14[src + 0];
                const baseY = c.centroidY + splats14[src + 1];
                const baseZ = c.centroidZ + splats14[src + 2];

                // Apply stellar proper motion based on time offset
                const currentJD = timeController.getJulianDate();
                if (Math.abs(currentJD - J2000_JD) > 1) {  // Only if time offset exists
                    // Generate deterministic velocity based on position
                    const seed = Math.abs(Math.floor(baseX + baseY * 1000 + baseZ * 1000000));
                    const velocity = estimateStellarVelocity(baseX, baseY, baseZ, seed);

                    // Propagate position
                    const newPos = propagateStarFull(
                        { x: baseX, y: baseY, z: baseZ },
                        velocity,
                        J2000_JD,
                        currentJD
                    );

                    worldPos[dst3 + 0] = newPos.x;
                    worldPos[dst3 + 1] = newPos.y;
                    worldPos[dst3 + 2] = newPos.z;
                } else {
                    // No time offset, use base positions
                    worldPos[dst3 + 0] = baseX;
                    worldPos[dst3 + 1] = baseY;
                    worldPos[dst3 + 2] = baseZ;
                }

                const dst16 = splatIdx * 16;
                gpu[dst16 + 0] = (worldPos[dst3 + 0] - camera.origin.x) as number;
                gpu[dst16 + 1] = (worldPos[dst3 + 1] - camera.origin.y) as number;
                gpu[dst16 + 2] = (worldPos[dst3 + 2] - camera.origin.z) as number;
                gpu[dst16 + 3] = 0;

                gpu[dst16 + 4] = splats14[src + 3];
                gpu[dst16 + 5] = splats14[src + 4];
                gpu[dst16 + 6] = splats14[src + 5];
                gpu[dst16 + 7] = 0;

                gpu[dst16 + 8] = splats14[src + 6];
                gpu[dst16 + 9] = splats14[src + 7];
                gpu[dst16 + 10] = splats14[src + 8];
                gpu[dst16 + 11] = splats14[src + 9];

                gpu[dst16 + 12] = splats14[src + 10];
                gpu[dst16 + 13] = splats14[src + 11];
                gpu[dst16 + 14] = splats14[src + 12];
                gpu[dst16 + 15] = splats14[src + 13];
            }

            cursor += n;
        }

        // Fill solar bodies after stars
        fillSolar(worldPos, gpu, starCount);

        finalWorldPos = worldPos;
        finalGpuSplats = gpu;
        uploadSplats();

        if (starCount > 0) {
            hasAnyStars = true;
            loading.classList.add('hidden');
        }

        updateUi(loadedInViewCells, desiredEntries.length, starCount, targetSplats);

        dirtyRebuild = false;
        lastRebuildMs = nowMs;
    };

    // Use the Rust/WASM loader for cell fetch + LZ4 decompress + parse (default).
    //
    // Query param override:
    // - `?engine=wasm` forces WASM (hard-fails if it can't init)
    // - `?engine=ts` (or `?engine=js`) forces the TypeScript loader
    // - `?engine=auto` (default) tries WASM and falls back to TS if init fails
    const engineParamRaw = (params.get('engine') ?? 'auto').toLowerCase();
    const engineParam = engineParamRaw === '' ? 'auto' : engineParamRaw;
    const wantsTsCells = engineParam === 'ts' || engineParam === 'js';
    const wantsWasmCells = engineParam === 'wasm';
    const engineAuto = engineParam === 'auto';

    let useWasmCells = !wantsTsCells;
    let wasmFetchCell: ((baseUrl: string, fileName: string) => Promise<any>) | null = null;
    let wasmParseCell: ((bytes: Uint8Array) => any) | null = null;

    if (useWasmCells) {
        loading.innerHTML = `
            <div class="spinner"></div>
            <div>Loading WASM engine…</div>
            <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
                Initializing universe-engine.wasm
            </div>
        `;
        try {
            const wasm = await import('../wasm/universe_engine.js');
            await wasm.default();
            wasmFetchCell = wasm.fetch_cell;
            wasmParseCell = wasm.parse_cell ?? null;
            console.log('[DATASET] Using WASM cell loader');
        } catch (e) {
            console.error('[DATASET] Failed to init WASM loader', e);

            if (wantsWasmCells) {
                // Forced WASM: show a clear error and fail fast.
                loading.classList.remove('hidden');
                loading.innerHTML = `
                    <div style="color: #f00">WASM Engine Failed</div>
                    <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                        Failed to initialize <code>universe-engine.wasm</code>.
                    </div>
                    <div style="font-size: 12px; margin-top: 20px; opacity: 0.5">
                        Try rebuilding with <code>./wasm-build.sh</code> and redeploying.
                    </div>
                `;
                return false;
            }

            // Auto: fall back gracefully.
            useWasmCells = false;
            wasmFetchCell = null;
            wasmParseCell = null;
            console.warn('[DATASET] Falling back to TypeScript cell loader (WASM init failed).', { engineParam, engineAuto });

            // Keep going; the TS loader path below will be used.
            loading.innerHTML = `
                <div class="spinner"></div>
                <div>Loading dataset…</div>
                <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
                    WASM unavailable; using TypeScript loader
                </div>
            `;
        }
    }

    const concurrency = Math.max(4, Math.min(32, (navigator as any).hardwareConcurrency || 8));

    // Packfile fast-path: one download instead of thousands of tiny requests.
    // This is the main reason loading is slow on mobile/remote connections.
    let packBuf: Uint8Array | null = null;
    let packMap: Map<string, { offset: number; size: number }> | null = null;
    let packFile: string | null = null;
    let packTotalSize = 0;
    try {
        const pack = await fetchPackIndex('/universe');
        if (pack && pack.total_size_bytes > 0) {
            console.log('[DATASET] Found packfile:', pack.pack_file, pack.total_size_bytes);
            packFile = pack.pack_file;
            packTotalSize = pack.total_size_bytes;
            packMap = new Map(pack.cells.map((e) => [e.file_name, { offset: e.offset, size: e.size }]));

            // If the packfile is small, download it once. For large packfiles,
            // use HTTP Range requests per cell to avoid pulling GBs into memory.
            const MAX_INLINE_PACK = 64 * 1024 * 1024; // 64 MiB
            if (pack.total_size_bytes <= MAX_INLINE_PACK) {
                console.log('[DATASET] Downloading inline packfile...');
                const res = await fetch(`/universe/${pack.pack_file}`, { cache: 'no-cache' });
                console.log('[DATASET] Pack fetch response:', res.status, res.ok);
                if (res.ok) {
                    const ab = await res.arrayBuffer();
                    packBuf = new Uint8Array(ab);
                    console.log('[DATASET] Pack loaded inline:', packBuf.byteLength, 'bytes');
                } else {
                    console.error('[DATASET] Pack fetch failed:', res.status, res.statusText);
                }
            } else {
                console.log('[DATASET] Packfile is large; using HTTP Range requests:', pack.pack_file);
            }
        }
    } catch (e) {
        console.warn('[DATASET] Packfile load skipped:', e);
    }

    const fetchPackSlice = async (offset: number, size: number): Promise<Uint8Array> => {
        if (!packFile) throw new Error('Packfile not available');
        const start = offset;
        const end = offset + size - 1;
        const res = await fetch(`/universe/${packFile}`, {
            cache: 'no-cache',
            headers: {
                Range: `bytes=${start}-${end}`,
            },
        });
        if (!res.ok) {
            throw new Error(`Pack range fetch failed: ${res.status} ${res.statusText}`);
        }
        const ab = await res.arrayBuffer();
        // Some servers may ignore Range and return the full file (200).
        // In that case, fall back to slicing locally once.
        if (res.status === 200 && packTotalSize > 0 && ab.byteLength === packTotalSize) {
            packBuf = new Uint8Array(ab);
            return packBuf.subarray(offset, offset + size);
        }
        return new Uint8Array(ab);
    };

    // Async cell loader (queue + workers)
    const entryByFile = new Map(manifest.cells.map((e) => [e.file_name, e]));
    const cacheCellsLimit = Math.min(totalCellsInManifest, Math.max(maxCells, Math.floor(maxCells * 2)));

    const queue: string[] = [];
    const waiters: Array<(file: string) => void> = [];

    const enqueue = (file: string) => {
        if (loaded.has(file) || inflight.has(file)) return;
        inflight.add(file);
        const w = waiters.shift();
        if (w) w(file);
        else queue.push(file);
    };

    const nextFile = async (): Promise<string> => {
        const v = queue.shift();
        if (v) return v;
        return await new Promise((resolve) => waiters.push(resolve));
    };

    const evictIfNeeded = () => {
        if (loaded.size <= cacheCellsLimit) return;
        // Evict least recently used cells that are not in the current desired set.
        const victims = Array.from(loaded.values()).sort((a, b) => a.lastUsed - b.lastUsed);
        for (const v of victims) {
            if (loaded.size <= cacheCellsLimit) break;
            if (desiredSet.has(v.file)) continue;
            loaded.delete(v.file);
            dirtyRebuild = true;
        }
    };

    const loadCellFile = async (file: string): Promise<LoadedCell> => {
        console.log('[CELL] Loading:', file, { packBuf: !!packBuf, packMap: !!packMap, wasmParseCell: !!wasmParseCell });
        let centroidX = 0;
        let centroidY = 0;
        let centroidZ = 0;
        let splats14: Float32Array;

        if (packBuf && packMap && packMap.has(file)) {
            const p = packMap.get(file)!;
            const bytes = packBuf.subarray(p.offset, p.offset + p.size);
            if (wasmParseCell) {
                const wasmCell = wasmParseCell(bytes);
                centroidX = wasmCell.centroid_x;
                centroidY = wasmCell.centroid_y;
                centroidZ = wasmCell.centroid_z;
                splats14 = wasmCell.splats();
                try {
                    wasmCell.free?.();
                } catch {
                    // ignore
                }
            } else {
                const buf = bytes.slice().buffer;
                const cell = parseCell(buf);
                centroidX = cell.metadata.bounds.centroid.x;
                centroidY = cell.metadata.bounds.centroid.y;
                centroidZ = cell.metadata.bounds.centroid.z;
                splats14 = cell.splats;
            }
        } else if (packMap && packMap.has(file)) {
            const p = packMap.get(file)!;
            const bytes = await fetchPackSlice(p.offset, p.size);
            if (wasmParseCell) {
                const wasmCell = wasmParseCell(bytes);
                centroidX = wasmCell.centroid_x;
                centroidY = wasmCell.centroid_y;
                centroidZ = wasmCell.centroid_z;
                splats14 = wasmCell.splats();
                try {
                    wasmCell.free?.();
                } catch {
                    // ignore
                }
            } else {
                const buf = bytes.slice().buffer;
                const cell = parseCell(buf);
                centroidX = cell.metadata.bounds.centroid.x;
                centroidY = cell.metadata.bounds.centroid.y;
                centroidZ = cell.metadata.bounds.centroid.z;
                splats14 = cell.splats;
            }
        } else if (wasmFetchCell) {
            const wasmCell = await wasmFetchCell('/universe', file);
            centroidX = wasmCell.centroid_x;
            centroidY = wasmCell.centroid_y;
            centroidZ = wasmCell.centroid_z;
            splats14 = wasmCell.splats();
            try {
                wasmCell.free?.();
            } catch {
                // ignore
            }
        } else {
            const cell = await fetchCell(file, '/universe');
            centroidX = cell.metadata.bounds.centroid.x;
            centroidY = cell.metadata.bounds.centroid.y;
            centroidZ = cell.metadata.bounds.centroid.z;
            splats14 = cell.splats;
        }

        const expected = entryByFile.get(file)?.splat_count ?? 0;
        const splatCount = Math.min(expected, Math.floor(splats14.length / 14));
        const bytes = 8 * 3 + (splats14?.byteLength ?? 0);
        return { file, centroidX, centroidY, centroidZ, splats14, splatCount, lastUsed: performance.now(), bytes };
    };

    // Start workers
    console.log('[WORKERS] Starting', concurrency, 'workers');
    for (let i = 0; i < concurrency; i++) {
        void (async () => {
            console.log(`[WORKER ${i}] Started`);
            while (true) {
                const file = await nextFile();
                console.log(`[WORKER ${i}] Got file:`, file);
                try {
                    const cell = await loadCellFile(file);
                    // Avoid growing unbounded due to stale in-flight loads.
                    if (desiredSet.has(file) || loaded.size < cacheCellsLimit) {
                        loaded.set(file, cell);
                        dirtyRebuild = true;
                    }
                } catch (e) {
                    console.warn('[DATASET] Failed to load cell', file, e);
                } finally {
                    inflight.delete(file);
                }
            }
        })();
    }

    const updateDesired = () => {
        const nextEntries = computeTargetEntries();
        const nextSet = new Set(nextEntries.map((e) => e.file_name));
        if (!sameSet(desiredSet, nextSet)) {
            desiredEntries = nextEntries;
            desiredSet = nextSet;
            dirtyRebuild = true;
        }

        const now = performance.now();
        for (const e of desiredEntries) {
            const f = e.file_name;
            const c = loaded.get(f);
            if (c) c.lastUsed = now;
            else enqueue(f);
        }
        evictIfNeeded();
    };

    // Initial buffers: solar system only (so "Earth" jumps make sense immediately)
    {
        finalWorldPos = new Float64Array(solarSplats * 3);
        finalGpuSplats = new Float32Array(solarSplats * 16);
        fillSolar(finalWorldPos, finalGpuSplats, 0);
        uploadSplats();
    }

    // Prime desired set and start background loads.
    desiredEntries = computeTargetEntries();
    console.log('[INIT] Computed target entries:', desiredEntries.length);
    desiredSet = new Set(desiredEntries.map((e) => e.file_name));
    console.log('[INIT] Enqueueing', desiredEntries.length, 'cells');
    for (const e of desiredEntries) enqueue(e.file_name);
    console.log('[INIT] Queue size:', queue.length, 'Waiters:', waiters.length, 'Inflight:', inflight.size);
    dirtyRebuild = true;

    // Render loop
    let last = performance.now();
    let fpsSmoothed = 0;
    let lastStreamUpdateMs = 0;
    let lastStreamPos = { ...camera.position };
    let lastStreamForward = camera.forward();

    const makeHudState = (fps: number) => ({
        type: 'State' as const,
        epoch_jd: 2451545.0,
        time_rate: 0,
        camera_x: camera.position.x,
        camera_y: camera.position.y,
        camera_z: camera.position.z,
        fps,
        clients: 1,
    });

    let lastDebugUpdateMs = 0;
    const updateDebugOverlay = (now: number, dt: number) => {
        if (!debugOverlayEl) return;
        // Throttle DOM updates a bit to avoid layout churn.
        if (now - lastDebugUpdateMs < 250) return;
        lastDebugUpdateMs = now;

        let loadedBytes = 0;
        for (const c of loaded.values()) loadedBytes += c.bytes;
        const loadedCells = loaded.size;
        const desiredCells = desiredEntries.length;
        const inflightCells = inflight.size;

        const packMode =
            packFile && packBuf ? `inline(${(packBuf.byteLength / (1024 * 1024)).toFixed(1)}MiB)` : packFile ? 'range' : 'none';

        debugOverlayEl.innerHTML = [
            `<div><b>renderer</b>: ${rendererKind}</div>`,
            `<div><b>cells_engine</b>: ${useWasmCells ? 'wasm' : 'ts'} (${engineParam})</div>`,
            `<div><b>pack</b>: ${packMode}</div>`,
            `<div><b>cells</b>: loaded ${loadedCells} / target ${desiredCells} (inflight ${inflightCells})</div>`,
            `<div><b>cache</b>: ${(loadedBytes / (1024 * 1024)).toFixed(1)} MiB</div>`,
            `<div><b>fps</b>: ${fpsSmoothed.toFixed(1)} · <b>dt</b>: ${(dt * 1000).toFixed(1)} ms</div>`,
        ].join('');
    };

    function frame(now: number) {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;

        input.update(dt);

        // Update time controller
        timeController.tick(dt);

        // Stream selection / background loading (throttled).
        {
            const dxs = camera.position.x - lastStreamPos.x;
            const dys = camera.position.y - lastStreamPos.y;
            const dzs = camera.position.z - lastStreamPos.z;
            const moved2 = dxs * dxs + dys * dys + dzs * dzs;
            const MOVE_THRESH = 2e10; // meters (~0.13 AU)
            const INTERVAL_MS = 250;
            let rotated = false;
            if (selectionMode === 'view') {
                const f = camera.forward();
                const dot = f.x * lastStreamForward.x + f.y * lastStreamForward.y + f.z * lastStreamForward.z;
                // ~15 degrees
                rotated = dot < 0.9659;
            }

            if (now - lastStreamUpdateMs > INTERVAL_MS || moved2 > MOVE_THRESH * MOVE_THRESH || rotated) {
                lastStreamUpdateMs = now;
                lastStreamPos = { ...camera.position };
                lastStreamForward = camera.forward();
                updateDesired();
            }
        }

        // Rebuild GPU buffers only when needed (throttled).
        rebuildBuffers(now);

        // Buffering overlay (dataset mode): show briefly when we are missing most target cells.
        if (hasAnyStars) {
            let loadedInViewCells = 0;
            for (const e of desiredEntries) if (loaded.has(e.file_name)) loadedInViewCells++;
            const need = desiredEntries.length;
            const active = inflight.size > 0 && need > 0 && loadedInViewCells < Math.min(10, Math.ceil(need * 0.2));
            if (active) buffering.classList.remove('hidden');
            else buffering.classList.add('hidden');
        } else {
            buffering.classList.add('hidden');
        }

        // Floating origin rebase: keep camera near origin for precision.
        // We only rewrite all splat positions when rebasing, not every frame.
        const dx = camera.position.x - camera.origin.x;
        const dy = camera.position.y - camera.origin.y;
        const dz = camera.position.z - camera.origin.z;
        const dist2 = dx * dx + dy * dy + dz * dz;
        const REBASE_DIST = 5e9; // meters (~0.03 AU)
        if (dist2 > REBASE_DIST * REBASE_DIST) {
            camera.origin = { ...camera.position };
            recomputeRelativePositions();
            uploadSplats();
        }

        // Render + update navigation overlays.
        let viewProj: Float32Array;
        if (rendererKind === 'webgpu') {
            const u = camera.cameraUniform(video.width / video.height);
            rendererWebGpu!.render(u);
            viewProj = u.subarray(32, 48);
        } else {
            const view = camera.viewMatrix();
            const proj = camera.projectionMatrix(video.width / video.height);
            viewProj = mat4Mul(proj, view);
            rendererWebGl!.render(
                view,
                proj,
                camera.far,
                camera.logDepthC,
            );
        }

        updateNavAndOverlays(viewProj);

        // Draw warp streaks during jumps
        const jumpState = flightControls.getJumpState();
        const isJumping = jumpState.phase !== 'inactive';
        const warpIntensity = isJumping ? Math.abs(flightControls.getThrottle()) : 0;
        drawWarpStreaks(warpIntensity);

        const fpsInst = dt > 0 ? 1 / dt : 0;
        fpsSmoothed = fpsSmoothed ? fpsSmoothed * 0.9 + fpsInst * 0.1 : fpsInst;
        hud.update(makeHudState(fpsSmoothed));
        updateDebugOverlay(now, dt);

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
    return true;
}

async function runStreamMode(dom: Dom, params: URLSearchParams) {
    const video = resetVideoCanvas(dom);
    const { videoContainer, loading, clickPrompt, status, buffering } = dom;

    // Determine WebSocket base URL
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host;
    const baseUrl = `${wsProtocol}//${wsHost}`;

    console.log('Connecting to:', baseUrl);

    // Registration scaffold: URL param (?registered=1) or localStorage toggle
    const registered =
        params.get('registered') === '1' ||
        params.get('registered')?.toLowerCase() === 'true' ||
        window.localStorage.getItem('universe_registered') === '1';

    const supportsWebCodecs = typeof (window as any).VideoDecoder !== 'undefined';

    // Create client
    const client = new UniverseClient(baseUrl, { registered, preferH264: supportsWebCodecs });

    // Create HUD
    const hud = new HUD();

    // Create input handler
    const input = new InputHandler(client);

    // Canvas draw helpers ("contain" fit)
    const ctx = video.getContext('2d', { alpha: false })!;
    const dpr = window.devicePixelRatio || 1;

    function resizeCanvas() {
        const rect = videoContainer.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (video.width !== w || video.height !== h) {
            video.width = w;
            video.height = h;
        }
    }

    function drawContain(source: CanvasImageSource, srcW: number, srcH: number) {
        resizeCanvas();
        const cw = video.width;
        const ch = video.height;
        const scale = Math.min(cw / srcW, ch / srcH);
        const dw = Math.floor(srcW * scale);
        const dh = Math.floor(srcH * scale);
        const dx = Math.floor((cw - dw) / 2);
        const dy = Math.floor((ch - dh) / 2);
        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(source, dx, dy, dw, dh);
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Connection status updates
    client.onStatusChange = (newStatus: string) => {
        status.textContent = `● ${newStatus}`;
        status.className = newStatus.toLowerCase().replace(/\s+/g, '');

        if (newStatus === 'CONNECTED') {
            loading.classList.add('hidden');
            clickPrompt.classList.remove('hidden');
        } else if (newStatus === 'DISCONNECTED') {
            loading.classList.remove('hidden');
            loading.innerHTML = `
                <div class="spinner"></div>
                <div style="color: #f00">Connection Lost</div>
                <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                    Attempting to reconnect...
                </div>
            `;
        }
    };

    // State updates from server
    client.onStateUpdate = (state) => {
        hud.update(state);
    };

    // Jump budget updates
    client.onJumpStatusUpdate = (remaining, max, isRegistered) => {
        hud.updateJumpStatus(remaining, max, isRegistered);
    };

    // Buffering overlay (teleport UX)
    client.onBufferingChange = (active: boolean) => {
        if (active) buffering.classList.remove('hidden');
        else buffering.classList.add('hidden');
    };

    // Latency updates
    client.onLatencyUpdate = (latency: number) => {
        hud.updateLatency(latency);
    };

    // Frame received (MJPEG)
    client.onFrameReceived = async (blob: Blob) => {
        // MJPEG fallback: decode JPEG and draw to canvas
        try {
            const bmp = await createImageBitmap(blob);
            drawContain(bmp, bmp.width, bmp.height);
            bmp.close();
        } catch {
            // ignore
        }
    };

    // H.264 (WebCodecs) path
    let decoder: VideoDecoder | null = null;
    let decoderReady = false;
    let h264FallbackStarted = false;
    let decoderErrorCount = 0;

    function annexbToAvcc(data: Uint8Array): Uint8Array {
        // Convert Annex-B start-code format to AVCC length-prefixed NALs.
        const nals: Uint8Array[] = [];
        const len = data.length;

        const find = (from: number): { pos: number; scLen: number } | null => {
            for (let i = from; i + 3 < len; i++) {
                if (data[i] === 0 && data[i + 1] === 0) {
                    if (data[i + 2] === 1) return { pos: i, scLen: 3 };
                    if (i + 4 < len && data[i + 2] === 0 && data[i + 3] === 1) return { pos: i, scLen: 4 };
                }
            }
            return null;
        };

        let cur = find(0);
        while (cur) {
            const next = find(cur.pos + cur.scLen);
            const start = cur.pos + cur.scLen;
            const end = next ? next.pos : len;
            if (end > start) {
                nals.push(data.subarray(start, end));
            }
            cur = next;
        }

        let total = 0;
        for (const nal of nals) total += 4 + nal.length;
        const out = new Uint8Array(total);
        const dv = new DataView(out.buffer);
        let off = 0;
        for (const nal of nals) {
            dv.setUint32(off, nal.length, false); // big-endian
            off += 4;
            out.set(nal, off);
            off += nal.length;
        }
        return out;
    }

    async function fallbackToMjpegOnce(reason: string) {
        if (h264FallbackStarted) return;
        h264FallbackStarted = true;

        decoderReady = false;
        try {
            decoder?.close();
        } catch {
            // ignore
        }
        decoder = null;

        try {
            await client.fallbackToMjpeg(reason);
        } catch (e) {
            console.warn('[STREAM] MJPEG fallback failed:', e);
        }
    }

    client.onVideoConfig = async (codec: string, avcc: Uint8Array) => {
        if (!supportsWebCodecs) return;
        if (h264FallbackStarted) return;

        decoderReady = false;
        try {
            decoder?.close();
        } catch {
            // ignore
        }
        decoder = null;

        const baseConfig: VideoDecoderConfig = {
            codec,
            description: avcc,
            optimizeForLatency: true,
            hardwareAcceleration: 'prefer-hardware',
        } as VideoDecoderConfig;

        // Guard: some environments expose WebCodecs but do not support H.264 decode.
        // This prevents a hard failure that would leave the canvas blank.
        if (typeof (VideoDecoder as any).isConfigSupported === 'function') {
            let supported = false;
            try {
                const res = await (VideoDecoder as any).isConfigSupported(baseConfig);
                supported = !!res?.supported;
            } catch {
                // If the support check itself fails, fall back to attempting configure below.
                supported = true;
            }

            if (!supported) {
                console.warn('[WebCodecs] Unsupported VideoDecoder configuration; H.264 decode disabled.', {
                    codec,
                });
                await fallbackToMjpegOnce(`Unsupported H.264 decoder configuration (${codec})`);
                return;
            }
        }

        decoder = new VideoDecoder({
            output: (frame: VideoFrame) => {
                try {
                    const w = frame.displayWidth || frame.codedWidth;
                    const h = frame.displayHeight || frame.codedHeight;
                    drawContain(frame, w, h);
                } finally {
                    frame.close();
                }
            },
            error: (e) => {
                console.error('VideoDecoder error:', e);
                decoderErrorCount++;
                // Some platforms expose WebCodecs but have no H.264 decoder; fall back quickly.
                if (decoderErrorCount >= 2) {
                    void fallbackToMjpegOnce('VideoDecoder error (repeated)');
                }
            },
        });

        try {
            decoder.configure(baseConfig);
            decoderReady = true;
        } catch (e) {
            decoderReady = false;
            try {
                decoder.close();
            } catch {
                // ignore
            }
            decoder = null;
            console.warn('[WebCodecs] VideoDecoder.configure failed; H.264 decode disabled.', e);
            await fallbackToMjpegOnce('VideoDecoder.configure failed');
        }
    };

    client.onH264Frame = (annexb: Uint8Array, timestampUs: number, isKey: boolean) => {
        if (!decoder || !decoderReady) return;
        try {
            const avcc = annexbToAvcc(annexb);
            decoder.decode(
                new EncodedVideoChunk({
                    type: isKey ? 'key' : 'delta',
                    timestamp: timestampUs,
                    data: avcc,
                }),
            );
        } catch {
            // ignore decode errors; decoder may be reconfigured on next keyframe/config
        }
    };

    // Click to start (for pointer lock)
    clickPrompt.addEventListener('click', () => {
        clickPrompt.classList.add('hidden');
        input.requestPointerLock();
    });

    // Also allow clicking video
    video.addEventListener('click', () => {
        if (!input.isLocked()) {
            clickPrompt.classList.remove('hidden');
        }
    });

    // Connect
    try {
        await client.connect();
        console.log('%c✓ Connected to Universe server', 'color: #0f0');
    } catch (error) {
        console.error('Connection failed:', error);
        status.textContent = '● CONNECTION FAILED';
        status.className = 'disconnected';
        loading.innerHTML = `
            <div style="color: #f00">Connection Failed</div>
            <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                Please ensure the Universe server is running on port 7878
            </div>
            <div style="font-size: 12px; margin-top: 20px; opacity: 0.5">
                Server command: cargo run -p universe-cli -- serve
            </div>
        `;
    }
}

async function main() {
    console.log('%c🚀 Universe Client Starting...', 'color: #0af; font-size: 16px; font-weight: bold');

    // Phase 2.3: Service worker tile cache (production only)
    if (import.meta.env.PROD && 'serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker
                .register('/sw.js')
                .then(() => console.log('[SW] registered'))
                .catch((e) => console.warn('[SW] registration failed', e));
        });
    }

    const dom: Dom = {
        video: document.getElementById('video') as HTMLCanvasElement,
        videoContainer: document.getElementById('video-container') as HTMLDivElement,
        loading: document.getElementById('loading') as HTMLDivElement,
        clickPrompt: document.getElementById('click-prompt') as HTMLDivElement,
        status: document.getElementById('status') as HTMLDivElement,
        buffering: document.getElementById('buffering') as HTMLDivElement,
    };

    const params = new URLSearchParams(window.location.search);
    const mode = (params.get('mode') ?? 'dataset').toLowerCase();
    const debug = params.get('debug') === '1' || params.get('debug') === 'true';

    // Phase 0.1: Stream mode is now DEBUG-ONLY. Production path is dataset mode only.
    if (mode === 'stream') {
        if (!debug) {
            console.warn(
                '%c⚠️ STREAM MODE IS DEBUG-ONLY',
                'color: #ff0; font-weight: bold; font-size: 14px;',
            );
            console.warn(
                'Stream mode (pixel streaming) is deprecated for production use. ' +
                'Add ?debug=1 to enable it for debugging purposes. ' +
                'Defaulting to dataset mode...',
            );
            // Fall through to dataset mode
        } else {
            console.warn(
                '%c🔧 DEBUG MODE: Using pixel streaming (dev/debug only)',
                'color: #f80; font-weight: bold;',
            );
            await runStreamMode(dom, params);
            return;
        }
    }

    // Production path: dataset mode (client-side rendering)
    if (mode === 'dataset' || mode === 'auto') {
        const ok = await runDatasetMode(dom, params);
        if (!ok) {
            // Dataset mode failed (e.g., missing manifest, GPU unavailable)
            // In production, we should show a proper error, not fall back to streaming.
            console.error(
                'Dataset mode failed. This is a production error. ' +
                'Stream mode is not available as a fallback in production builds.',
            );
            // The error UI is already shown by runDatasetMode
        }
        return;
    }

    // Unknown mode: default to dataset
    console.warn(`Unknown mode "${mode}", defaulting to dataset mode`);
    await runDatasetMode(dom, params);
}

// Start application
main().catch(console.error);
