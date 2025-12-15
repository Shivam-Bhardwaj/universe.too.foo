export interface CellId {
    l: number;
    theta: number;
    phi: number;
}

export interface CartesianPosition {
    x: number;
    y: number;
    z: number;
}

export interface CellBounds {
    min: CartesianPosition;
    max: CartesianPosition;
    centroid: CartesianPosition;
}

export interface CellMetadata {
    id: CellId;
    bounds: CellBounds;
    splat_count: number;
    compressed: boolean;
}

export interface HLGConfig {
    r_min: number;
    log_base: number;
    n_theta: number;
    n_phi: number;
}

export interface CellEntry {
    id: CellId;
    file_name: string;
    splat_count: number;
    file_size_bytes: number;
}

export interface CellManifest {
    version: number;
    config: HLGConfig;
    total_splats: number;
    total_size_bytes: number;
    cells: CellEntry[];
}

export interface CellPackEntry {
    file_name: string;
    offset: number;
    size: number;
}

export interface CellPackIndex {
    version: number;
    pack_file: string;
    total_size_bytes: number;
    cells: CellPackEntry[];
}

export interface ParsedCell {
    metadata: CellMetadata;
    /** 14-float GaussianSplat records: pos3, scale3, rot4, color3, opacity1 */
    splats: Float32Array;
}

const textDecoder = new TextDecoder();

// LZ4 *block* decompressor (not LZ4 frame).
// Compatible with Rust `lz4_flex::compress_prepend_size` / `decompress_size_prepended`.
function lz4DecompressBlock(input: Uint8Array, outputSize: number): Uint8Array {
    const out = new Uint8Array(outputSize);
    let ip = 0;
    let op = 0;

    const readLength = (initial: number): number => {
        let len = initial;
        if (len === 15) {
            while (ip < input.length) {
                const b = input[ip++];
                len += b;
                if (b !== 255) break;
            }
        }
        return len;
    };

    while (ip < input.length) {
        const token = input[ip++];

        // Literals
        const litLen = readLength(token >>> 4);
        if (ip + litLen > input.length) {
            throw new Error('LZ4 literal length out of range');
        }
        if (op + litLen > out.length) {
            throw new Error('LZ4 output overflow (literals)');
        }
        out.set(input.subarray(ip, ip + litLen), op);
        ip += litLen;
        op += litLen;

        // If we've consumed all input, we're done (valid end-of-block)
        if (ip >= input.length) break;

        // Match offset
        if (ip + 2 > input.length) {
            throw new Error('LZ4 missing match offset');
        }
        const offset = input[ip] | (input[ip + 1] << 8);
        ip += 2;
        if (offset === 0) {
            throw new Error('LZ4 invalid match offset 0');
        }

        // Match length
        let matchLen = readLength(token & 0x0f) + 4;
        if (op + matchLen > out.length) {
            throw new Error('LZ4 output overflow (match)');
        }

        // Copy match (must support overlap)
        let ref = op - offset;
        if (ref < 0) {
            throw new Error('LZ4 match reference before output start');
        }
        while (matchLen-- > 0) {
            out[op++] = out[ref++];
        }
    }

    if (op !== out.length) {
        // Some encoders may produce slightly less than the declared size, but lz4_flex should not.
        throw new Error(`LZ4 decompressed size mismatch: got ${op}, expected ${out.length}`);
    }

    return out;
}

export async function fetchManifest(basePath = '/universe'): Promise<CellManifest> {
    const url = `${basePath.replace(/\/+$/, '')}/index.json`;
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) {
        throw new Error(`Failed to fetch manifest: ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as CellManifest;
}

export async function fetchCell(fileName: string, basePath = '/universe'): Promise<ParsedCell> {
    const url = `${basePath.replace(/\/+$/, '')}/cells/${fileName}`;
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) {
        throw new Error(`Failed to fetch cell ${fileName}: ${res.status} ${res.statusText}`);
    }
    const buf = await res.arrayBuffer();
    return parseCell(buf);
}

export async function fetchPackIndex(basePath = '/universe'): Promise<CellPackIndex | null> {
    const url = `${basePath.replace(/\/+$/, '')}/cells.pack.json`;
    const res = await fetch(url, { cache: 'no-cache' });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`Failed to fetch pack index: ${res.status} ${res.statusText}`);
    return (await res.json()) as CellPackIndex;
}

export function parseCell(buf: ArrayBuffer): ParsedCell {
    const u8 = new Uint8Array(buf);
    const dv = new DataView(buf);
    let off = 0;

    if (u8.byteLength < 4) {
        throw new Error('Cell buffer too small');
    }

    // [u32 meta_len LE]
    const metaLen = dv.getUint32(off, true);
    off += 4;

    if (off + metaLen > u8.byteLength) {
        throw new Error('Cell metadata length out of range');
    }

    // [meta JSON]
    const metaBytes = u8.subarray(off, off + metaLen);
    off += metaLen;
    const metaJson = textDecoder.decode(metaBytes);
    const metadata = JSON.parse(metaJson) as CellMetadata;

    // [lz4(size-prepended) splat bytes]
    const compressed = u8.subarray(off);
    if (compressed.byteLength < 4) {
        return { metadata, splats: new Float32Array() };
    }

    const dv2 = new DataView(compressed.buffer, compressed.byteOffset, compressed.byteLength);
    const uncompressedSize = dv2.getUint32(0, true);
    const compressedPayload = compressed.subarray(4);

    const decompressed = lz4DecompressBlock(compressedPayload, uncompressedSize);

    if (decompressed.byteLength % 4 !== 0) {
        throw new Error('Decompressed splat bytes not aligned to 4 bytes');
    }

    const floats = new Float32Array(decompressed.buffer, decompressed.byteOffset, decompressed.byteLength / 4);
    return { metadata, splats: floats };
}



