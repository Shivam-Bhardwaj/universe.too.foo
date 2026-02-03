#!/usr/bin/env node
/**
 * Offline dataset audit: checks whether star positions are being flattened onto a plane.
 *
 * Reads `universe/index.json`, samples top-N densest cells, parses `cells/*.bin`,
 * reconstructs world positions (centroid + local pos), and prints axis ranges + thickness ratio.
 *
 * Usage:
 *   node scripts/audit_cells_planarity.mjs [universe_dir] [max_cells]
 *
 * Example:
 *   node scripts/audit_cells_planarity.mjs universe 200
 */

import fs from 'node:fs';
import path from 'node:path';

function readU32LE(buf, off) {
  return buf.readUInt32LE(off);
}

// LZ4 *block* decompressor (not LZ4 frame).
// Compatible with Rust `lz4_flex::compress_prepend_size` / `decompress_size_prepended`.
function lz4DecompressBlock(input, outputSize) {
  const out = Buffer.allocUnsafe(outputSize);
  let ip = 0;
  let op = 0;

  const readLength = (initial) => {
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
    if (ip + litLen > input.length) throw new Error('LZ4 literal length out of range');
    input.copy(out, op, ip, ip + litLen);
    ip += litLen;
    op += litLen;

    if (ip >= input.length) break; // end-of-block

    // Match offset
    if (ip + 2 > input.length) throw new Error('LZ4 missing match offset');
    const offset = input[ip] | (input[ip + 1] << 8);
    ip += 2;
    if (offset === 0) throw new Error('LZ4 invalid match offset 0');

    // Match length
    let matchLen = readLength(token & 0x0f) + 4;
    if (op + matchLen > out.length) throw new Error('LZ4 output overflow (match)');

    // Copy match (supports overlap)
    let ref = op - offset;
    if (ref < 0) throw new Error('LZ4 match reference before output start');
    while (matchLen-- > 0) out[op++] = out[ref++];
  }

  if (op !== out.length) {
    throw new Error(`LZ4 decompressed size mismatch: got ${op}, expected ${out.length}`);
  }
  return out;
}

function parseCellFile(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.length < 4) throw new Error('Cell buffer too small');
  let off = 0;

  const metaLen = readU32LE(buf, off);
  off += 4;
  if (off + metaLen > buf.length) throw new Error('Cell metadata length out of range');

  const metaJson = buf.subarray(off, off + metaLen).toString('utf8');
  off += metaLen;
  const metadata = JSON.parse(metaJson);

  const compressed = buf.subarray(off);
  if (compressed.length < 4) return { metadata, floats: new Float32Array(0) };

  const uncompressedSize = readU32LE(compressed, 0);
  const payload = compressed.subarray(4);
  const decompressed = lz4DecompressBlock(payload, uncompressedSize);
  if (decompressed.length % 4 !== 0) throw new Error('Decompressed splat bytes not aligned to 4 bytes');

  // 56 bytes per splat = 14 f32 (pos3, scale3, rot4, color3, opacity1)
  const floats = new Float32Array(
    decompressed.buffer,
    decompressed.byteOffset,
    Math.floor(decompressed.byteLength / 4),
  );
  return { metadata, floats };
}

function main() {
  const universeDir = process.argv[2] ?? 'universe';
  const maxCells = Math.max(1, parseInt(process.argv[3] ?? '200', 10) || 200);

  const manifestPath = path.join(universeDir, 'index.json');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

  const cells = [...manifest.cells].sort((a, b) => (b.splat_count || 0) - (a.splat_count || 0)).slice(0, maxCells);
  const cellsDir = path.join(universeDir, 'cells');

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let n = 0;

  // Welford (per-axis)
  let meanX = 0, meanY = 0, meanZ = 0;
  let m2X = 0, m2Y = 0, m2Z = 0;

  for (const e of cells) {
    const fp = path.join(cellsDir, e.file_name);
    if (!fs.existsSync(fp)) continue;
    const { metadata, floats } = parseCellFile(fp);
    const cx = metadata?.bounds?.centroid?.x ?? 0;
    const cy = metadata?.bounds?.centroid?.y ?? 0;
    const cz = metadata?.bounds?.centroid?.z ?? 0;

    const splatCount = Math.floor(floats.length / 14);
    for (let i = 0; i < splatCount; i++) {
      const base = i * 14;
      const x = cx + floats[base + 0];
      const y = cy + floats[base + 1];
      const z = cz + floats[base + 2];

      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;

      n++;
      const dx = x - meanX;
      meanX += dx / n;
      m2X += dx * (x - meanX);

      const dy = y - meanY;
      meanY += dy / n;
      m2Y += dy * (y - meanY);

      const dz = z - meanZ;
      meanZ += dz / n;
      m2Z += dz * (z - meanZ);
    }
  }

  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const rangeZ = maxZ - minZ;
  const denom = Math.max(1e-9, Math.max(rangeX, rangeZ));
  const thicknessRatio = rangeY / denom;

  const AU = 1.496e11;

  console.log(`Universe dir: ${universeDir}`);
  console.log(`Cells sampled: ${cells.length} (requested ${maxCells})`);
  console.log(`Splats sampled: ${n}`);
  console.log(`AABB range (AU): x=${(rangeX / AU).toFixed(2)} y=${(rangeY / AU).toFixed(2)} z=${(rangeZ / AU).toFixed(2)}`);
  console.log(`Std (AU):       x=${(Math.sqrt(m2X / Math.max(1, n - 1)) / AU).toFixed(2)} y=${(Math.sqrt(m2Y / Math.max(1, n - 1)) / AU).toFixed(2)} z=${(Math.sqrt(m2Z / Math.max(1, n - 1)) / AU).toFixed(2)}`);
  console.log(`Thickness ratio y/max(x,z): ${thicknessRatio.toExponential(3)}`);
  console.log(thicknessRatio < 1e-3 ? 'LIKELY PLANAR: y spread is tiny vs x/z' : 'NOT PLANAR: y spread is comparable to x/z');
}

main();


