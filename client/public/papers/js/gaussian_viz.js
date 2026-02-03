// Gaussian Splatting Visualization
const gCanvas = document.getElementById('gaussian-canvas');
const gCtx = gCanvas.getContext('2d');
const scaleSlider = document.getElementById('scale-slider');
const opacitySlider = document.getElementById('opacity-slider');
const scaleValue = document.getElementById('scale-value');
const opacityValue = document.getElementById('opacity-value');

function drawGaussian(scale, opacity) {
    gCtx.clearRect(0, 0, gCanvas.width, gCanvas.height);
    gCtx.fillStyle = '#0f1220';
    gCtx.fillRect(0, 0, gCanvas.width, gCanvas.height);

    const width = gCanvas.width;
    const height = gCanvas.height;
    const imageData = gCtx.createImageData(width, height);

    // Simple 2D Gaussian
    const cx = width / 2;
    const cy = height / 2;
    const sigma = scale * 100;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dx = x - cx;
            const dy = y - cy;
            const dist_sq = (dx * dx + dy * dy) / (2 * sigma * sigma);
            const value = Math.exp(-dist_sq);

            const idx = (y * width + x) * 4;
            const color = value * opacity * 255;

            // Blue Gaussian
            imageData.data[idx] = color * 0.4;     // R
            imageData.data[idx + 1] = color * 0.5; // G
            imageData.data[idx + 2] = color;       // B
            imageData.data[idx + 3] = 255;         // A
        }
    }

    gCtx.putImageData(imageData, 0, 0);

    // Draw cross at center
    gCtx.strokeStyle = '#667eea';
    gCtx.lineWidth = 1;
    gCtx.beginPath();
    gCtx.moveTo(cx - 10, cy);
    gCtx.lineTo(cx + 10, cy);
    gCtx.moveTo(cx, cy - 10);
    gCtx.lineTo(cx, cy + 10);
    gCtx.stroke();

    // Draw info text
    gCtx.fillStyle = '#e0e0e0';
    gCtx.font = '14px monospace';
    gCtx.fillText(`σ = ${sigma.toFixed(1)} px`, 10, 30);
    gCtx.fillText(`α = ${opacity.toFixed(2)}`, 10, 50);
}

scaleSlider.addEventListener('input', (e) => {
    const scale = parseFloat(e.target.value);
    scaleValue.textContent = scale.toFixed(2);
    drawGaussian(scale, parseFloat(opacitySlider.value));
});

opacitySlider.addEventListener('input', (e) => {
    const opacity = parseFloat(e.target.value);
    opacityValue.textContent = opacity.toFixed(2);
    drawGaussian(parseFloat(scaleSlider.value), opacity);
});

drawGaussian(0.3, 0.9);
