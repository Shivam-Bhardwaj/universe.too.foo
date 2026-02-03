// HLG Grid Visualization
const canvas = document.getElementById('hlg-canvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('level-slider');
const valueDisplay = document.getElementById('level-value');

function drawHLGGrid(level) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#0f1220';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 50; // pixels per AU

    // Draw radial levels
    for (let l = level - 2; l <= level + 2; l++) {
        const r_min = Math.pow(2, l) * scale;
        const r_max = Math.pow(2, l + 1) * scale;

        // Inner circle
        ctx.strokeStyle = l === level ? '#667eea' : '#2a3150';
        ctx.lineWidth = l === level ? 2 : 1;
        ctx.beginPath();
        ctx.arc(centerX, centerY, r_min, 0, 2 * Math.PI);
        ctx.stroke();

        // Outer circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, r_max, 0, 2 * Math.PI);
        ctx.stroke();

        // Angular sectors
        const n_theta = Math.max(4, Math.pow(2, Math.floor(l / 2) + 1));
        ctx.strokeStyle = l === level ? '#667eea80' : '#2a315080';

        for (let i = 0; i < n_theta; i++) {
            const angle = (i * 2 * Math.PI) / n_theta;
            const x1 = centerX + r_min * Math.cos(angle);
            const y1 = centerY + r_min * Math.sin(angle);
            const x2 = centerX + r_max * Math.cos(angle);
            const y2 = centerY + r_max * Math.sin(angle);

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }

        // Label
        if (r_max < canvas.height / 2) {
            ctx.fillStyle = '#e0e0e0';
            ctx.font = '12px monospace';
            ctx.fillText(`L${l} (${Math.pow(2, l).toFixed(1)}-${Math.pow(2, l+1).toFixed(1)} AU)`,
                         centerX + r_max + 5, centerY);
        }
    }

    // Sun at center
    ctx.fillStyle = '#FDB813';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 8, 0, 2 * Math.PI);
    ctx.fill();

    // Legend
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '14px monospace';
    ctx.fillText(`Level ${level}: ${Math.pow(2, level).toFixed(1)}-${Math.pow(2, level+1).toFixed(1)} AU`, 10, 30);
    const n_theta = Math.max(4, Math.pow(2, Math.floor(level / 2) + 1));
    ctx.fillText(`Angular sectors: ${n_theta}`, 10, 50);
}

slider.addEventListener('input', (e) => {
    const level = parseInt(e.target.value);
    valueDisplay.textContent = level;
    drawHLGGrid(level);
});

drawHLGGrid(0);
