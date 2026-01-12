// Perlin Noise Implementation
// Based on Improved Noise by Ken Perlin (2002)

class PerlinNoise {
    constructor() {
        this.permutation = [];
        this.p = [];
        
        // Initialize permutation table
        for (let i = 0; i < 256; i++) {
            this.permutation[i] = Math.floor(Math.random() * 256);
        }
        
        // Duplicate for overflow
        for (let i = 0; i < 512; i++) {
            this.p[i] = this.permutation[i & 255];
        }
    }
    
    fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    
    lerp(t, a, b) {
        return a + t * (b - a);
    }
    
    grad(hash, x, y) {
        const h = hash & 15;
        const u = h < 8 ? x : y;
        const v = h < 4 ? y : h === 12 || h === 14 ? x : 0;
        return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
    }
    
    noise(x, y) {
        // Find unit square
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;
        
        // Relative position in square
        x -= Math.floor(x);
        y -= Math.floor(y);
        
        // Fade curves
        const u = this.fade(x);
        const v = this.fade(y);
        
        // Hash coordinates
        const aa = this.p[this.p[X] + Y];
        const ab = this.p[this.p[X] + Y + 1];
        const ba = this.p[this.p[X + 1] + Y];
        const bb = this.p[this.p[X + 1] + Y + 1];
        
        // Blend results
        return this.lerp(v,
            this.lerp(u, this.grad(aa, x, y), this.grad(ba, x - 1, y)),
            this.lerp(u, this.grad(ab, x, y - 1), this.grad(bb, x - 1, y - 1))
        );
    }
}
