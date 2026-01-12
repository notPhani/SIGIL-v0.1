window.history.scrollRestoration = 'manual';
window.addEventListener('beforeunload', () => {
    window.scrollTo(0, 0);
});
window.scrollTo(0, 0);

// Scroll-Driven Animation: Blob → Three.js Cube
const canvas = document.getElementById('blob-canvas');
const ctx = canvas.getContext('2d');
const noise = new PerlinNoise();
// Elements
const ctaButton = document.getElementById('cta-button');
const featureCard = document.getElementById('feature-card');
const cubeContainer = document.getElementById('cube-container');

// Mouse tracking
let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;

// Animation state
let animationState = 'hero';
let scrollProgress = 0;
let time = 0;

// Config
const config = {
    blob: {
        numPoints: 250,
        baseRadius: 240,
        radiusVariation: 200,
        noiseScale: 0.9,
        speed: 0.002,
        smoothness: 0.4,
        metallic: {
            highlight: '#C9B3E6',
            midtone: '#8B6FD6',
            shadow: '#4A2C6E',
            deep: '#2D1B4E'
        }
    },
    text: {
        content: 'SIGNATURES\nQUANTUM CAN\'T\nBREAK',
        normalColor: '#2D1B4E',
        intersectColor: '#e66750'
    },
    cube: {
        size: 2.5,
        targetY: 0.7
    }
};

// Resize
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Mouse tracking
window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
});

// ============================================
// THREE.JS SETUP
// ============================================

let scene, camera, renderer, cube, cubeInitialized = false;
let cubeY = 0; // Current Y position
let targetCubeY = 0; // Target Y position

function initThreeJS() {
    // Scene
    scene = new THREE.Scene();
    
    // Camera
    camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.z = 8;
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    cubeContainer.appendChild(renderer.domElement);
    
    // Cube geometry
    const geometry = new THREE.BoxGeometry(config.cube.size, config.cube.size, config.cube.size);
    
    // Materials for each face with metallic purple colors
    const materials = [
        new THREE.MeshBasicMaterial({ color: 0x8B6FD6 }), // Right
        new THREE.MeshBasicMaterial({ color: 0x4A2C6E }), // Left
        new THREE.MeshBasicMaterial({ color: 0xC9B3E6 }), // Top
        new THREE.MeshBasicMaterial({ color: 0x2D1B4E }), // Bottom
        new THREE.MeshBasicMaterial({ color: 0x8B6FD6 }), // Front
        new THREE.MeshBasicMaterial({ color: 0x4A2C6E })  // Back
    ];
    
    cube = new THREE.Mesh(geometry, materials);
    cube.scale.set(0, 0, 0); // Start invisible
    scene.add(cube);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 0.8);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);
    
    cubeInitialized = true;
    
    // Handle resize
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

function animateThreeJS() {
    if (!cubeInitialized) return;
    
    // Smooth Y position interpolation
    cubeY += (targetCubeY - cubeY) * 0.1;
    cube.position.y = cubeY;
    
    renderer.render(scene, camera);
}
// ============================================
// SECTION 2: TEXT WITH CARD MASK
// ============================================

const section2Config = {
    text: {
        content: 'YOUR\nIDENTITY\nUNFORGABLE',
        normalColor: '#2D1B4E',      // Purple outside card
        intersectColor: '#e66750'     // Warm inside card
    }
};

function drawSection2Text() {
    if (scrollProgress < 0.8) return;
    
    // Calculate fade-in alpha
    const fadeAlpha = Math.min((scrollProgress - 0.8) / 0.2, 1);
    
    const lines = section2Config.text.content.split('\n');
    const fontSize = Math.min(canvas.width * 0.12, 120);
    const lineHeight = fontSize * 1.1;
    const startY = canvas.height / 2 - (lines.length - 1) * lineHeight / 2;
    
    // Get card position
    const cardRect = featureCard.getBoundingClientRect();
    
    // 1. Draw purple text (base layer)
    ctx.globalAlpha = fadeAlpha;
    ctx.font = `900 ${fontSize}px Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = section2Config.text.normalColor;
    
    lines.forEach((line, i) => {
        ctx.fillText(line, canvas.width / 2, startY + i * lineHeight);
    });
    
    // 2. Clip to card rectangle
    ctx.save();
    ctx.beginPath();
    ctx.rect(cardRect.left, cardRect.top, cardRect.width, cardRect.height);
    ctx.clip();
    
    // 3. Draw warm text ONLY in card area
    ctx.fillStyle = section2Config.text.intersectColor;
    lines.forEach((line, i) => {
        ctx.fillText(line, canvas.width / 2, startY + i * lineHeight);
    });
    
    ctx.restore();
    ctx.globalAlpha = 1;
}


// ============================================
// SCROLL HANDLING
// ============================================


window.addEventListener('scroll', () => {
    const heroHeight = window.innerHeight;
    const scroll = window.scrollY;
    scrollProgress = Math.min(scroll / heroHeight, 1);
    
    // Update state
    if (scrollProgress < 0.3) {
        animationState = 'hero';
        cubeContainer.classList.remove('active');
    } else if (scrollProgress < 0.5) {
        animationState = 'shrinking';
        cubeContainer.classList.remove('active');
    } else if (scrollProgress < 0.8) {
        animationState = 'cube-spawn';
        if (!cubeInitialized) initThreeJS();
        cubeContainer.classList.add('active');
    } else {
        animationState = 'cube-interactive';
        cubeContainer.classList.add('active');}
    
    // Fade button
    if (ctaButton) {
        ctaButton.style.opacity = Math.max(0, 1 - scrollProgress * 3);
    }
    
    // Fade card in
    if (scrollProgress > 0.8) {
        featureCard.style.opacity = (scrollProgress - 0.8) / 0.2;
    }
});

// ============================================
// BLOB FUNCTIONS
// ============================================

function getAbstractPoints(centerX, centerY, time, radiusMultiplier = 1) {
    const points = [];
    
    for (let i = 0; i < config.blob.numPoints; i++) {
        const angle = (i / config.blob.numPoints) * Math.PI * 2;
        
        const noiseValue1 = noise.noise(
            Math.cos(angle) * config.blob.noiseScale + time * 1.5,
            Math.sin(angle) * config.blob.noiseScale + time * 1.5
        );
        
        const noiseValue2 = noise.noise(
            Math.cos(angle) * config.blob.noiseScale * 2 + time * 0.8,
            Math.sin(angle) * config.blob.noiseScale * 2 + time * 0.8
        );
        
        const combinedNoise = noiseValue1 + noiseValue2 * 0.5;
        const radius = (config.blob.baseRadius + combinedNoise * config.blob.radiusVariation) * radiusMultiplier;
        
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        points.push({ x, y });
    }
    
    return points;
}

function createMetallicGradient(centerX, centerY, radiusMultiplier = 1) {
    const gradient = ctx.createRadialGradient(
        centerX - 80, centerY - 80, 0,
        centerX, centerY, config.blob.baseRadius * 1.5 * radiusMultiplier
    );
    
    gradient.addColorStop(0, config.blob.metallic.highlight + 'B3');
    gradient.addColorStop(0.3, config.blob.metallic.midtone + 'B3');
    gradient.addColorStop(0.6, config.blob.metallic.shadow + 'B3');
    gradient.addColorStop(1, config.blob.metallic.deep + 'B3');
    
    return gradient;
}

function drawBlobShape(centerX, centerY, points) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    
    for (let i = 0; i < points.length; i++) {
        const current = points[i];
        const next = points[(i + 1) % points.length];
        const afterNext = points[(i + 2) % points.length];
        
        const cp1x = current.x + (next.x - current.x) * config.blob.smoothness;
        const cp1y = current.y + (next.y - current.y) * config.blob.smoothness;
        
        const cp2x = next.x - (afterNext.x - next.x) * config.blob.smoothness * 0.3;
        const cp2y = next.y - (afterNext.y - next.y) * config.blob.smoothness * 0.3;
        
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, next.x, next.y);
    }
    
    ctx.closePath();
}

function drawText(color, alpha = 1) {
    const lines = config.text.content.split('\n');
    const fontSize = Math.min(canvas.width * 0.12, 120);
    
    ctx.globalAlpha = alpha;
    ctx.font = `900 ${fontSize}px Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    
    const lineHeight = fontSize * 1.1;
    const startY = canvas.height / 2 - (lines.length - 1) * lineHeight / 2;
    
    lines.forEach((line, i) => {
        ctx.fillText(line, canvas.width / 2, startY + i * lineHeight);
    });
    
    ctx.globalAlpha = 1;
}

function applyFilmGrain(intensity = 1) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    
    for (let i = 0; i < pixels.length; i += 4) {
        const noise = (Math.random() - 0.5) * intensity * 255;
        pixels[i] += noise;
        pixels[i + 1] += noise;
        pixels[i + 2] += noise;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

// ============================================
// MAIN RENDER LOOP
// ============================================

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    if (animationState === 'hero') {
        // Hero: blob morphing + text
        const textAlpha = Math.max(0, 1 - scrollProgress * 3);
        const points = getAbstractPoints(centerX, centerY, time);
        
        drawText(config.text.normalColor, textAlpha);
        
        ctx.save();
        drawBlobShape(centerX, centerY, points);
        ctx.clip();
        
        const gradient = createMetallicGradient(centerX, centerY);
        ctx.fillStyle = gradient;
        drawBlobShape(centerX, centerY, points);
        ctx.fill();
        
        ctx.shadowBlur = 40;
        ctx.shadowColor = config.blob.metallic.midtone;
        ctx.fill();
        ctx.shadowBlur = 0;
        
        drawText(config.text.intersectColor, textAlpha);
        ctx.restore();
        
        time += config.blob.speed;
        
    } else if (animationState === 'shrinking') {
        // Blob shrinking to point
        const shrinkProgress = (scrollProgress - 0.3) / 0.2; // 0.3 to 0.5 range
        const radiusMultiplier = Math.max(0, 1 - shrinkProgress);
        const points = getAbstractPoints(centerX, centerY, time, radiusMultiplier);
        
        const gradient = createMetallicGradient(centerX, centerY, radiusMultiplier);
        ctx.fillStyle = gradient;
        drawBlobShape(centerX, centerY, points);
        ctx.fill();
        
        time += config.blob.speed * (1 - shrinkProgress);
        
    } else if (animationState === 'cube-spawn') {
        // Cube spawning and moving down
        const spawnProgress = (scrollProgress - 0.5) / 0.3; // 0.5 to 0.8 range
        
        // Scale cube from 0 to 1
        const scale = Math.min(spawnProgress * 2, 0.3);
        cube.scale.set(scale, scale, scale);
        
        // Move cube down
        targetCubeY = -0 * (1 - spawnProgress); // From 0 to -3
        
        // Auto rotation during spawn
        cube.rotation.x += 0.02;
        cube.rotation.y += 0.02;
        
    } else if (animationState === 'cube-interactive') {
    // Cube interactive with mouse
    const rotX = (mouseY - window.innerHeight / 2) / window.innerHeight * 0.5;
    const rotY = (mouseX - window.innerWidth / 2) / window.innerWidth * 0.5;
    
    cube.rotation.x += (rotX - cube.rotation.x) * 0.8;
    cube.rotation.y += (rotY - cube.rotation.y) * 0.8;
    
    targetCubeY = -2.5;
    
    // Draw section 2 text with card mask
    drawSection2Text();
}

// Render Three.js
if (cubeInitialized) {
    animateThreeJS();
}

// Apply grain to canvas
if (animationState === 'hero' || animationState === 'shrinking' || animationState === 'cube-interactive') {
    applyFilmGrain(0.04);
}

    
    // Render Three.js
    if (cubeInitialized) {
        animateThreeJS();
    }
    
    // Apply grain to canvas
    if (animationState === 'hero' || animationState === 'shrinking') {
        applyFilmGrain(0.04);
    }
    
    requestAnimationFrame(render);
}

// ============================================
// CARD TILT EFFECT
// ============================================

if (featureCard) {
    featureCard.addEventListener('mousemove', (e) => {
        const rect = featureCard.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        const rotateX = (y - centerY) / 10;
        const rotateY = (centerX - x) / 10;
        
        featureCard.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
    });
    
    featureCard.addEventListener('mouseleave', () => {
        featureCard.style.transform = 'perspective(1000px) rotateX(0) rotateY(0)';
    });
}

// Start
render();
