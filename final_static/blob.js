/* ============================================
   1. CORE INITIALIZATION & SCROLL RESET
   ============================================ */
if ('scrollRestoration' in history) {
    history.scrollRestoration = 'manual';
}

window.scrollTo(0, 0);

window.addEventListener('beforeunload', () => {
    window.scrollTo(0, 0);
});

window.addEventListener('load', () => {
    setTimeout(() => {
        window.scrollTo(0, 0);
    }, 10);
});

/* ============================================
   2. CONFIGURATION & STATE
   ============================================ */
const canvas = document.getElementById('blob-canvas');
const ctx = canvas.getContext('2d');

// NEW: Developer Canvas Context
const devCanvas = document.getElementById('dev-canvas');
const dctx = devCanvas ? devCanvas.getContext('2d') : null;

// NEW: SDK Canvas Context (5th page)
const sdkCanvas = document.getElementById('sdk-canvas');
const sctx = sdkCanvas ? sdkCanvas.getContext('2d') : null;

const noise = typeof PerlinNoise !== 'undefined' ? new PerlinNoise() : { noise: () => Math.random() };

const ctaButton = document.getElementById('cta-button');
const featureCard = document.getElementById('feature-card');
const cubeContainer = document.getElementById('cube-container');

let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;
let animationState = 'hero';
let scrollProgress = 0;
let time = 0;

function applyFilmGrain(intensity = 1) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    
    for (let i = 0; i < pixels.length; i += 4) {
        const noiseVal = (Math.random() - 0.5) * intensity * 255;
        pixels[i] += noiseVal;
        pixels[i + 1] += noiseVal;
        pixels[i + 2] += noiseVal;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

const config = {
    blob: {
        numPoints: 500,
        baseRadius: 290,
        radiusVariation: 190,
        noiseScale: 1.5,
        speed: 0.003,
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

const section2Config = {
    text: {
        content: 'YOUR\nIDENTITY\nUNFORGABLE',
        normalColor: '#2D1B4E',
        intersectColor: '#e66750'
    }
};

/* ============================================
   3. THREE.JS ENGINE (CUBE)
   ============================================ */
let scene, camera, renderer, cube, cubeInitialized = false;
let cubeY = -3;
let targetCubeY = 0;

function initThreeJS() {
    if (cubeInitialized) return;
    
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 8;
    
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    cubeContainer.appendChild(renderer.domElement);
    
    const geometry = new THREE.BoxGeometry(config.cube.size, config.cube.size, config.cube.size);
    const materials = [
        new THREE.MeshStandardMaterial({ color: 0x8B6FD6, metalness: 0.7, roughness: 0.2 }),
        new THREE.MeshStandardMaterial({ color: 0x4A2C6E, metalness: 0.7, roughness: 0.2 }),
        new THREE.MeshStandardMaterial({ color: 0xC9B3E6, metalness: 0.7, roughness: 0.2 }),
        new THREE.MeshStandardMaterial({ color: 0x2D1B4E, metalness: 0.7, roughness: 0.2 }),
        new THREE.MeshStandardMaterial({ color: 0x8B6FD6, metalness: 0.7, roughness: 0.2 }),
        new THREE.MeshStandardMaterial({ color: 0x4A2C6E, metalness: 0.7, roughness: 0.2 })
    ];
    
    cube = new THREE.Mesh(geometry, materials);
    cube.scale.set(0, 0, 0);
    scene.add(cube);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);
    
    cubeInitialized = true;
}

function animateThreeJS() {
    if (!cubeInitialized) return;
    cubeY += (targetCubeY - cubeY) * 0.08;
    cube.position.y = cubeY;
    renderer.render(scene, camera);
}

/* ============================================
   4. CANVAS & BLOB UTILS
   ============================================ */
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    if (devCanvas) {
        devCanvas.width = window.innerWidth;
        devCanvas.height = window.innerHeight;
    }
    if (sdkCanvas) {
        sdkCanvas.width = window.innerWidth;
        sdkCanvas.height = window.innerHeight;
    }
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
});

function getAbstractPoints(centerX, centerY, time, radiusMultiplier = 1) {
    const points = [];
    for (let i = 0; i < config.blob.numPoints; i++) {
        const angle = (i / config.blob.numPoints) * Math.PI * 2;
        const n1 = noise.noise(Math.cos(angle) * config.blob.noiseScale + time * 1.5, Math.sin(angle) * config.blob.noiseScale + time * 1.5);
        const radius = (config.blob.baseRadius + n1 * config.blob.radiusVariation) * radiusMultiplier;
        points.push({ x: centerX + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius });
    }
    return points;
}

function drawBlobShape(points) {
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

function drawSection2Text() {
    if (scrollProgress < 0.8) return;
    const fadeAlpha = Math.min((scrollProgress - 0.8) / 0.2, 1);
    const lines = section2Config.text.content.split('\n');
    const fontSize = Math.min(canvas.width * 0.12, 120);
    const lineHeight = fontSize * 1.1;
    const startY = canvas.height / 2 - (lines.length - 1) * lineHeight / 2;
    const cardRect = featureCard.getBoundingClientRect();

    ctx.save();
    ctx.globalAlpha = fadeAlpha;
    ctx.font = `900 ${fontSize}px Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // 1. Base Layer (Normal)
    ctx.fillStyle = section2Config.text.normalColor;
    lines.forEach((line, i) => ctx.fillText(line, canvas.width / 2, startY + i * lineHeight));

    // 2. Clipped Layer (Inside Card)
    ctx.beginPath();
    ctx.rect(cardRect.left, cardRect.top, cardRect.width, cardRect.height);
    ctx.clip();
    ctx.fillStyle = section2Config.text.intersectColor;
    lines.forEach((line, i) => ctx.fillText(line, canvas.width / 2, startY + i * lineHeight));
    ctx.restore();
}

/* ============================================
   NEW: DEVELOPER SLIDE RENDERING (STATE 4)
   ============================================ */
function getDevBlobPoints(centerX, centerY) {
    return getAbstractPoints(centerX, centerY, time, 1);
}

function drawDevBlobPath(points) {
    dctx.beginPath();
    dctx.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length; i++) {
        const current = points[i];
        const next = points[(i + 1) % points.length];
        const afterNext = points[(i + 2) % points.length];
        const cp1x = current.x + (next.x - current.x) * config.blob.smoothness;
        const cp1y = current.y + (next.y - current.y) * config.blob.smoothness;
        const cp2x = next.x - (afterNext.x - next.x) * config.blob.smoothness * 0.3;
        const cp2y = next.y - (afterNext.y - next.y) * config.blob.smoothness * 0.3;
        dctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, next.x, next.y);
    }
    dctx.closePath();
}

function applyDevFilmGrain() {
    const imageData = dctx.getImageData(0, 0, devCanvas.width, devCanvas.height);
    const pixels = imageData.data;
    
    for (let i = 0; i < pixels.length; i += 4) {
        const noiseVal = (Math.random() - 0.5) * 0.2 * 255;
        pixels[i] += noiseVal;
        pixels[i + 1] += noiseVal;
        pixels[i + 2] += noiseVal;
    }
    
    dctx.putImageData(imageData, 0, 0);
}

function drawDeveloperContent() {
    if (!dctx || animationState !== 'dev-view') return;
    dctx.clearRect(0, 0, devCanvas.width, devCanvas.height);
    
    const blobX = devCanvas.width / 2;
    const blobY = devCanvas.height / 2;
    const points = getDevBlobPoints(blobX, blobY);

    const introAlpha = Math.min((scrollProgress - 2.5) * 2, 1);
    dctx.globalAlpha = introAlpha;

    // Draw the blob FIRST (behind everything)
    dctx.save();
    drawDevBlobPath(points);
    
    // Fill blob with metallic gradient
    const grad = dctx.createRadialGradient(blobX - 80, blobY - 80, 0, blobX, blobY, config.blob.baseRadius * 1.5);
    grad.addColorStop(0, config.blob.metallic.highlight + 'B3');
    grad.addColorStop(0.3, config.blob.metallic.midtone + 'B3');
    grad.addColorStop(0.6, config.blob.metallic.shadow + 'B3');
    grad.addColorStop(1, config.blob.metallic.deep + 'B3');
    dctx.fillStyle = grad;
    dctx.fill();
    
    // Add glow
    dctx.shadowBlur = 40;
    dctx.shadowColor = config.blob.metallic.midtone;
    dctx.fill();
    dctx.shadowBlur = 0;
    dctx.restore();
    
    // Apply film grain effect - same intensity as first page (0.2)
    applyDevFilmGrain();
}

/* ============================================
   SDK SLIDE RENDERING (STATE 5 - Single Card)
   ============================================ */
function getSDKBlobPoints(centerX, centerY) {
    return getAbstractPoints(centerX, centerY, time, 1);
}

function drawSDKBlobPath(points) {
    sctx.beginPath();
    sctx.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length; i++) {
        const current = points[i];
        const next = points[(i + 1) % points.length];
        const afterNext = points[(i + 2) % points.length];
        const cp1x = current.x + (next.x - current.x) * config.blob.smoothness;
        const cp1y = current.y + (next.y - current.y) * config.blob.smoothness;
        const cp2x = next.x - (afterNext.x - next.x) * config.blob.smoothness * 0.3;
        const cp2y = next.y - (afterNext.y - next.y) * config.blob.smoothness * 0.3;
        sctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, next.x, next.y);
    }
    sctx.closePath();
}

function applySDKFilmGrain() {
    const imageData = sctx.getImageData(0, 0, sdkCanvas.width, sdkCanvas.height);
    const pixels = imageData.data;
    
    for (let i = 0; i < pixels.length; i += 4) {
        const noiseVal = (Math.random() - 0.5) * 0.2 * 255;
        pixels[i] += noiseVal;
        pixels[i + 1] += noiseVal;
        pixels[i + 2] += noiseVal;
    }
    
    sctx.putImageData(imageData, 0, 0);
}

function drawSDKContent() {
    if (!sctx || animationState !== 'sdk-view') return;
    sctx.clearRect(0, 0, sdkCanvas.width, sdkCanvas.height);
    
    const blobX = sdkCanvas.width / 2;
    const blobY = sdkCanvas.height / 2;
    const points = getSDKBlobPoints(blobX, blobY);

    const introAlpha = Math.min((scrollProgress - 3.5) * 2, 1);
    sctx.globalAlpha = introAlpha;

    // Draw the blob FIRST (behind everything)
    sctx.save();
    drawSDKBlobPath(points);
    
    // Fill blob with metallic gradient
    const grad = sctx.createRadialGradient(blobX - 80, blobY - 80, 0, blobX, blobY, config.blob.baseRadius * 1.5);
    grad.addColorStop(0, config.blob.metallic.highlight + 'B3');
    grad.addColorStop(0.3, config.blob.metallic.midtone + 'B3');
    grad.addColorStop(0.6, config.blob.metallic.shadow + 'B3');
    grad.addColorStop(1, config.blob.metallic.deep + 'B3');
    sctx.fillStyle = grad;
    sctx.fill();
    
    // Add glow
    sctx.shadowBlur = 40;
    sctx.shadowColor = config.blob.metallic.midtone;
    sctx.fill();
    sctx.shadowBlur = 0;
    sctx.restore();
    
    // Apply film grain effect
    applySDKFilmGrain();
}

/* ============================================
   5. SCROLL HANDLING
   ============================================ */
window.addEventListener('scroll', () => {
    const scroll = window.scrollY;
    const vh = window.innerHeight;
    scrollProgress = scroll / vh;
    
    const dots = document.querySelectorAll('.scroll-dot');
    dots.forEach(dot => dot.classList.remove('active'));

    const gridSection = document.getElementById('grid-section');
    const devSection = document.getElementById('developer-slide');
    const sdkSection = document.getElementById('sdk-slide');

    if (scrollProgress < 0.3) {
        animationState = 'hero';
        cubeContainer.classList.remove('active');
        if (dots[0]) dots[0].classList.add('active');
    } else if (scrollProgress < 0.5) {
        animationState = 'shrinking';
        cubeContainer.classList.remove('active');
    } else if (scrollProgress < 0.8) {
        animationState = 'cube-spawn';
        if (!cubeInitialized) initThreeJS();
        cubeContainer.classList.add('active');
        if (dots[1]) dots[1].classList.add('active');
    } else if (scrollProgress < 1.5) {
        animationState = 'cube-interactive';
        cubeContainer.classList.add('active');
        if (dots[1]) dots[1].classList.add('active');
    } else if (scrollProgress < 2.5) {
        animationState = 'grid-view';
        if (dots[2]) dots[2].classList.add('active');
    } else if (scrollProgress < 3.5) {
        animationState = 'dev-view';
        if (dots[3]) dots[3].classList.add('active');
    } else {
        animationState = 'sdk-view';
        if (dots[4]) dots[4].classList.add('active');
    }

    // Dynamic Visibility Logic
    if (ctaButton) ctaButton.style.opacity = Math.max(0, 1 - scrollProgress * 3);
    
    if (featureCard) {
        const cardShow = (scrollProgress > 0.8 && scrollProgress < 1.5);
        featureCard.style.opacity = cardShow ? Math.min((scrollProgress - 0.8) / 0.2, 1) : 0;
        featureCard.style.pointerEvents = cardShow ? "all" : "none";
    }
    
    // Grid Transition
    if (gridSection) {
        const gridVisible = scrollProgress > 1.5 && scrollProgress < 2.5;
        gridSection.style.opacity = gridVisible ? "1" : "0";
        gridSection.style.pointerEvents = gridVisible ? "all" : "none";
    }

    // Developer Transition
    if (devSection) {
        const devVisible = scrollProgress >= 2.5 && scrollProgress < 3.5;
        devSection.style.opacity = devVisible ? "1" : "0";
        devSection.style.pointerEvents = devVisible ? "all" : "none";
        
        // Show/hide the dev cards container
        const devCardsContainer = document.getElementById('dev-cards-container');
        if (devCardsContainer) {
            if (devVisible) {
                devCardsContainer.classList.add('active');
            } else {
                devCardsContainer.classList.remove('active');
            }
        }
    }

    // SDK Transition (5th page)
    if (sdkSection) {
        const sdkVisible = scrollProgress >= 3.5;
        sdkSection.style.opacity = sdkVisible ? "1" : "0";
        sdkSection.style.pointerEvents = sdkVisible ? "all" : "none";
        
        // Show/hide the SDK form card
        const sdkFormCard = document.getElementById('sdk-form-card');
        if (sdkFormCard) {
            if (sdkVisible) {
                sdkFormCard.classList.add('active');
            } else {
                sdkFormCard.classList.remove('active');
            }
        }
    }

    // Global Canvas/Cube Visibility
    if (scrollProgress > 1.5 && scrollProgress < 2.5) {
        canvas.style.opacity = "0";
        cubeContainer.style.opacity = "0";
    } else {
        canvas.style.opacity = "1";
        cubeContainer.style.opacity = ""; 
    }
});

/* ============================================
   6. MAIN RENDER LOOP
   ============================================ */
function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    if (animationState === 'hero') {
        const textAlpha = Math.max(0, 1 - scrollProgress * 3);
        const points = getAbstractPoints(centerX, centerY, time);
        const lines = config.text.content.split('\n');
        const fs = Math.min(canvas.width * 0.12, 120);
        const lineHeight = fs * 1.1;
        const startY = centerY - (lines.length - 1) * lineHeight / 2;
        
        ctx.font = `900 ${fs}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = config.text.normalColor;
        ctx.globalAlpha = textAlpha;
        lines.forEach((l, i) => ctx.fillText(l, centerX, startY + i * lineHeight));

        ctx.save();
        drawBlobShape(points);
        ctx.clip();
        
        const grad = ctx.createRadialGradient(centerX - 80, centerY - 80, 0, centerX, centerY, config.blob.baseRadius * 1.5);
        grad.addColorStop(0, config.blob.metallic.highlight + 'B3');
        grad.addColorStop(0.3, config.blob.metallic.midtone + 'B3');
        grad.addColorStop(0.6, config.blob.metallic.shadow + 'B3');
        grad.addColorStop(1, config.blob.metallic.deep + 'B3');
        ctx.fillStyle = grad;
        drawBlobShape(points);
        ctx.fill();
        
        ctx.shadowBlur = 40;
        ctx.shadowColor = config.blob.metallic.midtone;
        ctx.fill();
        ctx.shadowBlur = 0;
        
        ctx.fillStyle = config.text.intersectColor;
        ctx.globalAlpha = textAlpha;
        lines.forEach((l, i) => ctx.fillText(l, centerX, startY + i * lineHeight));
        ctx.restore();
        
        ctx.globalAlpha = 1;
        time += config.blob.speed;
    } 
    else if (animationState === 'shrinking') {
        const shrinkFactor = Math.max(0, 1 - (scrollProgress - 0.3) / 0.2);
        const points = getAbstractPoints(centerX, centerY, time, shrinkFactor);
        ctx.fillStyle = config.blob.metallic.midtone;
        drawBlobShape(points);
        ctx.fill();
    } 
    else if (animationState === 'cube-spawn' || animationState === 'cube-interactive' || animationState === 'grid-view' || animationState === 'dev-view' || animationState === 'sdk-view') {
        if (cube) {
            const rx = (mouseY - window.innerHeight / 2) * 0.001;
            const ry = (mouseX - window.innerWidth / 2) * 0.001;
            cube.rotation.x += (rx - cube.rotation.x) * 0.1;
            cube.rotation.y += (ry - cube.rotation.y) * 0.1;
            
            // Smaller scale for background states
            const scale = (animationState === 'cube-spawn') ? Math.min((scrollProgress - 0.5) * 1.2, 0.4) : 
                          (animationState === 'dev-view' || animationState === 'sdk-view') ? 0.2 : 0.4;
            cube.scale.set(scale, scale, scale);
        }
        targetCubeY = (animationState === 'grid-view' || animationState === 'dev-view' || animationState === 'sdk-view') ? -2.5 : 0;
        
        if (animationState === 'cube-interactive') {
            drawSection2Text();
        }
    }

    if (animationState === 'dev-view') {
        drawDeveloperContent();
        time += config.blob.speed;
    }

    if (animationState === 'sdk-view') {
        drawSDKContent();
        time += config.blob.speed;
    }

    if (cubeInitialized) animateThreeJS();
    
    if (animationState === 'hero' || animationState === 'shrinking') {
        applyFilmGrain(0.2);
    } else if (animationState === 'cube-interactive') {
        applyFilmGrain(0.02);
    }
    
    requestAnimationFrame(render);
}

/* ============================================
   7. INTERACTIVE CARD EFFECTS
   ============================================ */
if (featureCard) {
    featureCard.addEventListener('mousemove', (e) => {
        const rect = featureCard.getBoundingClientRect();
        const rx = (e.clientY - rect.top - rect.height/2) / 10;
        const ry = (rect.width/2 - (e.clientX - rect.left)) / 10;
        featureCard.style.transform = `perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg)`;
    });
    featureCard.addEventListener('mouseleave', () => {
        featureCard.style.transform = 'perspective(1000px) rotateX(0) rotateY(0)';
    });
}

/* ============================================
   8. DOOMSDAY CLOCK COUNTDOWN
   ============================================ */
let doomsdayStarted = false;

function triggerDoomsdayBlast() {
    const card = document.querySelector('.doomsday-card');
    if (!card) return;
    
    const blast = document.createElement('div');
    blast.className = 'doomsday-blast';
    card.appendChild(blast);
    
    const shockwave = document.createElement('div');
    shockwave.className = 'doomsday-shockwave';
    card.appendChild(shockwave);
    
    requestAnimationFrame(() => {
        blast.classList.add('active');
        shockwave.classList.add('active');
    });
    
    setTimeout(() => {
        blast.remove();
        shockwave.remove();
    }, 1500);
}

function startDoomsdayClock() {
    if (doomsdayStarted) return;
    doomsdayStarted = true;
    
    const timerEl = document.getElementById('doomsday-timer');
    const progressEl = document.getElementById('timer-progress');
    if (!timerEl) return;
    
    let currentValue = 16;
    const interval = 1000; 
    
    const countdown = setInterval(() => {
        currentValue--;
        timerEl.textContent = currentValue;
        
        const progress = (currentValue / 16) * 100;
        if (progressEl) {
            progressEl.style.width = progress + '%';
        }
        
        if (currentValue <= 5) {
            timerEl.style.animation = 'pulse-glow 0.5s ease-in-out infinite';
        }
        
        if (currentValue <= 0) {
            clearInterval(countdown);
            timerEl.textContent = '0';
            timerEl.style.animation = 'none';
            triggerDoomsdayBlast();
            
            setTimeout(() => {
                doomsdayStarted = false;
                currentValue = 16;
                timerEl.textContent = '16';
                timerEl.style.animation = 'pulse-glow 2s ease-in-out infinite';
                if (progressEl) progressEl.style.width = '100%';
            }, 2000);
        }
    }, interval);
}

const gridObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && entry.intersectionRatio > 0.3) {
            startDoomsdayClock();
        }
    });
}, { threshold: 0.3 });

const gridSection = document.getElementById('grid-section');
if (gridSection) {
    gridObserver.observe(gridSection);
}

/* ============================================
   9. SDK TRANSACTION FORM HANDLING
   ============================================ */
const sdkForm = document.getElementById('sigil-tx-form');
const sdkError = document.getElementById('sdk-error');
const sdkStatus = document.getElementById('sdk-status');
const sdkSubmitBtn = document.getElementById('sdk-submit-btn');

function showSdkError(message) {
    if (sdkError) {
        sdkError.textContent = message;
        sdkError.classList.add('show');
    }
}

function hideSdkError() {
    if (sdkError) {
        sdkError.classList.remove('show');
    }
}

function showSdkStatus(message, isSuccess = false) {
    if (sdkStatus) {
        sdkStatus.textContent = message;
        sdkStatus.classList.add('show');
        if (isSuccess) {
            sdkStatus.classList.add('success');
        } else {
            sdkStatus.classList.remove('success');
        }
    }
}

function hideSdkStatus() {
    if (sdkStatus) {
        sdkStatus.classList.remove('show');
        sdkStatus.classList.remove('success');
    }
}

function checkMetaMask() {
    if (typeof window.ethereum === 'undefined') {
        showSdkError('⚠️ MetaMask not detected. Please install MetaMask extension.');
        return false;
    }
    hideSdkError();
    return true;
}

if (sdkForm) {
    sdkForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideSdkError();
        hideSdkStatus();
        
        // Check MetaMask first
        if (!checkMetaMask()) {
            return;
        }
        
        const recipient = document.getElementById('sdk-recipient').value;
        const amount = document.getElementById('sdk-amount').value;
        const message = document.getElementById('sdk-message').value || 'SIGIL Transaction';
        
        // Validate recipient address format
        if (!/^0x[a-fA-F0-9]{40}$/.test(recipient)) {
            showSdkError('⚠️ Invalid recipient address. Must be a valid Ethereum address (0x...)');
            return;
        }
        
        // Validate amount
        if (!amount || parseFloat(amount) <= 0) {
            showSdkError('⚠️ Please enter a valid amount greater than 0');
            return;
        }
        
        // Disable button and show processing
        if (sdkSubmitBtn) {
            sdkSubmitBtn.disabled = true;
            sdkSubmitBtn.textContent = '⏳ Processing...';
        }
        
        showSdkStatus('🔐 Generating SIGIL signature...');
        
        try {
            // Request account access
            await window.ethereum.request({ method: 'eth_requestAccounts' });
            showSdkStatus('✅ Wallet connected. Preparing quantum-safe transaction...');
            
            // Simulate SIGIL signature generation (in real implementation, call your API)
            await new Promise(r => setTimeout(r, 1500));
            showSdkStatus('🛡️ SIGIL signature verified! Opening MetaMask...', true);
            
        } catch (error) {
            showSdkError(`❌ Error: ${error.message}`);
        } finally {
            if (sdkSubmitBtn) {
                sdkSubmitBtn.disabled = false;
                sdkSubmitBtn.textContent = 'Sign with SIGIL & Send';
            }
        }
    });
    
    // Check MetaMask on form focus
    sdkForm.addEventListener('focusin', () => {
        checkMetaMask();
    });
}

render();