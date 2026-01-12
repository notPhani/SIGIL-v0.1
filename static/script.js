// Scroll fade effect for navbar and hero content
window.addEventListener('scroll', () => {
    const scrollPosition = window.scrollY;
    const windowHeight = window.innerHeight;
    
    // Calculate fade threshold (start fading at 20% of viewport height)
    const fadeStart = windowHeight * 0.05;
    const fadeEnd = windowHeight * 0.15;
    
    // Get elements
    const navbar = document.getElementById('navbar');
    const heroTitle = document.getElementById('hero-title');
    const heroBtn = document.getElementById('hero-btn');
    
    // Calculate opacity based on scroll position
    let opacity = 1;
    
    if (scrollPosition > fadeStart) {
        opacity = 1 - (scrollPosition - fadeStart) / (fadeEnd - fadeStart);
        opacity = Math.max(0, Math.min(1, opacity)); // Clamp between 0 and 1
    }
    
    // Apply fade
    navbar.style.opacity = opacity;
    heroTitle.style.opacity = opacity;
    heroBtn.style.opacity = opacity;
    
    // Add transform for smoother effect
    const translateY = (1 - opacity) * -30;
    navbar.style.transform = `translateY(${translateY}px)`;
    heroTitle.style.transform = `translateY(${translateY}px)`;
    heroBtn.style.transform = `translateY(${translateY}px)`;
    
    // Optionally hide completely when fully faded
    if (opacity === 0) {
        navbar.style.pointerEvents = 'none';
        heroBtn.style.pointerEvents = 'none';
    } else {
        navbar.style.pointerEvents = 'auto';
        heroBtn.style.pointerEvents = 'auto';
    }
});
