document.addEventListener('DOMContentLoaded', () => {
    // Keep relative demo link stable if the page is opened from a nested path.
    const btd6 = document.getElementById('btd6-link');
    if (btd6 && !btd6.getAttribute('href')) {
        btd6.setAttribute('href', 'btd6/');
    }
});
