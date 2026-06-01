// CutFEMx Documentation Custom JavaScript

// Immediately hide "More" button to prevent flash
(function() {
    function hideMoreButton() {
        // Multiple selectors to catch the "More" button
        const selectors = [
            '.navbar-nav .btn.dropdown-toggle[aria-controls="pst-nav-more-links"]',
            '.navbar-nav .btn:contains("More")',
            '.navbar-nav .nav-item.dropdown .btn.dropdown-toggle',
            'button[data-bs-toggle="dropdown"][aria-controls*="more"]'
        ];
        
        selectors.forEach(selector => {
            try {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    if (el.textContent.includes('More')) {
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                        // Also hide the parent nav-item
                        const navItem = el.closest('.nav-item');
                        if (navItem) {
                            navItem.style.display = 'none';
                        }
                    }
                });
            } catch (e) {
                // Ignore selector errors for older browsers
            }
        });
        
        // Also check for buttons with "More" text content
        document.querySelectorAll('.navbar-nav button').forEach(btn => {
            if (btn.textContent.trim() === 'More') {
                btn.style.display = 'none';
                const navItem = btn.closest('.nav-item');
                if (navItem) {
                    navItem.style.display = 'none';
                }
            }
        });
    }
    
    // Hide immediately if DOM is already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideMoreButton);
    } else {
        hideMoreButton();
    }
    
    // Also hide on any dynamic content changes
    if (typeof MutationObserver !== 'undefined') {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    hideMoreButton();
                }
            });
        });
        
        // Start observing once DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                observer.observe(document.querySelector('.navbar-nav') || document.body, {
                    childList: true,
                    subtree: true
                });
            });
        } else {
            observer.observe(document.querySelector('.navbar-nav') || document.body, {
                childList: true,
                subtree: true
            });
        }
    }
})();

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add expand/collapse functionality to long code blocks
    document.querySelectorAll('.highlight').forEach(function(highlight) {
        const pre = highlight.querySelector('pre');
        if (pre && pre.scrollHeight > 400) {
            pre.style.maxHeight = '400px';
            pre.style.overflow = 'hidden';
            
            const expandBtn = document.createElement('button');
            expandBtn.className = 'expand-btn';
            expandBtn.innerHTML = 'Show more ▼';
            expandBtn.style.display = 'block';
            expandBtn.style.margin = '10px auto';
            expandBtn.style.background = 'var(--cutfemx-surface)';
            expandBtn.style.border = '1px solid var(--cutfemx-border)';
            expandBtn.style.borderRadius = '4px';
            expandBtn.style.padding = '8px 16px';
            expandBtn.style.cursor = 'pointer';
            
            let expanded = false;
            expandBtn.addEventListener('click', function() {
                if (expanded) {
                    pre.style.maxHeight = '400px';
                    pre.style.overflow = 'hidden';
                    expandBtn.innerHTML = 'Show more ▼';
                } else {
                    pre.style.maxHeight = 'none';
                    pre.style.overflow = 'visible';
                    expandBtn.innerHTML = 'Show less ▲';
                }
                expanded = !expanded;
            });
            
            highlight.appendChild(expandBtn);
        }
    });

    // Add table of contents highlighting
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            const id = entry.target.getAttribute('id');
            const tocLink = document.querySelector(`a[href="#${id}"]`);
            if (tocLink) {
                if (entry.isIntersecting) {
                    tocLink.classList.add('active');
                } else {
                    tocLink.classList.remove('active');
                }
            }
        });
    }, {
        rootMargin: '-20% 0px -80% 0px'
    });

    // Observe all headings
    document.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]').forEach(heading => {
        observer.observe(heading);
    });

    // Add loading animation for images (except hero images which should be immediately visible)
    document.querySelectorAll('img').forEach(function(img) {
        // Skip images in hero section - they should be immediately visible
        if (img.closest('.hero-section') || img.classList.contains('hero-logo-img') || 
            img.closest('.hero-logo') || img.closest('.hero-content')) {
            // Ensure hero images are immediately visible
            img.style.opacity = '1';
            img.style.visibility = 'visible';
            img.style.display = 'block';
            return;
        }
        
        img.addEventListener('load', function() {
            this.style.opacity = '1';
            this.style.transition = 'opacity 0.3s ease';
        });
        
        // Add fade-in effect for non-hero images
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.3s ease';
    });

    // Ensure hero logo is immediately visible after DOM is ready
    setTimeout(() => {
        const heroImages = document.querySelectorAll('.hero-section img, .hero-logo img, .hero-logo-img');
        heroImages.forEach(img => {
            img.style.opacity = '1';
            img.style.visibility = 'visible';
            img.style.display = 'block';
        });
    }, 50);

    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('input[type="search"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to close modals/search
        if (e.key === 'Escape') {
            const searchInput = document.querySelector('input[type="search"]');
            if (searchInput && document.activeElement === searchInput) {
                searchInput.blur();
            }
        }
    });

    // Add theme toggle functionality if needed
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
        }
    }

    // Add print styles
    const printButton = document.querySelector('.print-btn');
    if (printButton) {
        printButton.addEventListener('click', function() {
            window.print();
        });
    }

    // Work with PyData theme's existing sidebar toggle system
    function setupPydataNavigation() {
        console.log('Setting up PyData navigation with theme compatibility...');
        
        // The PyData theme already has a working toggle system, we just need to ensure it works properly
        const primaryToggle = document.querySelector('.primary-toggle, .pst-navbar-icon.sidebar-toggle');
        const primarySidebarCheckbox = document.querySelector('#pst-primary-sidebar-checkbox');
        const primarySidebar = document.querySelector('.bd-sidebar-primary');
        
        // Hide the toggle button on wide screens
        function updateToggleVisibility() {
            if (primaryToggle) {
                if (window.innerWidth >= 992) {
                    primaryToggle.style.display = 'none';
                } else {
                    primaryToggle.style.display = 'flex';
                }
            }
        }
        
        // Initial visibility check
        updateToggleVisibility();
        
        console.log('Elements found:');
        console.log('Toggle:', !!primaryToggle);
        console.log('Checkbox:', !!primarySidebarCheckbox);
        console.log('Sidebar:', !!primarySidebar);
        
        if (primaryToggle && primarySidebarCheckbox && primarySidebar) {
            console.log('Enhancing existing PyData sidebar functionality...');
            
            // Ensure the button is properly styled (but visibility controlled above)
            primaryToggle.style.cursor = 'pointer';
            
            // Add click handler that works with the existing system
            primaryToggle.addEventListener('click', function(e) {
                console.log('Primary toggle clicked');
                
                // Don't prevent default - let the theme handle it
                // Just add our enhancements
                setTimeout(() => {
                    const isOpen = primarySidebarCheckbox.checked;
                    console.log('Sidebar is now:', isOpen ? 'open' : 'closed');
                    
                    // Add visual feedback
                    if (isOpen) {
                        primaryToggle.classList.add('active');
                        document.body.classList.add('sidebar-open');
                    } else {
                        primaryToggle.classList.remove('active');
                        document.body.classList.remove('sidebar-open');
                    }
                }, 10);
            });
            
            // Handle overlay clicks
            const overlay = document.querySelector('.overlay-primary');
            if (overlay) {
                overlay.addEventListener('click', function() {
                    console.log('Overlay clicked');
                    // The theme should handle this, but add our cleanup
                    setTimeout(() => {
                        primaryToggle.classList.remove('active');
                        document.body.classList.remove('sidebar-open');
                    }, 10);
                });
            }
            
            // Monitor checkbox changes for visual feedback
            primarySidebarCheckbox.addEventListener('change', function() {
                const isOpen = this.checked;
                console.log('Checkbox changed, open:', isOpen);
                
                if (isOpen) {
                    primaryToggle.classList.add('active');
                    document.body.classList.add('sidebar-open');
                } else {
                    primaryToggle.classList.remove('active');
                    document.body.classList.remove('sidebar-open');
                }
            });
            
            // Handle window resize
            window.addEventListener('resize', function() {
                updateToggleVisibility(); // Update button visibility
                
                if (window.innerWidth >= 992 && primarySidebarCheckbox.checked) {
                    primarySidebarCheckbox.checked = false;
                    primaryToggle.classList.remove('active');
                    document.body.classList.remove('sidebar-open');
                    console.log('Sidebar closed due to window resize');
                }
            });
            
            console.log('PyData sidebar enhancement complete');
        } else {
            console.log('Some required elements not found, sidebar toggle may not work');
        }
    }

    // Enhanced dropdown navigation for pydata-sphinx-theme
    
    // Function to create dropdown navigation for User Guide
    function createDropdownNavigation() {
        const navbar = document.querySelector('.navbar-nav');
        if (!navbar) return;
        
        // Find the User Guide link and convert it to a dropdown
        const userGuideLink = Array.from(navbar.querySelectorAll('a')).find(
            link => link.textContent.trim() === 'User Guide'
        );
        
        if (userGuideLink) {
            const parentLi = userGuideLink.closest('li');
            if (parentLi && !parentLi.classList.contains('dropdown')) {
                parentLi.classList.add('dropdown');
                userGuideLink.classList.add('dropdown-toggle');
                userGuideLink.setAttribute('href', '#');
                userGuideLink.setAttribute('role', 'button');
                userGuideLink.setAttribute('data-bs-toggle', 'dropdown');
                userGuideLink.setAttribute('aria-expanded', 'false');
                
                // Create dropdown menu
                const dropdownMenu = document.createElement('ul');
                dropdownMenu.className = 'dropdown-menu';
                dropdownMenu.setAttribute('aria-labelledby', userGuideLink.id || 'userGuideDropdown');
                
                // Define User Guide submenu items
                const subPages = [
                    { text: 'User Guide Overview', href: 'user-guide/index.html' },
                    { text: 'Level Sets', href: 'user-guide/level-sets.html' },
                    { text: 'Element Classification', href: 'user-guide/element-classification.html' },
                    { text: 'Cut Meshes', href: 'user-guide/cut-meshes.html' },
                    { text: 'Stabilization', href: 'user-guide/stabilization.html' },
                    { text: 'Boundary Conditions', href: 'user-guide/boundary-conditions.html' },
                    { text: 'DOF Constraints', href: 'user-guide/dof-constraints.html' },
                    { text: 'Quadrature', href: 'user-guide/quadrature.html' }
                ];
                
                // Create dropdown items
                subPages.forEach(page => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.className = 'dropdown-item';
                    a.href = getCorrectPath(page.href);
                    a.textContent = page.text;
                    li.appendChild(a);
                    dropdownMenu.appendChild(li);
                });
                
                parentLi.appendChild(dropdownMenu);
                
                console.log('User Guide dropdown created successfully');
            }
        }
    }
    
    // Function to get correct relative path based on current location
    function getCorrectPath(href) {
        const currentPath = window.location.pathname;
        console.log('Current path:', currentPath, 'Original href:', href);
        
        // Check if we're in a subdirectory
        const isInSubdir = currentPath.includes('/user-guide/') || 
                          currentPath.includes('/getting-started/');
        
        // If we're in a subdirectory and the href doesn't start with ../
        if (isInSubdir && !href.startsWith('../')) {
            const correctedPath = '../' + href;
            console.log('In subdir, corrected path:', correctedPath);
            return correctedPath;
        } 
        // If we're not in a subdirectory but the href starts with ../
        else if (!isInSubdir && href.startsWith('../')) {
            const correctedPath = href.substring(3);
            console.log('Not in subdir, corrected path:', correctedPath);
            return correctedPath;
        }
        
        console.log('No path correction needed:', href);
        return href;
    }
    
    // Function to handle dropdown on hover and click
    function addDropdownBehavior() {
        const dropdowns = document.querySelectorAll('.navbar-nav .dropdown');
        
        dropdowns.forEach(dropdown => {
            const dropdownToggle = dropdown.querySelector('.dropdown-toggle');
            const dropdownMenu = dropdown.querySelector('.dropdown-menu');
            
            if (dropdownToggle && dropdownMenu) {
                let hideTimeout;
                
                // Show dropdown on hover
                dropdown.addEventListener('mouseenter', function() {
                    clearTimeout(hideTimeout);
                    dropdownMenu.classList.add('show');
                    dropdownToggle.setAttribute('aria-expanded', 'true');
                });
                
                // Hide dropdown on mouse leave with delay
                dropdown.addEventListener('mouseleave', function() {
                    hideTimeout = setTimeout(() => {
                        dropdownMenu.classList.remove('show');
                        dropdownToggle.setAttribute('aria-expanded', 'false');
                    }, 150); // Small delay to allow moving to menu items
                });
                
                // Handle click events on toggle
                dropdownToggle.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    const isOpen = dropdownMenu.classList.contains('show');
                    
                    // Close all other dropdowns
                    document.querySelectorAll('.navbar-nav .dropdown-menu.show').forEach(menu => {
                        if (menu !== dropdownMenu) {
                            menu.classList.remove('show');
                            const toggle = menu.parentElement.querySelector('.dropdown-toggle');
                            if (toggle) toggle.setAttribute('aria-expanded', 'false');
                        }
                    });
                    
                    // Toggle current dropdown
                    if (!isOpen) {
                        dropdownMenu.classList.add('show');
                        dropdownToggle.setAttribute('aria-expanded', 'true');
                    } else {
                        dropdownMenu.classList.remove('show');
                        dropdownToggle.setAttribute('aria-expanded', 'false');
                    }
                });
                
                // Ensure dropdown items are clickable
                const dropdownItems = dropdownMenu.querySelectorAll('.dropdown-item');
                dropdownItems.forEach(item => {
                    item.addEventListener('click', function(e) {
                        // Don't prevent default - let the link work normally
                        console.log('Dropdown item clicked:', this.href);
                        // Close dropdown after click
                        setTimeout(() => {
                            dropdownMenu.classList.remove('show');
                            dropdownToggle.setAttribute('aria-expanded', 'false');
                        }, 100);
                    });
                    
                    // Prevent dropdown from closing when hovering over items
                    item.addEventListener('mouseenter', function() {
                        clearTimeout(hideTimeout);
                    });
                });
            }
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.navbar-nav .dropdown')) {
                document.querySelectorAll('.navbar-nav .dropdown-menu.show').forEach(menu => {
                    menu.classList.remove('show');
                    const toggle = menu.parentElement.querySelector('.dropdown-toggle');
                    if (toggle) toggle.setAttribute('aria-expanded', 'false');
                });
            }
        });
    }
    
    // Initialize dropdown functionality
    setTimeout(() => {
        addHomeNavigation();
        fixHomeNavigation();
        createDropdownNavigation();
        addDropdownBehavior();
    }, 100);
    
    // Re-initialize on theme changes or dynamic content updates
    const navObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.target.closest('.navbar')) {
                setTimeout(() => {
                    addHomeNavigation();
                    fixHomeNavigation();
                    createDropdownNavigation();
                    addDropdownBehavior();
                }, 50);
            }
        });
    });
    
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        navObserver.observe(navbar, {
            childList: true,
            subtree: true
        });
    }

    // Initialize PyData theme navigation
    setupPydataNavigation();

    // Function to add Home navigation item and fix navigation
    function addHomeNavigation() {
        const navbar = document.querySelector('.navbar-nav');
        if (!navbar) return;
        
        // Check if Home link already exists
        const existingHome = Array.from(navbar.querySelectorAll('a')).find(
            link => link.textContent.trim() === 'Home'
        );
        
        if (!existingHome) {
            // Create Home navigation item
            const homeLi = document.createElement('li');
            homeLi.className = 'nav-item';
            
            const homeLink = document.createElement('a');
            homeLink.className = 'nav-link';
            homeLink.textContent = 'Home';
            
            // Determine the correct path to home based on current location
            const currentPath = window.location.pathname;
            const isInSubdir = currentPath.includes('/user-guide/') || 
                              currentPath.includes('/getting-started/');
            
            if (isInSubdir) {
                homeLink.setAttribute('href', '../index.html');
            } else {
                homeLink.setAttribute('href', 'index.html');
            }
            
            homeLi.appendChild(homeLink);
            
            // Insert Home as the first navigation item
            navbar.insertBefore(homeLi, navbar.firstChild);
            
            console.log('Home navigation item added:', homeLink.href);
        }
    }

    // Function to fix existing Home navigation link
    function fixHomeNavigation() {
        const navbar = document.querySelector('.navbar-nav');
        if (!navbar) return;
        
        // Find the Home link and ensure it points to the correct location
        const homeLink = Array.from(navbar.querySelectorAll('a')).find(
            link => link.textContent.trim() === 'Home'
        );
        
        if (homeLink) {
            // Determine the correct path to home based on current location
            const currentPath = window.location.pathname;
            const isInSubdir = currentPath.includes('/user-guide/') || 
                              currentPath.includes('/getting-started/');
            
            if (isInSubdir) {
                homeLink.setAttribute('href', '../index.html');
            } else {
                homeLink.setAttribute('href', 'index.html');
            }
            
            console.log('Home navigation link fixed:', homeLink.href);
        }
    }

    // Fix Home link on initial load
    fixHomeNavigation();

    // Re-fix Home link on theme changes or dynamic content updates
    const homeLinkObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.target.closest('.navbar')) {
                setTimeout(() => {
                    fixHomeNavigation();
                }, 50);
            }
        });
    });
    
    if (navbar) {
        homeLinkObserver.observe(navbar, {
            childList: true,
            subtree: true
        });
    }

    console.log('CutFEMx documentation JavaScript loaded successfully!');
});
