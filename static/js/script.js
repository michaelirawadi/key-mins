// Mobile menu toggle
function toggleMobileMenu() {
  const mobileNav = document.getElementById("mobile-nav");
  const menuIcon = document.getElementById("menu-icon");

  if (mobileNav.style.display === "block") {
    mobileNav.style.display = "none";
    menuIcon.className = "fas fa-bars";
  } else {
    mobileNav.style.display = "block";
    menuIcon.className = "fas fa-times";
  }
}

// Close mobile menu when clicking on a link
document.addEventListener("DOMContentLoaded", () => {
  const mobileNavLinks = document.querySelectorAll(".mobile-nav-link");

  mobileNavLinks.forEach((link) => {
    link.addEventListener("click", () => {
      const mobileNav = document.getElementById("mobile-nav");
      const menuIcon = document.getElementById("menu-icon");

      mobileNav.style.display = "none";
      menuIcon.className = "fas fa-bars";
    });
  });
});

// Smooth scroll for anchor links
document.addEventListener("DOMContentLoaded", () => {
  const links = document.querySelectorAll('a[href^="#"]');

  links.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();

      const targetId = this.getAttribute("href");
      const targetSection = document.querySelector(targetId);

      if (targetSection) {
        targetSection.scrollIntoView({
          behavior: "smooth",
        });
      }
    });
  });
});

// Add loading state to forms
document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll('form[method="POST"]');

  forms.forEach((form) => {
    form.addEventListener("submit", () => {
      const submitBtn = form.querySelector('button[type="submit"]');
      if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML =
          '<i class="fas fa-spinner fa-spin"></i> Processing...';
        submitBtn.disabled = true;

        // Re-enable after 5 seconds as fallback
        setTimeout(() => {
          submitBtn.innerHTML = originalText;
          submitBtn.disabled = false;
        }, 5000);
      }
    });
  });
});

// Add hover effects to cards
document.addEventListener("DOMContentLoaded", () => {
  const cards = document.querySelectorAll(".feature-card, .card");

  cards.forEach((card) => {
    card.addEventListener("mouseenter", function () {
      this.style.transform = "translateY(-4px)";
    });

    card.addEventListener("mouseleave", function () {
      this.style.transform = "translateY(0)";
    });
  });
});
