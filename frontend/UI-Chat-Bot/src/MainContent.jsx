import React, { useState, useEffect } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import styles from "./LandingPage.module.css"; // Updated import

// Import icons
import {
  FiLogIn,
  FiUserPlus,
  FiInfo,
  FiMoon,
  FiSun,
  FiSend,
  FiFile,
  FiUsers,
  FiClock,
} from "react-icons/fi";

const MainContent = () => {
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [typing, setTyping] = useState(false);

  const handleGetStarted = () => {
    navigate("/login-admin");
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setTyping((prev) => !prev);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div
      className={`${styles["landing-page"]} ${
        darkMode ? styles["dark-mode"] : ""
      }`}
    >
      {/* Navigation Bar */}
      <nav className={styles.navbar}>
        <div className={styles["navbar-container"]}>
          <div className={styles["navbar-brand"]}>
            <span className={styles["gradient-text"]}>Own Train</span>
          </div>

          <div className={styles["navbar-links"]}>
            <a href="/login-admin" className={styles["nav-link"]}>
              <FiLogIn className={styles["nav-icon"]} /> Log In
            </a>
            <a href="/about" className={styles["nav-link"]}>
              <FiInfo className={styles["nav-icon"]} /> About
            </a>
            <a href="/contact-us" className={styles["nav-link"]}>
              <FiUserPlus className={styles["nav-icon"]} /> Contact us
            </a>
            <button
              className={styles["dark-mode-toggle"]}
              onClick={toggleDarkMode}
            >
              {darkMode ? <FiSun /> : <FiMoon />}
            </button>
          </div>

          <button
            className={styles["mobile-menu-button"]}
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            ☰
          </button>
        </div>

        {isMenuOpen && (
          <div className={styles["mobile-menu"]}>
            <a href="/login-admin" className={styles["mobile-nav-link"]}>
              <FiLogIn className={styles["nav-icon"]} /> Log In
            </a>
            {/* <a href="/signup" className={styles["mobile-nav-link"]}>
              <FiUserPlus className={styles["nav-icon"]} /> Sign Up
            </a> */}
            <a href="/about" className={styles["mobile-nav-link"]}>
              <FiInfo className={styles["nav-icon"]} /> About
            </a>
            <a href="/contact-us" className={styles["mobile-nav-link"]}>
              <FiUserPlus className={styles["nav-icon"]} /> Contact Us
            </a>
            <button
              className={`${styles["mobile-nav-link"]} ${styles["dark-mode-toggle"]}`}
              onClick={toggleDarkMode}
            >
              {darkMode ? <FiSun /> : <FiMoon />}
            </button>
          </div>
        )}
      </nav>

      <Outlet />

      {/* Footer */}
      <footer className={styles.footer}>
        <div className={styles["footer-content"]}>
          <div className={styles["footer-brand"]}>
            <div className={styles["navbar-brand"]}>
              <span className={styles["gradient-text"]}>Own Train</span>
            </div>
            <p className={styles["footer-tagline"]}>
              The smartest AI assistant for all your needs
            </p>
          </div>

          <div className={styles["footer-links"]}>
            <div className={styles["link-group"]}>
              <h4>Product</h4>
              <a href="/features">Features</a>
              <a href="/pricing">Pricing</a>
              <a href="/demo">Demo</a>
            </div>

            <div className={styles["link-group"]}>
              <h4>Company</h4>
              <a href="/about">About Us</a>
              <a href="/careers">Careers</a>
              <a href="/blog">Blog</a>
            </div>

            <div className={styles["link-group"]}>
              <h4>Support</h4>
              <a href="/help">Help Center</a>
              <a href="/contact">Contact Us</a>
              <a href="/privacy">Privacy</a>
            </div>
          </div>
        </div>

        <div className={styles["footer-bottom"]}>
          <p>© {new Date().getFullYear()} Own Train All rights reserved.</p>
          <div className={styles["social-links"]}>
            <a href="/twitter" aria-label="Twitter">
              𝕏
            </a>
            <a
              href="https://www.linkedin.com/company/intellie-ai/"
              aria-label="LinkedIn"
            >
              in
            </a>
            <a href="/github" aria-label="GitHub">
              ⌨️
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default MainContent;
