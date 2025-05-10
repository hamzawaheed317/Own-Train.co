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
  FiZap,
  FiFileText,
  FiShield,
  FiLayers,
  FiBarChart2,
  FiGitMerge,
} from "react-icons/fi";

const LandingPage = () => {
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

      {/* Main Content */}
      <main className={styles["main-content"]}>
        <div className={styles["content-left"]}>
          <h1 className={styles["main-heading"]}>
            Custom chat with <span className={styles["gradient-text"]}>AI</span>
          </h1>
          <p className={styles.subheading}>
            Your smart conversation assistant for work, study, digital business
            and everyday needs
          </p>
          <button
            className={`${styles["cta-button"]} ${styles["pulse-animation"]}`}
            onClick={handleGetStarted}
          >
            Get Started
          </button>

          <div className={styles["features-list"]}>
            <div
              className={`${styles["feature-item"]} ${styles["hover-scale"]}`}
            >
              <span className={styles["feature-icon"]}>⚡</span>
              <span>Instant, accurate responses</span>
            </div>
            <div
              className={`${styles["feature-item"]} ${styles["hover-scale"]}`}
            >
              <span className={styles["feature-icon"]}>🧠</span>
              <span>Advanced AI with deep knowledge</span>
            </div>
            <div
              className={`${styles["feature-item"]} ${styles["hover-scale"]}`}
            >
              <span className={styles["feature-icon"]}>🔒</span>
              <span>Enterprise-grade security</span>
            </div>
          </div>
        </div>

        <div className={styles["content-right"]}>
          <div
            className={`${styles["chatbot-preview"]} ${styles["hover-float"]}`}
          >
            <div className={styles["chatbot-header"]}>
              <div
                className={`${styles["chatbot-avatar"]} ${styles["gradient-bg"]}`}
              >
                AI
              </div>
              <div className={styles["chatbot-status"]}>
                <span className={styles["status-indicator"]}></span> Online
              </div>
            </div>
            <div className={styles["chatbot-messages"]}>
              <div className={`${styles["message"]} ${styles["bot-message"]}`}>
                Hello! How can I assist you today?
              </div>
              <div className={`${styles["message"]} ${styles["user-message"]}`}>
                Tell me about the features of this AI chatbot
              </div>
              <div className={`${styles["message"]} ${styles["bot-message"]}`}>
                Our AI provides real-time responses, supports file uploads, and
                offers multi-tenant setup for teams!
              </div>
              {typing && (
                <div className={styles["typing-indicator"]}>
                  <div className={styles["typing-dot"]}></div>
                  <div className={styles["typing-dot"]}></div>
                  <div className={styles["typing-dot"]}></div>
                </div>
              )}
            </div>
            <div className={styles["chatbot-input-area"]}>
              <input
                type="text"
                placeholder="Type your message..."
                className={styles["chatbot-input"]}
                disabled
              />
              <button
                className={`${styles["chatbot-send-button"]} ${styles["gradient-bg"]}`}
              >
                <FiSend />
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Testimonials Section */}
      <section className={styles.testimonials}>
        <h2 className={styles["section-title"]}>Trusted by Thousands</h2>
        <div className={styles["testimonial-cards"]}>
          <div
            className={`${styles["testimonial-card"]} ${styles["hover-scale"]}`}
          >
            <div className={styles["testimonial-text"]}>
              "Integrated AI chatbots trained on Pfizer’s proprietary data and
              documents. Response times are down 70%!"
            </div>
            <div className={styles["testimonial-author"]}>
              <div className={styles["author-avatar"]}>Pfizer</div>
              <div className={styles["author-info"]}>
                <div className={styles["author-name"]}>Albert Bourla</div>
                <div className={styles["author-title"]}>CEO</div>
              </div>
            </div>
          </div>

          <div
            className={`${styles["testimonial-card"]} ${styles["hover-scale"]}`}
          >
            <div className={styles["testimonial-text"]}>
              "AI-powered storytelling with chat-based slide generation. Users
              describe their idea and Tome generates polished presentations."
            </div>
            <div className={styles["testimonial-author"]}>
              <div className={styles["author-avatar"]}>Tome</div>
              <div className={styles["author-info"]}>
                <div className={styles["author-name"]}>Henri Liriani</div>
                <div className={styles["author-title"]}>CTO & Co-founder</div>
              </div>
            </div>
          </div>
          <div
            className={`${styles["testimonial-card"]} ${styles["hover-scale"]}`}
          >
            <div className={styles["testimonial-text"]}>
              "Drift offers AI-powered chatbots for sales and customer support
              automation. Chatbot helps businesses automate & customer support"
            </div>
            <div className={styles["testimonial-author"]}>
              <div className={styles["author-avatar"]}>Drift</div>
              <div className={styles["author-info"]}>
                <div className={styles["author-name"]}>David Cancel</div>
                <div className={styles["author-title"]}>CEO</div>
              </div>
            </div>
          </div>
          <div
            className={`${styles["testimonial-card"]} ${styles["hover-scale"]}`}
          >
            <div className={styles["testimonial-text"]}>
              "Businesses use Ada to build custom AI chatbots for handling FAQs,
              troubleshooting, and customer service automation."
            </div>
            <div className={styles["testimonial-author"]}>
              <div className={styles["author-avatar"]}>Ada</div>
              <div className={styles["author-info"]}>
                <div className={styles["author-name"]}> David Hariri</div>
                <div className={styles["author-title"]}>CTO</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      {/* <section className={styles["features-section"]}>
        <h2 className={styles["section-title"]}>Powerful Features</h2>
        <div className={styles["features-grid"]}>
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiClock className={styles["feature-icon"]} />
            </div>
            <h3>Real-time responses</h3>
            <p>
              Get instant answers without delays with our low-latency AI engine
            </p>
          </div>

          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiFile className={styles["feature-icon"]} />
            </div>
            <h3>File support</h3>
            <p>
              Upload PDFs, Word docs, Excel sheets and extract insights
              instantly
            </p>
          </div>

          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiUsers className={styles["feature-icon"]} />
            </div>
            <h3>Multi-tenant setup</h3>
            <p>
              Perfect for teams and organizations with role-based access control
            </p>
          </div>
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiUsers className={styles["feature-icon"]} />
            </div>
            <h3>Multi-tenant setup</h3>
            <p>
              Perfect for teams and organizations with role-based access control
            </p>
          </div>
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiUsers className={styles["feature-icon"]} />
            </div>
            <h3>Multi-tenant setup</h3>
            <p>
              Perfect for teams and organizations with role-based access control
            </p>
          </div>
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiUsers className={styles["feature-icon"]} />
            </div>
            <h3>Multi-tenant setup</h3>
            <p>
              Perfect for teams and organizations with role-based access control
            </p>
          </div>
        </div>
      </section> */}
      {/* Features Section */}
      <section className={styles["features-section"]}>
        <div className={styles["section-header"]}>
          <h2 className={styles["section-title"]}>
            Enterprise-Grade AI Capabilities
          </h2>
          <p className={styles["section-subtitle"]}>
            Cutting-edge technology designed for business transformation
          </p>
        </div>

        <div className={styles["features-grid"]}>
          {/* Feature 1 */}
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiZap className={styles["feature-icon"]} />
            </div>
            <h3>Instant AI Responses</h3>
            <p>
              Our optimized AI engine delivers sub-second response times with
              99.9% uptime, ensuring uninterrupted productivity for your team
            </p>
          </div>

          {/* Feature 2 */}
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiFileText className={styles["feature-icon"]} />
            </div>
            <h3>Advanced Document Processing</h3>
            <p>
              Extract actionable insights from PDFs, Word docs, Excel sheets,
              and PowerPoint presentations with industry-leading accuracy
            </p>
          </div>

          {/* Feature 3 */}
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiShield className={styles["feature-icon"]} />
            </div>
            <h3>Enterprise Security</h3>
            <p>
              SOC 2 Type II compliant infrastructure with end-to-end encryption,
              ensuring your data remains protected at all times
            </p>
          </div>

          {/* Feature 4 */}
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiLayers className={styles["feature-icon"]} />
            </div>
            <h3>Custom Knowledge Integration</h3>
            <p>
              Seamlessly integrate your proprietary data, SOPs, and knowledge
              bases to create a tailored AI solution for your organization
            </p>
          </div>

          {/* Feature 5 */}
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiBarChart2 className={styles["feature-icon"]} />
            </div>
            <h3>Analytics Dashboard</h3>
            <p>
              Comprehensive usage analytics and performance metrics with
              customizable reporting for data-driven decision making
            </p>
          </div>

          {/* Feature 6 */}
          <div className={`${styles["feature-card"]} ${styles["hover-float"]}`}>
            <div className={styles["feature-icon-container"]}>
              <FiGitMerge className={styles["feature-icon"]} />
            </div>
            <h3>API & Integrations</h3>
            <p>
              Robust REST API and pre-built connectors for Slack, Teams, and
              Salesforce to embed AI across your workflow
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
