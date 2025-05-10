import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Float } from "@react-three/drei";
import * as THREE from "three";
import { FiMail, FiPhone, FiMapPin, FiSend } from "react-icons/fi";
import "./ContactPage.css";

function ContactSphere() {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.getElapsedTime() * 0.1;
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.2;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.8}>
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[1, 2]} />
        <meshStandardMaterial
          color="#4361ee"
          wireframe
          emissive="#4cc9f0"
          emissiveIntensity={0.5}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>
    </Float>
  );
}

const ContactPage = () => {
  const formRef = useRef();

  const handleSubmit = (e) => {
    e.preventDefault();
    // Add form submission logic here
    alert("Message sent successfully!");
    formRef.current.reset();
  };

  return (
    <div className="contact-page">
      {/* Hero Section */}
      <section className="contact-hero">
        <div className="hero-content">
          <h1>
            Contact <span className="gradient-text">Own Train</span>
          </h1>
          <p className="subtitle">
            We'd love to hear from you! Reach out for inquiries, support, or
            partnerships.
          </p>
        </div>
        <div className="hero-canvas">
          <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <ContactSphere />
            <OrbitControls enableZoom={false} />
          </Canvas>
        </div>
      </section>

      {/* Contact Content */}
      <div className="contact-container">
        {/* Contact Information */}
        <div className="contact-info">
          <div className="info-card">
            <div className="info-icon">
              <FiMail size={32} />
            </div>
            <h3>Email Us</h3>
            <p>support@aichatcompanion.com</p>
            <p>business@aichatcompanion.com</p>
          </div>

          <div className="info-card">
            <div className="info-icon">
              <FiPhone size={32} />
            </div>
            <h3>Call Us</h3>
            <p>+(971) 55 617 6030 (Toll-Free)</p>
            {/* <p>+() 123-4567 (International)</p> */}
          </div>

          <div className="info-card">
            <div className="info-icon">
              <FiMapPin size={32} />
            </div>
            <h3>Visit Us</h3>
            <p>Own Train HQ</p>
            <p>Deira, near city center</p>
          </div>
        </div>

        {/* Contact Form */}
        <div className="contact-form-container">
          <h2>Send Us a Message</h2>
          <form ref={formRef} onSubmit={handleSubmit} className="contact-form">
            <div className="form-group">
              <label htmlFor="name">Full Name</label>
              <input
                type="text"
                id="name"
                name="name"
                placeholder="Your name"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <input
                type="email"
                id="email"
                name="email"
                placeholder="your.email@example.com"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="subject">Subject</label>
              <select id="subject" name="subject" required>
                <option value="">Select a subject</option>
                <option value="support">Technical Support</option>
                <option value="business">Business Inquiry</option>
                <option value="feedback">Product Feedback</option>
                <option value="partnership">Partnership Opportunity</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="message">Your Message</label>
              <textarea
                id="message"
                name="message"
                rows="5"
                placeholder="Type your message here..."
                required
              ></textarea>
            </div>

            <button type="submit" className="submit-button">
              <FiSend className="button-icon" />
              Send Message
            </button>
          </form>
        </div>
      </div>

      {/* Map Section */}
      <section className="map-section">
        <h2>Our Location</h2>
        <div className="map-container">
          <iframe
            title="Company Location"
            src="https://www.google.com/maps/embed?pb=!1m10!1m8!1m3!1d57725.50433141756!2d55.316675!3d25.275832!3m2!1i1024!2i768!4f13.1!5e0!3m2!1sen!2sus!4v1745568398079!5m2!1sen!2sus"
            width="100%"
            height="450"
            style={{ border: 0 }}
            allowFullScreen=""
            loading="lazy"
          ></iframe>
        </div>
      </section>

      {/* CTA Section */}
      <section className="contact-cta">
        <h2>Need Immediate Assistance?</h2>
        <p>
          Our support team is available 24/7 to help you with any questions.
        </p>
        <div className="cta-buttons">
          <button className="cta-button primary">
            <FiPhone /> Call Now
          </button>
          <button className="cta-button secondary">
            <FiMail /> Email Support
          </button>
        </div>
      </section>
    </div>
  );
};

export default ContactPage;
