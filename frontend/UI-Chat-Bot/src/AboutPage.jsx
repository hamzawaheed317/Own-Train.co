import React, { useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  Text3D,
  Float,
  useGLTF,
  Stars,
} from "@react-three/drei";
import * as THREE from "three";
import "./AboutPage.css";
import { FiUsers, FiCode, FiGlobe, FiAward } from "react-icons/fi";

import { useTexture } from "@react-three/drei";

function LogoModel({ position = [0, 0, 0], scale = 1 }) {
  const meshRef = useRef();

  // ✅ Make sure final.png is inside your public folder!
  const texture = useTexture("/final.png");

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.5;
    }
  });

  return (
    <group ref={meshRef} position={position} scale={scale}>
      <mesh>
        <planeGeometry args={[2, 2]} />
        <meshBasicMaterial map={texture} transparent side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

// 3D Logo Component (replace with your actual logo model)
// function LogoModel({ position = [0, 0, 0], scale = 1 }) {
//   const group = useRef();

//   useFrame((state) => {
//     if (group.current) {
//       group.current.rotation.y = state.clock.getElapsedTime() * 0.3;
//     }
//   });

//   return (
//     <group ref={group} position={position} scale={scale}>
//       {/* Replace this with your actual logo model */}
//       <mesh>
//         <torusKnotGeometry args={[1, 0.4, 100, 16]} />
//         <meshStandardMaterial color="#4361ee" metalness={0.8} roughness={0.2} />
//       </mesh>
//     </group>
//   );
// }

// 3D About Icon Component (for team members)
function AboutIcon3D({ text = "AI", position = [0, 0, 0], color = "#4361ee" }) {
  const meshRef = useRef();
  const [hovered, setHover] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.5;
      meshRef.current.scale.x = THREE.MathUtils.lerp(
        meshRef.current.scale.x,
        hovered ? 1.2 : 1,
        0.1
      );
      meshRef.current.scale.y = THREE.MathUtils.lerp(
        meshRef.current.scale.y,
        hovered ? 1.2 : 1,
        0.1
      );
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.5}>
      <Text3D
        ref={meshRef}
        font="https://threejs.org/examples/fonts/helvetiker_regular.typeface.json"
        size={0.5}
        height={0.1}
        position={position}
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
      >
        {text}
        <meshStandardMaterial
          color={hovered ? "#4cc9f0" : color}
          emissive={hovered ? "#4cc9f0" : "#000"}
          emissiveIntensity={0.4}
          metalness={0.8}
          roughness={0.2}
        />
      </Text3D>
    </Float>
  );
}

const AboutPage = () => {
  return (
    <div className="about-page">
      {/* Hero Section */}
      <section className="about-hero">
        <div className="hero-content">
          <h1>
            About <span className="gradient-text">Own Train</span>
          </h1>
          <p className="subtitle">
            Revolutionizing conversations through artificial intelligence
          </p>
        </div>
        <div className="hero-canvas">
          <Canvas
            camera={{ position: [0, 0, 8], fov: 45 }}
            gl={{ antialias: true }}
          >
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <Stars
              radius={100}
              depth={50}
              count={2000}
              factor={4}
              fade
              speed={1}
            />
            <LogoModel position={[0, 0, 0]} scale={1.5} />
            <OrbitControls
              enableZoom={true}
              minDistance={5}
              maxDistance={15}
              autoRotate
              autoRotateSpeed={0.5}
            />
          </Canvas>
        </div>
      </section>

      {/* Mission Section */}
      <section className="mission-section">
        <div className="mission-content">
          <h2>Our Mission</h2>
          <p>
            At Own Train, we're dedicated to creating intelligent, responsive,
            and ethical AI solutions that enhance human productivity and
            creativity. Our goal is to make advanced AI accessible to everyone.
          </p>
        </div>
      </section>

      {/* Features Section */}
      <section className="about-features">
        <h2>Why Choose Us</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <FiUsers size={48} />
            </div>
            <h3>User-Centric Design</h3>
            <p>
              Built with real users in mind, our interface is intuitive and
              accessible to both tech-savvy individuals and beginners.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <FiCode size={48} />
            </div>
            <h3>Cutting-Edge Technology</h3>
            <p>
              Leveraging the latest advancements in NLP and machine learning to
              deliver accurate and contextual responses.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <FiGlobe size={48} />
            </div>
            <h3>Global Knowledge</h3>
            <p>
              Our models are trained on diverse datasets to understand and
              respond to queries from around the target Datasets.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <FiAward size={48} />
            </div>
            <h3>Proven Results</h3>
            <p>
              Trusted by thousands of users worldwide with a 98% satisfaction
              rate in independent surveys.
            </p>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="team-section">
        <h2>Meet The Team</h2>
        <div className="team-members">
          <div className="team-member">
            <div className="member-avatar">
              <Canvas>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={0.8} />
                <AboutIcon3D text="DEV" position={[0, 0, 0]} color="#4cc9f0" />
                <OrbitControls
                  enableZoom={false}
                  autoRotate
                  autoRotateSpeed={2}
                />
              </Canvas>
            </div>
            <h3>Hamza Waheed</h3>
            <p className="role">CEO & Founder</p>
            <p className="bio">
              5+ years in AI research and product development. Former lead
              engineer at TechAI Corp.
            </p>
          </div>
          <div className="team-member">
            <div className="member-avatar">
              <Canvas>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={0.8} />
                <AboutIcon3D text="AI" position={[0, 0, 0]} color="#ff758c" />
                <OrbitControls
                  enableZoom={false}
                  autoRotate
                  autoRotateSpeed={2}
                />
              </Canvas>
            </div>
            <h3>Abdul Rauf</h3>
            <p className="role">Team Leader</p>
            <p className="bio">
              5+ years in Software Development with Cutting Edge technologies.
              Specializes in natural language understanding and generation.
            </p>
          </div>
          {/* <div className="team-member">
            <div className="member-avatar">
              <Canvas>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={0.8} />
                <AboutIcon3D text="UX" position={[0, 0, 0]} color="#7f7fd5" />
                <OrbitControls
                  enableZoom={false}
                  autoRotate
                  autoRotateSpeed={2}
                />
              </Canvas>
            </div>
            <h3>Michael Chen</h3>
            <p className="role">UX/UI Designer</p>
            <p className="bio">
              Creates intuitive interfaces that make complex AI accessible to
              everyone.
            </p>
          </div> */}
        </div>
      </section>

      {/* Values Section */}
      <section className="values-section">
        <h2>Our Core Values</h2>
        <div className="values-list">
          <div className="value-item">
            <h3>Innovation</h3>
            <p>
              We constantly push boundaries to deliver cutting-edge AI
              solutions.
            </p>
          </div>
          <div className="value-item">
            <h3>Transparency</h3>
            <p>
              We believe in open communication about how our AI systems work.
            </p>
          </div>
          <div className="value-item">
            <h3>Privacy</h3>
            <p>
              Your data security is our top priority. We implement robust
              protections.
            </p>
          </div>
          <div className="value-item">
            <h3>Accessibility</h3>
            <p>
              We design our products to be usable by people of all backgrounds.
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="about-cta">
        <h2>Ready to Experience the Future of AI Chat?</h2>
        <button className="cta-button">Get Started Now</button>
      </section>
    </div>
  );
};

export default AboutPage;
