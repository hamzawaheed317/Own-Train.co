import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import LoginPage from "./components/LoginPage";
import HomePage from "./components/HomePage";
import AdminPannel from "./AdminPannel";
import LandingPage from "./LandingPage";
import AboutPage from "./AboutPage";
import ContactPage from "./ContactUsPage";
import LoginAdminPage from "./LoginAdmin";
import Dashboard from "./adminComponents/Dashboard";
import FeedbackWidget from "./components/FeedbackWidget";
import Layout from "./Layout";
import MainContent from "./MainContent";

// Protected Route Component
const ProtectedRoute = ({ element: Element }) => {
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(null);

  useEffect(() => {
    const verifyAuth = async () => {
      try {
        const response = await fetch(
          "http://localhost:5000/admin/verify-auth",
          {
            credentials: "include",
          }
        );
        setIsAuthenticated(response.ok);
        if (!response.ok) {
          navigate("/login-admin");
        }
      } catch (error) {
        console.error("Auth verification failed:", error);
        navigate("/login-admin");
      }
    };
    verifyAuth();
  }, [navigate]);

  if (isAuthenticated === null) {
    return <div className="loading-spinner"></div>;
  }

  return isAuthenticated ? <Element /> : null;
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainContent />}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact-us" element={<ContactPage />} />
        </Route>

        <Route path="/login" element={<LoginPage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/feedback" element={<FeedbackWidget />} />
        <Route path="/login-admin" element={<LoginAdminPage />} />
        <Route path="/dashboard" element={<Dashboard />} />

        <Route
          path="/admin/admin-pannel"
          element={<ProtectedRoute element={AdminPannel} />}
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
