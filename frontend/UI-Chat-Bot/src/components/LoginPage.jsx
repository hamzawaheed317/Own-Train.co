import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./LoginPage.css";
import logo from "./final.png";

const LoginPage = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    telephone: "",
  });

  const [error, setError] = useState("");
  const navigate = useNavigate();

  // Auto-redirect if already logged in
  useEffect(() => {
    try {
      const user = localStorage.getItem("user");
      if (user && JSON.parse(user)) {
        navigate("/home");
      }
    } catch (err) {
      console.error("Auth parse error:", err);
      localStorage.removeItem("user"); // Clear corrupted entry
    }
  }, [navigate]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");

    try {
      const response = await fetch("http://localhost:5000/user/login", {
        // Changed endpoint
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await response.json();
      console.log("Data", data);
      if (!response.ok) {
        throw new Error(data.message || "Authentication failed");
      }

      // Store only essential data
      localStorage.setItem(
        "user",
        JSON.stringify({
          id: data.user._id,
          admin: data.user.admin,
          name: data.user.name,
          email: data.user.email,
        })
      );
      console.log("User Stored is : ", localStorage.getItem("user"));
      navigate("/home");
    } catch (err) {
      setError(err.message || "Login failed");
      console.error("Login error:", err);
    }
  };

  return (
    <div className="container">
      <div className="form-container">
        <h1 className="title">
          <span className="title-heading">Own Train</span>
          <img src={logo} alt="Own Train Logo" className="logo-img" />
        </h1>
        <h5 className="subtitle">
          Dear Customer, please enter the required information and press start
          to begin the live chat session!
        </h5>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleLogin} autoComplete="off">
          <div className="input-group">
            <input
              className="input-field"
              type="text"
              name="name"
              placeholder="Name"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </div>

          <div className="input-group">
            <input
              className="input-field"
              type="email"
              name="email"
              placeholder="Email"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>

          <div className="input-group">
            <input
              className="input-field"
              type="tel"
              name="telephone"
              placeholder="Telephone"
              value={formData.telephone}
              onChange={handleChange}
            />
          </div>

          <button type="submit" className="login-button">
            Start
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;
