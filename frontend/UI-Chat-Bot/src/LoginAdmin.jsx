import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./LoginAdmin.css";
import logo from "./final.png";

const LoginPage = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // Check authentication status on component mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch(
          "http://localhost:5000/admin/verify-auth",
          {
            credentials: "include",
          }
        );

        if (response.ok) {
          navigate("/admin/admin-pannel");
        }
      } catch (error) {
        console.error("Auth check failed:", error);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");

    try {
      console.log(formData);
      const response = await fetch("http://localhost:5000/admin/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      console.log("result", result);
      if (response.ok) {
        navigate("/admin/admin-pannel");
      } else {
        setError(result.message || "Login failed. Please try again.");
      }
    } catch (error) {
      console.error("Login error:", error);
      setError("An error occurred. Please try again.");
    }
  };

  if (isLoading) {
    return (
      <div className="container">
        <div className="loading-spinner"></div>
      </div>
    );
  }
  // const handleChange = (e) => {
  //   console.log("Handle change has been called");
  //   console.log("e", e);
  //   const { name, value } = e.target;
  //   setFormData((prev) => ({
  //     ...prev,
  //     [name]: value,
  //   }));
  // };

  // const handleLogin = () => {
  //   const { username, email } = formData;
  //   if (username === "Abdul Rauf" && email === "abc123@gmail.com") {
  //     navigate("/home");
  //   } else {
  //     alert("Invalid credentials");
  //   }
  // };

  return (
    <div className="container">
      <div className="form-container">
        <h1 className="title">
          <span
            className="title-heading"
            style={{ position: "relative", right: "-12px" }}
          >
            Own Train
          </span>
          <img
            src={logo}
            alt="logo"
            style={{
              width: "80px",
              position: "relative",
              top: "18px",
            }}
          />
        </h1>
        <h5 style={{ marginBottom: "10px", marginTop: "40px" }}>
          Dear Admin, please enter the required information and press login to
          deep dive the future of AI!
        </h5>
        <form action="" autoComplete="off">
          <div className="input-group">
            <input
              className="input-field"
              type="text"
              name="email"
              placeholder="Email"
              autoComplete="off"
              inputMode="email"
              spellCheck="false"
              autoCorrect="off"
              value={formData.email}
              onChange={handleChange}
            />
          </div>

          <div className="input-group">
            <input
              className="input-field"
              type="password"
              name="password"
              placeholder="Password"
              autoComplete="off"
              value={formData.password}
              onChange={handleChange}
            />
          </div>

          <button className="login-button" onClick={handleLogin}>
            Login
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;
