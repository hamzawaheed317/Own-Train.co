import React, { useState, useEffect } from "react";
import { FaBars, FaTimes } from "react-icons/fa";
import logo from "./final.png";
import "./Sidebar.css";

const Sidebar = ({ activeTab, setActiveTab }) => {
  const [isMobile, setIsMobile] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const checkScreenSize = () => {
      setIsMobile(window.innerWidth < 768);
      if (window.innerWidth >= 768) {
        setSidebarOpen(true);
      } else {
        setSidebarOpen(false);
      }
    };

    checkScreenSize();
    window.addEventListener("resize", checkScreenSize);
    return () => window.removeEventListener("resize", checkScreenSize);
  }, []);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (isMobile) setSidebarOpen(false);
  };

  return (
    <>
      {isMobile && (
        <button className="mobile-menu-btn" onClick={toggleSidebar}>
          {sidebarOpen ? <FaTimes size={24} /> : <FaBars size={24} />}
        </button>
      )}

      <div className={`sidebar ${sidebarOpen ? "open" : "close"}`}>
        <div className="logo-container">
          <img src={logo} alt="logo" className="logo-img" />
          <h2 className="logo-text">Own Train</h2>
        </div>
        <nav>
          <ul className="nav-list">
            <li
              className={`nav-item ${
                activeTab === "dashboard" ? "active" : ""
              }`}
              onClick={() => handleTabChange("dashboard")}
            >
              Dashboard
            </li>
            {/* <li
              className={`nav-item ${activeTab === "queries" ? "active" : ""}`}
              onClick={() => handleTabChange("queries")}
            >
              Query Requests
            </li> */}
            <li
              className={`nav-item ${activeTab === "train" ? "active" : ""}`}
              onClick={() => handleTabChange("train")}
            >
              Train Model
            </li>
            <li
              className={`nav-item ${
                activeTab === "feedbacks" ? "active" : ""
              }`}
              onClick={() => handleTabChange("feedbacks")}
            >
              Feedbacks
            </li>
          </ul>
        </nav>
      </div>

      {isMobile && sidebarOpen && (
        <div className="sidebar-overlay" onClick={toggleSidebar} />
      )}
    </>
  );
};

export default Sidebar;
