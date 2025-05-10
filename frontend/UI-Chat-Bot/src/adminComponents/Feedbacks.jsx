import React, { useState, useRef, useEffect } from "react";
import { Bar } from "react-chartjs-2";
import "./Feedbacks.css";
import {
  Chart as ChartJS,
  LinearScale,
  CategoryScale,
  BarElement,
  Tooltip,
  Legend,
  Title,
  Filler,
} from "chart.js";
import axios from "axios";

ChartJS.register(
  LinearScale,
  CategoryScale,
  BarElement,
  Tooltip,
  Legend,
  Title,
  Filler
);

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Chart Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="chart-error">
          <p>Chart couldn't be displayed. Please try refreshing.</p>
        </div>
      );
    }
    return this.props.children;
  }
}

const Feedbacks = () => {
  const [stats, setStats] = useState(null);
  const [feedbacks, setFeedbacks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("all");
  const chartRef = useRef(null);
  const trafficChartRef = useRef(null);

  useEffect(() => {
    const fetchFeedbackStats = async () => {
      try {
        const userDetails = JSON.parse(localStorage.getItem("user"));
        console.log("User in the feedback coomponent is :", userDetails);
        if (!userDetails) {
          setLoading(false);
          return;
        }
        console.log("fetch feedbacks");
        setLoading(true);
        const response = await axios.get(
          "http://localhost:5000/feedback/stats",
          {
            params: {
              email: userDetails.email || "",
              id: userDetails.id || "",
              admin: userDetails.admin || "",
            },
          }
        );

        console.log(response);
        setStats(response.data.stats);
        setFeedbacks(response.data.data || []);
        console.log(response.data.stats);
        console.log(response.data.data || []);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching feedback stats:", err);
        setError("Failed to load feedback data");
        setLoading(false);
      }
    };

    fetchFeedbackStats();
  }, []);

  // Calculate feedback distribution from actual data
  const calculateFeedbackDistribution = () => {
    if (!stats) return [0, 0];
    return [stats.likes, stats.dislikes];
  };

  const filteredFeedbacks = feedbacks.filter((f) => {
    if (filter === "all") return true;
    return filter === "positive"
      ? f.feedbackType === "like"
      : f.feedbackType === "dislike";
  });

  const feedbackDistribution = calculateFeedbackDistribution();

  const totalFeedback = stats ? stats.likes + stats.dislikes : 0;
  const positivePercentage = stats
    ? Math.round((stats.likes / totalFeedback) * 100)
    : 0;
  const averageRating = stats ? (stats.averageRating || 0).toFixed(1) : "0.0";

  // Chart data - modified for like/dislike
  const getFeedbackData = () => ({
    labels: ["Likes", "Dislikes"],
    datasets: [
      {
        label: "Feedback Count",
        data: feedbackDistribution,
        backgroundColor: [
          "rgba(46, 204, 113, 0.7)", // Green for likes
          "rgba(231, 76, 60, 0.7)", // Red for dislikes
        ],
        borderColor: ["rgba(46, 204, 113, 1)", "rgba(231, 76, 60, 1)"],
        borderWidth: 1,
        borderRadius: 6,
        borderSkipped: false,
      },
    ],
  });

  const getTrafficData = () => ({
    labels: stats ? stats : [],
    datasets: [
      {
        label: "Monthly Users",
        data: stats ? stats : [],
        backgroundColor: "rgba(155, 89, 182, 0.7)",
        borderColor: "rgba(155, 89, 182, 1)",
        borderWidth: 1,
        borderRadius: 6,
        borderSkipped: false,
      },
    ],
  });

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: "rgba(30, 41, 59, 0.95)",
        titleColor: "#f8fafc",
        bodyColor: "#e2e8f0",
        borderColor: "rgba(255, 255, 255, 0.1)",
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        displayColors: true,
        usePointStyle: true,
        callbacks: {
          label: (context) => {
            const label = context.dataset.label || "";
            const value = context.raw;
            return `${label}: ${value}`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: "rgba(226, 232, 240, 0.5)",
          drawBorder: false,
        },
        ticks: {
          color: "#64748b",
          stepSize: 1,
        },
      },
      x: {
        grid: {
          display: false,
          drawBorder: false,
        },
        ticks: {
          color: "#64748b",
        },
      },
    },
    interaction: {
      intersect: false,
      mode: "index",
    },
  };

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current) chartRef.current.update();
      if (trafficChartRef.current) trafficChartRef.current.update();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  if (loading) {
    return (
      <div className="feedbacks-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
        </div>
        <p style={{}}>Loading feedback data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="feedbacks-container">
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="feedbacks-container">
      <header className="feedbacks-header">
        <h1>User Feedback Analytics</h1>
        <p>Insights from customer interactions with our AI assistant</p>
      </header>

      <div className="metrics-grid">
        {/* <div className="metric-card primary">
          <div className="metric-content">
            <h3>User Satisfaction</h3>
            <div className="metric-value">
              {averageRating}
              <span className="rating-stars">
                {Array(Math.round(parseFloat(averageRating)))
                  .fill("★")
                  .join("")}
              </span>
            </div>
            <p className="metric-change">
              {stats && stats.ratingChange >= 0 ? "↑" : "↓"}{" "}
              {stats ? Math.abs(stats.ratingChange) : 0} from last month
            </p>
          </div>
          <div className="metric-icon">
            <svg viewBox="0 0 24 24">
              <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
            </svg>
          </div>
        </div> */}

        <div className="metric-card secondary">
          <div className="metric-content">
            <h3>Total Feedbacks</h3>
            <div className="metric-value">{totalFeedback}</div>
            <p className="metric-change">
              {/* {stats && stats.feedbackChange >= 0 ? "↑" : "↓"}{" "} */}
              {/* {stats ? Math.abs(stats.feedbackChange) : 0} from last month */}
              ↑
            </p>
          </div>
          <div className="metric-icon">
            <svg viewBox="0 0 24 24">
              <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm0 4c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm6 12H6v-1.4c0-2 4-3.1 6-3.1s6 1.1 6 3.1V19z" />
            </svg>
          </div>
        </div>

        <div className="metric-card accent">
          <div className="metric-content">
            <h3>Positive Feedback</h3>
            <div className="metric-value">{positivePercentage}%</div>
            <p className="metric-change">
              {/* {stats && stats.positiveChange >= 0 ? "↑" : "↓"}{" "} */}
              {/* {stats ? Math.abs(stats.positiveChange) : 0}% from last month */}
              ↑
            </p>
          </div>
          <div className="metric-icon">
            <svg viewBox="0 0 24 24">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
            </svg>
          </div>
        </div>
      </div>

      <div className="filter-controls">
        <button
          className={`filter-btn ${filter === "all" ? "active" : ""}`}
          onClick={() => setFilter("all")}
        >
          All Feedback
        </button>
        <button
          className={`filter-btn ${filter === "positive" ? "active" : ""}`}
          onClick={() => setFilter("positive")}
        >
          Likes
        </button>
        <button
          className={`filter-btn ${filter === "negative" ? "active" : ""}`}
          onClick={() => setFilter("negative")}
        >
          Dislikes
        </button>
      </div>

      <div className="content-grid">
        <div className="feedback-list-container">
          <h2 className="section-title">
            {filter === "all"
              ? "All Feedback"
              : filter === "positive"
              ? "Positive Feedback (Likes)"
              : "Negative Feedback (Dislikes)"}
            <span className="feedback-count">
              {filteredFeedbacks.length} results
            </span>
          </h2>

          <div className="feedback-list">
            {filteredFeedbacks.length > 0 ? (
              filteredFeedbacks.map((feedback) => (
                <div
                  key={feedback._id || feedback.responseId}
                  className="feedback-card"
                >
                  <div className="feedback-header">
                    <div className="user-info">
                      <span className="user-avatar">
                        {feedback.name?.charAt(0) || "U"}
                      </span>
                      <span className="user-name">
                        {feedback.name || "Anonymous User"}
                      </span>
                    </div>
                    <div className="feedback-meta">
                      <span
                        className={`rating ${
                          feedback.feedbackType === "like" ? "high" : "low"
                        }`}
                      >
                        {feedback.feedbackType === "like"
                          ? "👍 Like"
                          : "👎 Dislike"}
                      </span>
                      <span className="date">
                        {new Date(feedback.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  {feedback.reason && (
                    <div className="feedback-comment">
                      <p>
                        <strong>Reason:</strong> {feedback.reason}
                      </p>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="no-feedback">
                <p>No feedback found for the selected filter.</p>
              </div>
            )}
          </div>
        </div>

        <div className="charts-container">
          <div className="chart-card">
            <div className="chart-header">
              <h3>Feedback Distribution</h3>
              <div className="chart-legend">
                <div className="legend-item">
                  <span
                    className="legend-color"
                    style={{ backgroundColor: "#2ecc71" }}
                  ></span>
                  <span>Likes</span>
                </div>
                <div className="legend-item">
                  <span
                    className="legend-color"
                    style={{ backgroundColor: "#e74c3c" }}
                  ></span>
                  <span>Dislikes</span>
                </div>
              </div>
            </div>
            <div className="chart-wrapper">
              <ErrorBoundary>
                <Bar
                  ref={chartRef}
                  data={getFeedbackData()}
                  options={chartOptions}
                />
              </ErrorBoundary>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Feedbacks;
