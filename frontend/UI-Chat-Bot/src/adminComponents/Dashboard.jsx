import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import "./Dashboard.css";
import {
  Chart as ChartJS,
  LinearScale,
  CategoryScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  Filler,
} from "chart.js";

ChartJS.register(
  LinearScale,
  CategoryScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  Filler
);

const Dashboard = () => {
  const [users, setUsers] = useState([]);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch("http://localhost:5000/admin/all-users", {
          credentials: "include",
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        });
        if (!response.ok) {
          throw new Error("Failed to fetch users");
        }
        const data = await response.json();
        setUsers(data);
      } catch (error) {
        console.error("Error fetching users:", error);
      }
    };

    fetchUsers();
  }, []);

  // Process user data to create chart data
  const processChartData = () => {
    // Group users by month (assuming users have createdAt field)
    const monthlyCounts = {};
    const currentYear = new Date().getFullYear();

    // Initialize with 0 counts for all months
    const monthNames = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    monthNames.forEach((month) => {
      monthlyCounts[month] = 0;
    });

    // Count users per month
    users.forEach((user) => {
      if (user.createdAt) {
        const date = new Date(user.createdAt);
        if (date.getFullYear() === currentYear) {
          const month = monthNames[date.getMonth()];
          monthlyCounts[month]++;
        }
      }
    });

    // Take only the first 7 months for the chart (Jan-Jul as in your original)
    const labels = monthNames.slice(0, 7);
    const userCounts = labels.map((month) => monthlyCounts[month]);

    // Generate dummy satisfaction rates (85-95%)
    const satisfactionRates = labels.map(
      () => Math.floor(Math.random() * 11) + 85
    );

    return {
      labels,
      datasets: [
        {
          label: "User Activity",
          data: userCounts,
          borderColor: "#6366F1",
          backgroundColor: "rgba(99, 102, 241, 0.2)",
          borderWidth: 2,
          tension: 0.4,
          fill: true,
          pointBackgroundColor: "#6366F1",
          pointBorderColor: "#fff",
          pointHoverRadius: 6,
          pointHoverBorderWidth: 2,
        },
        {
          label: "Satisfaction Rate",
          data: satisfactionRates,
          borderColor: "#10B981",
          backgroundColor: "rgba(16, 185, 129, 0.2)",
          borderWidth: 2,
          tension: 0.4,
          fill: true,
          pointBackgroundColor: "#10B981",
          pointBorderColor: "#fff",
          pointHoverRadius: 6,
          pointHoverBorderWidth: 2,
          yAxisID: "y1",
        },
      ],
    };
  };

  const chartData = processChartData();

  return (
    <div className="dashboard">
      <h1>Dashboard Analytics</h1>

      <div className="metrics-container">
        <div className="metric-card">
          <div className="metric-icon">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="#6366F1"
              width="100px"
              height="100px"
            >
              <circle cx="12" cy="6" r="4" />
              <path d="M12 14c-5 0-6 3-6 3v2h12v-2s-1-3-6-3z" />
            </svg>
          </div>
          <div className="metric-content">
            <h3>Users</h3>
            <p>
              {users.length} <span className="trend-up">↑ </span>
            </p>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="#10B981"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
            </svg>
          </div>
          <div className="metric-content">
            <h3>AI Accuracy</h3>
            <p>
              95% <span className="trend-up">↑ 3%</span>
            </p>
          </div>
        </div>
      </div>

      <div className="chart-container">
        <div className="chart-header">
          <h3>Monthly Performance Metrics</h3>
          <div className="chart-legend">
            <div className="legend-item">
              <span className="legend-color user-activity"></span>
              <span>User Activity</span>
            </div>
            <div className="legend-item">
              <span className="legend-color satisfaction"></span>
              <span>Satisfaction Rate</span>
            </div>
          </div>
        </div>
        <div className="chart-wrapper">
          <Line
            data={chartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  display: false,
                },
                tooltip: {
                  backgroundColor: "#1E293B",
                  titleColor: "#F8FAFC",
                  bodyColor: "#E2E8F0",
                  borderColor: "#334155",
                  borderWidth: 1,
                  padding: 12,
                  usePointStyle: true,
                  callbacks: {
                    label: (context) => {
                      let label = context.dataset.label || "";
                      if (label) {
                        label += ": ";
                      }
                      if (context.datasetIndex === 1) {
                        label += context.raw + "%";
                      } else {
                        label += context.raw;
                      }
                      return label;
                    },
                  },
                },
              },
              scales: {
                y: {
                  beginAtZero: false,
                  min: 0, // Adjusted to start from 0
                  max: Math.max(...chartData.datasets[0].data) + 50, // Dynamic max based on data
                  grid: {
                    color: "rgba(203, 213, 225, 0.3)",
                  },
                  ticks: {
                    color: "#64748B",
                  },
                  title: {
                    display: true,
                    text: "User Activity",
                    color: "#64748B",
                  },
                },
                y1: {
                  position: "right",
                  min: 80,
                  max: 100,
                  grid: {
                    drawOnChartArea: false,
                  },
                  ticks: {
                    color: "#64748B",
                    callback: (value) => value + "%",
                  },
                  title: {
                    display: true,
                    text: "Satisfaction Rate",
                    color: "#64748B",
                  },
                },
                x: {
                  grid: {
                    color: "rgba(203, 213, 225, 0.3)",
                  },
                  ticks: {
                    color: "#64748B",
                  },
                },
              },
              interaction: {
                intersect: false,
                mode: "index",
              },
            }}
          />
        </div>
      </div>

      <div className="quick-stats">
        <p>Efficient visual representation of your prototype metrics</p>
      </div>
    </div>
  );
};

export default Dashboard;
