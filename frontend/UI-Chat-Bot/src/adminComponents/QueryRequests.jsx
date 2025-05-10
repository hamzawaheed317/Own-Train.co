import React, { useState } from "react";
import "./QueryRequests.css";

const QueryRequests = () => {
  const [queries, setQueries] = useState([
    {
      id: 1,
      query: "How to reset password?",
      status: "Resolved",
      date: "2025-04-10",
      email: "user1@example.com",
      phone: "+1234567890",
    },
    {
      id: 2,
      query: "Payment not processed",
      status: "Pending",
      date: "2025-04-09",
      email: "user2@example.com",
      phone: "+1234567891",
    },
    {
      id: 3,
      query: "Account verification",
      status: "Resolved",
      date: "2025-04-08",
      email: "user3@example.com",
      phone: "+1234567892",
    },
    {
      id: 4,
      query: "Feature request",
      status: "Pending",
      date: "2025-04-07",
      email: "user4@example.com",
      phone: "+1234567893",
    },
  ]);

  const [filter, setFilter] = useState("all");
  const [showResolveModal, setShowResolveModal] = useState(false);
  const [currentQuery, setCurrentQuery] = useState(null);
  const [responseText, setResponseText] = useState("");

  const filteredQueries =
    filter === "all"
      ? queries
      : queries.filter((q) => q.status.toLowerCase() === filter);

  const openResolveModal = (query) => {
    setCurrentQuery(query);
    setResponseText(""); // Clear previous response text
    setShowResolveModal(true);
  };

  const closeResolveModal = () => {
    setShowResolveModal(false);
    setCurrentQuery(null);
    setResponseText("");
  };

  const handleEmailClick = () => {
    if (!currentQuery || !responseText.trim()) return;

    const subject = `Regarding your query: ${currentQuery.query}`;
    const body = `Dear Customer,\n\n${responseText}\n\nBest regards,\nSupport Team`;

    // Open default email client
    window.location.href = `mailto:${
      currentQuery.email
    }?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

    // Mark as resolved after sending email
    markAsResolved();
  };

  const handleWhatsappClick = () => {
    if (!currentQuery || !responseText.trim()) return;

    const message = `${responseText}`;
    const url = `https://wa.me/${currentQuery.phone}?text=${encodeURIComponent(
      message
    )}`;
    window.open(url, "_blank");

    // Mark as resolved after opening WhatsApp
    markAsResolved();
  };

  const markAsResolved = () => {
    setQueries(
      queries.map((q) =>
        q.id === currentQuery.id ? { ...q, status: "Resolved" } : q
      )
    );
    closeResolveModal();
  };

  return (
    <div className="query-requests">
      <h1>Query Requests</h1>

      <div className="stats">
        <div className="stat-card">
          <h3>Total Queries</h3>
          <p>{queries.length}</p>
        </div>
        <div className="stat-card">
          <h3>Resolved</h3>
          <p>{queries.filter((q) => q.status === "Resolved").length}</p>
        </div>
        <div className="stat-card">
          <h3>Pending</h3>
          <p>{queries.filter((q) => q.status === "Pending").length}</p>
        </div>
      </div>

      <div className="filter-controls">
        <button
          className={filter === "all" ? "active" : ""}
          onClick={() => setFilter("all")}
        >
          All
        </button>
        <button
          className={filter === "resolved" ? "active" : ""}
          onClick={() => setFilter("resolved")}
        >
          Resolved
        </button>
        <button
          className={filter === "pending" ? "active" : ""}
          onClick={() => setFilter("pending")}
        >
          Pending
        </button>
      </div>

      <div className="queries-list">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Query</th>
              <th>Status</th>
              <th>Date</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredQueries.map((query) => (
              <tr key={query.id}>
                <td>{query.id}</td>
                <td>{query.query}</td>
                <td>
                  <span className={`status ${query.status.toLowerCase()}`}>
                    {query.status}
                  </span>
                </td>
                <td>{query.date}</td>
                <td>
                  {query.status === "Pending" && (
                    <button
                      className="resolve-btn"
                      onClick={() => openResolveModal(query)}
                    >
                      Resolve Query
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Resolve Query Modal */}
      {showResolveModal && currentQuery && (
        <div className="modal-overlay">
          <div className="resolve-modal">
            <div className="modal-header">
              <h2>Resolve Query</h2>
              <button className="close-btn" onClick={closeResolveModal}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <p>
                <strong>Query:</strong> {currentQuery.query}
              </p>

              <div className="response-input-container">
                <label htmlFor="response-text">Your Response:</label>
                <textarea
                  id="response-text"
                  className="response-textarea"
                  value={responseText}
                  onChange={(e) => setResponseText(e.target.value)}
                  placeholder="Type your response here..."
                  rows={6}
                />
              </div>

              <div className="action-buttons">
                <button
                  className="email-btn"
                  onClick={handleEmailClick}
                  disabled={!responseText.trim()}
                  style={{ backgroundColor: "purple" }}
                >
                  Send via Email
                </button>
                <button
                  className="whatsapp-btn"
                  onClick={handleWhatsappClick}
                  disabled={!responseText.trim()}
                  style={{ backgroundColor: "green" }}
                >
                  Send via WhatsApp
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QueryRequests;
